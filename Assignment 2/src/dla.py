"""TODO.

Group:        10
Course:       Scientific Computing

Description:
-----------
TODO:
"""

import multiprocessing as mp
from functools import partial
from multiprocessing.sharedctypes import SynchronizedArray

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib import animation
from matplotlib.artist import Artist
from matplotlib.image import AxesImage
from numba import njit, prange


@njit(cache=True, parallel=True)
def _step_nutrients(
    nutrients: npt.NDArray,
    growths: npt.NDArray,
    omega: float,
    first_row: int = 1,
    last_row: int = -1,
    first_column: int = 0,
    last_column: int = -1,
) -> float:
    """Compute one SOR iteration step on the grid (JIT-compiled).

    Updates each non-growth interior point as a weighted combination of its
    current value and the Gauss-Seidel update. Operates in-place on state.

    Args:
        nutrients: Current nutrient concentration grid values.
            Modified in-place with new iteration results.
        growths: Boolean mask of absorbing sink regions.
        omega: Relaxation parameter for acceleration/deceleration.

    Returns:
        Maximum absolute change between old and new values across the grid.

    Note:
        - Uses periodic boundary conditions (horizontal wrapping).
        - Skips updates for sink regions (value stays 0).
        - Modifies state array in-place.

    """
    max_diff = 0.0
    grid_size = nutrients.shape[0]
    last_column = grid_size - 1 if last_column == -1 else last_column
    last_row = grid_size - 1 if last_row == -1 else last_row
    for row in prange(first_row, last_row):
        for column in prange(first_column, last_column + 1):
            if growths[row, column]:
                continue

            up = nutrients[row - 1, column]
            left = nutrients[row, column - 1]
            right = nutrients[row, (column + 1) % grid_size]
            current = nutrients[row, column]

            down = nutrients[row + 1, column]

            new = 0.25 * omega * (up + down + left + right) + (1 - omega) * current

            diff = abs(current - new)
            max_diff = max(max_diff, diff)

            nutrients[row, column] = new

    for column in prange(first_column, last_column + 1):
        if growths[last_row, column]:
            continue

        up = nutrients[last_row, column]
        left = nutrients[last_row, column - 1]
        right = nutrients[last_row, (column + 1) % grid_size]
        current = nutrients[last_row, column]

        new = 1 / 3 * omega * (up + left + right) + (1 - omega) * current

        diff = abs(current - new)
        max_diff = max(max_diff, diff)

        nutrients[last_row, column] = new
    return max_diff


class NutrientWorker:
    def __init__(
        self,
        id: int,
        nutrients: SynchronizedArray,
        grid_size: int,
        begin_column: int,
        end_column: int,
        growths: npt.NDArray,
        omega: float,
    ) -> None:
        self._id = id

        self._nutrients_shared = nutrients
        self._growths = growths
        self._parent = None

        self._grid_size = grid_size
        self._begin_column = begin_column
        self._end_column = end_column

        self._omega = omega
        self._epsilon1 = 0.0
        self._epsilon2 = 0.0

    def _sync(self) -> bool:
        if self._parent is None:
            error = "Not all workers are connected"
            raise ValueError(error)

        self._parent.send(max(self._epsilon1, self._epsilon2))  # pyright: ignore[reportAttributeAccessIssue]
        return self._parent.recv()  # pyright: ignore[reportAttributeAccessIssue]

    def _run_top_half(self, nutrients: npt.NDArray) -> float:
        return _step_nutrients(
            nutrients,
            self._growths,
            self._omega,
            first_row=1,
            last_row=int(self._grid_size / 2),
            first_column=self._begin_column,
            last_column=self._end_column,
        )

    def _run_bottom_half(self, nutrients: npt.NDArray) -> float:
        return _step_nutrients(
            nutrients,
            self._growths,
            self._omega,
            first_row=int(self._grid_size / 2) + 1,
            last_row=-1,
            first_column=self._begin_column,
            last_column=self._end_column,
        )

    def run(self) -> None:
        nutrients = np.frombuffer(
            self._nutrients_shared.get_obj(),
            dtype=np.float64,
        ).reshape((self._grid_size, self._grid_size))

        while True:
            if self._id % 2 == 0:
                self._epsilon1 = self._run_top_half(nutrients)

                if self._sync():
                    return

                self._epsilon2 = self._run_bottom_half(nutrients)
            else:
                self._epsilon1 = self._run_bottom_half(nutrients)

                if self._sync():
                    return

                self._epsilon2 = self._run_top_half(nutrients)

            if self._sync():
                return

    def set_parent(self, pipe: object) -> None:
        self._parent = pipe


class DLA:
    def __init__(
        self,
        grid_size: int,
        eta: float,
        *,
        workers: int = 1,
        worker_step_size: int = 1000,
        epsilon: float = 1e-10,
        omega: float = 1,
        seed: int = 43,
    ) -> None:
        if grid_size <= 2:  # noqa: PLR2004
            size_error = "Grid size is too small for any meaningful calculation."
            raise ValueError(size_error)

        self._gen = np.random.Generator(np.random.PCG64(seed))

        self._growths = np.zeros((grid_size, grid_size), dtype=np.bool)
        self._candidate_list = []
        self._candidate_array = np.zeros((grid_size, grid_size), dtype=np.bool)
        self._nutrients = np.zeros((grid_size, grid_size), dtype=np.float64)

        self._grid_size = grid_size
        self._epsilon = epsilon
        self._omega = omega
        self._eta = eta

        self._workers = workers
        self._worker_step_size = worker_step_size

        self.reset_grid()

    def reset_grid(self) -> None:
        """Reset the initial conditions for the nutrients, candidates and growths."""
        middle = int(self._grid_size / 2)
        # Empty growths and place seed at middle bottom
        self._growths[:, :] = 0
        self._growths[-1, middle] = 1

        # Add new candidates
        self._candidate_list = []
        self._candidate_array[:, :] = False
        self.add_candidates(self._grid_size - 1, middle)

        # Reset nutrient concentrations
        self._nutrients[1:, :] = 0
        self._nutrients[0, :] = 1

    def add_candidates(self, row: int, column: int) -> None:
        for row_offset, column_offset in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
            new_row = row + row_offset
            new_column = (column + column_offset) % self._grid_size

            # Don't add candidate if it falls outside, is occupied, or already added
            if (
                (new_row >= self._grid_size)
                or (self._candidate_array[new_row, new_column])
                or (self._growths[new_row, new_column])
            ):
                continue

            self._candidate_list.append((new_row, new_column))
            self._candidate_array[new_row, new_column] = True

    def stabilize_nutrients(self) -> None:
        while self.step_nutrients() > self._epsilon:
            pass

    def stabilize_nutrients_mp(self) -> None:
        nutrients_shared = mp.Array("d", self._grid_size * self._grid_size)
        # X as a Numpy array
        nutrients_repr = np.frombuffer(
            nutrients_shared.get_obj(),
            dtype=np.float64,
        ).reshape((self._grid_size, self._grid_size))

        # copy data to the shared array
        np.copyto(nutrients_repr, self._nutrients)
        width = np.ceil(self._grid_size / self._workers)
        worker_objects = [
            NutrientWorker(
                i,
                nutrients_shared,
                grid_size=self._grid_size,
                begin_column=int(i * width),
                end_column=min((i + 1) * width - 1, self._grid_size - 1),
                growths=self._growths,
                omega=self._omega,
            )
            for i in range(self._workers)
        ]
        pipes = []
        for i in range(self._workers):
            left_pipe, right_pipe = mp.Pipe(duplex=True)
            pipes.append(left_pipe)
            worker_objects[i].set_parent(right_pipe)
        current_epsilon = self._epsilon + 1
        worker_threads = [mp.Process(target=worker.run) for worker in worker_objects]
        [worker.start() for worker in worker_threads]
        current_epsilon = np.max([pipe.recv() for pipe in pipes])
        while current_epsilon > self._epsilon:
            [pipe.send(False) for pipe in pipes]
            current_epsilon = np.max([pipe.recv() for pipe in pipes])
        [pipe.send(True) for pipe in pipes]
        np.copyto(self._nutrients, nutrients_repr)

    def grow_candidate(self) -> None:
        # Calculate probabilites for each candidate dependent on nutrient concentration
        ## Absolute value, as smallest concentrations may be just below 0
        concentrations = np.abs(
            np.array(
                [self._nutrients[row, column] for row, column in self._candidate_list]
            ),
        )
        concentrations = np.pow(concentrations, self._eta)
        total_concentration = np.sum(concentrations)
        probabilities = concentrations / total_concentration

        # Grow out new candidate
        index = self._gen.choice(range(len(self._candidate_list)), 1, p=probabilities)[
            0
        ]
        row, column = self._candidate_list[index]
        self._candidate_array[row, column] = False
        self._growths[row, column] = True
        self._nutrients[row, column] = 0
        self._candidate_list[index] = self._candidate_list[-1]
        self._candidate_list.pop()
        self.add_candidates(row, column)

    def step(self) -> None:
        if self._workers <= 1:
            self.stabilize_nutrients()
        else:
            self.stabilize_nutrients_mp()
        self.grow_candidate()

    def step_nutrients(self) -> float:
        """Perform one iteration step of the SOR diffusion solver.

        Returns:
            Maximum difference between old and new values in the iteration,
            used for convergence detection.

        """
        return _step_nutrients(
            self._nutrients,
            self._growths,
            self._omega,
            first_row=1,
            last_row=self._grid_size - 1,
            first_column=0,
            last_column=self._grid_size - 1,
        )

    @property
    def nutrients(self) -> npt.NDArray:
        """Get the current nutrient concentrations.

        Returns:
            2D numpy array with concentration values at each grid point.

        """
        return self._nutrients

    @property
    def candidates(self) -> npt.NDArray:
        """Get the current candidate cells.

        Returns:
            2D numpy array with the candidates as True and the other cells as False.

        """
        return self._candidate_array

    @property
    def growths(self) -> npt.NDArray:
        """Get the current shape of the growths in the grid.

        Returns:
            2D numpy array with the growths as True and the other cells as False.

        """
        return self._growths

    @property
    def grid_size(self) -> int:
        """Get the size of the square grid."""
        return self._grid_size


# Animated grid functions:
def show_growth_step(
    _frame: int,
    dla: DLA,
    nutrients_im: AxesImage,
    growths_im: AxesImage,
    candidates_im: AxesImage,
) -> list[Artist]:
    dla.step()
    nutrients_im.set_data(dla.nutrients)
    growths_im.set_data(dla.growths)
    candidates_im.set_data(dla.candidates)
    return [nutrients_im, growths_im, candidates_im]


def show_growth(dla: DLA) -> None:
    fig = plt.figure()
    axis = fig.subplots(ncols=3)
    nutrients_im = axis[0].imshow(dla.nutrients)
    growths_im = axis[1].imshow(dla.growths)
    candidates_im = axis[2].imshow(dla.candidates)

    _anim = animation.FuncAnimation(
        fig,
        partial(
            show_growth_step,
            dla=dla,
            nutrients_im=nutrients_im,
            growths_im=growths_im,
            candidates_im=candidates_im,
        ),
        100,
        interval=1,
        blit=True,
    )
    plt.show()


def main() -> None:
    grid_size = 100
    dla = DLA(grid_size, 1, omega=1.8, workers=16)
    show_growth(dla)


if __name__ == "__main__":
    main()
