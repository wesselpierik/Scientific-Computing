"""Main iteration logic for time independent concentration iteration.

Group:        10
Course:       Scientific Computing

Description:
TODO:
"""

from functools import partial
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib import animation
from matplotlib.artist import Artist
from matplotlib.image import AxesImage


class BaseGrid:
    def __init__(self, grid_size: int) -> None:
        if grid_size <= 2:  # noqa: PLR2004
            size_error = "Grid size is too small for any meaningful calculation."
            raise ValueError(size_error)

        self._state = np.zeros((grid_size, grid_size))
        self._state[0, :] = 1
        self._state[1:, :] = 0
        self._sinks = np.zeros((grid_size, grid_size), dtype=bool)
        self._grid_size = grid_size

    def step(self) -> float:
        max_diff = 0
        for row, column in product(
            range(1, self._grid_size - 1),
            range(self._grid_size),
        ):
            if not self._sinks[row, column]:
                diff = self._step_cell(row, column)
                max_diff = max(max_diff, diff)
        return max_diff

    def _step_cell(self, _row: int, _column: int) -> float:
        raise NotImplementedError

    def add_sink(
        self,
        bottom_row: int,
        top_row: int,
        left_column: int,
        right_column: int,
    ) -> None:
        self._state[top_row : (bottom_row + 1), left_column : (right_column + 1)] = 0
        self._sinks[top_row : (bottom_row + 1), left_column : (right_column + 1)] = True

    @property
    def state(self) -> npt.NDArray:
        return self._state


class Jacobi(BaseGrid):
    def __init__(self, grid_size: int) -> None:
        super().__init__(grid_size)

    def step(self) -> None:
        self._result = self._state.copy()
        super().step()
        self._state = self._result.copy()

    def _step_cell(self, row: int, column: int) -> None:
        up = self._state[row - 1, column]
        down = self._state[row + 1, column]
        left = self._state[row, column - 1]
        right = self._state[row, (column + 1) % self._grid_size]
        self._result[row, column] = 0.25 * (up + down + left + right)
        return self._result[row, column] - self._state[row, column]


class GaussSeidel(BaseGrid):
    def __init__(self, grid_size: int) -> None:
        super().__init__(grid_size)

    def _step_cell(self, row: int, column: int) -> float:
        up = self._state[row - 1, column]
        down = self._state[row + 1, column]
        left = self._state[row, column - 1]
        right = self._state[row, (column + 1) % self._grid_size]
        current = self._state
        new = 0.25 * (up + down + left + right)
        diff = current - new
        self._state[row, column] = new
        return diff


class SOR(BaseGrid):
    def __init__(self, grid_size: int, omega: float) -> None:
        super().__init__(grid_size)
        self._omega = omega

    def _step_cell(self, row: int, column: int) -> float:
        up = self._state[row - 1, column]
        down = self._state[row + 1, column]
        left = self._state[row, column - 1]
        right = self._state[row, (column + 1) % self._grid_size]
        current = self._state[row, column]
        new = (
            0.25 * self._omega * (up + down + left + right)
            + (1 - self._omega) * current
        )
        diff = current - new
        self._state[row, column] = new
        return diff


def show_diffusion_step(frame: int, grid: BaseGrid, im: AxesImage) -> list[Artist]:
    grid.step()
    im.set_data(grid.state)
    return [im]


def show_diffusion(grid: BaseGrid) -> None:
    fig = plt.figure()
    axis = fig.subplots(nrows=1)
    im = axis.imshow(grid.state)
    _anim = animation.FuncAnimation(
        fig,
        partial(show_diffusion_step, grid=grid, im=im),
        100,
        interval=100,
        blit=True,
    )
    plt.show()


def main() -> None:
    jacobi = SOR(50, 1.7)
    show_diffusion(jacobi)


if __name__ == "__main__":
    main()
