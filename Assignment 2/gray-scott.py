import numpy as np
import numba
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


@numba.njit(cache=True)
def laplacian(
    state: np.ndarray,
    i: int,
    j: int,
    k: int,
    grid_size: int,
) -> float:
    return (
        state[i, (j + 1) % grid_size, k]
        + state[i, (j - 1), k]
        + state[(i + 1) % grid_size, j, k]
        + state[(i - 1), j, k]
        - 4 * state[i, j, k]
    )


@numba.njit(cache=True, fastmath=True)
def _step_cell(
    old_state: np.ndarray,
    state: np.ndarray,
    i: int,
    j: int,
    Du: float,
    Dv: float,
    f: float,
    k: float,
    dt: float,
    grid_size: int,
) -> None:
    Lu = laplacian(old_state, i, j, 0, grid_size)
    Lv = laplacian(old_state, i, j, 1, grid_size)

    reaction = old_state[i, j, 0] * old_state[i, j, 1] ** 2

    state[i, j, 0] = old_state[i, j, 0] + (
        Du * Lu - reaction + f * (1 - old_state[i, j, 0])
    ) * dt
    state[i, j, 1] = old_state[i, j, 1] + (
        Dv * Lv + reaction - (f + k) * old_state[i, j, 1]
    ) * dt


class BaseGrid:
    """Abstract base class for time-independent diffusion grid solvers.

    Provides common initialization and interface for iterative methods solving
    the steady-state concentration distribution on a 2D grid with periodic
    boundary conditions and sink regions.
    """

    def __init__(self, grid_size: int) -> None:
        """Initialize the grid with boundary conditions.

        Sets up a square grid with as boundary conditions 1 at the top and 0
        at the bottom. Starts the rest of the grid as 0.

        Args:
            grid_size: Dimension of the square grid. Must be > 2.

        Raises:
            ValueError: If grid_size <= 2, raised because smaller grids cannot
                produce meaningful solutions for the diffusion equation.

        """
        if grid_size <= 2:  # noqa: PLR2004
            size_error = (
                "Grid size is too small for any meaningful calculation."
            )
            raise ValueError(size_error)

        self._state = np.zeros((grid_size, grid_size))
        self._state[0, :] = 1
        self._state[1:, :] = 0
        self._sinks = np.zeros((grid_size, grid_size), dtype=bool)
        self._grid_size = grid_size

    def step(self) -> float:
        """Perform one iteration step of the solver.

        Must be implemented by subclasses.

        Returns:
            Maximum difference between old and new values in the iteration,
            used for convergence detection.

        Raises:
            NotImplementedError: This is an abstract method.

        """
        raise NotImplementedError

    def add_sink(
        self,
        bottom_row: int,
        top_row: int,
        left_column: int,
        right_column: int,
    ) -> None:
        """Define an absorbing sink region on the grid.

        Creates a rectangular region where concentration is held at zero,
        modeling a sink or absorption boundary. The region includes boundaries.

        Args:
            bottom_row: Maximum row index (inclusive) of the sink region.
            top_row: Minimum row index (inclusive) of the sink region.
            left_column: Minimum column index (inclusive) of the sink region.
            right_column: Maximum column index (inclusive) of the sink region.

        Note:
            Coordinates are inclusive on all sides and do not take into account
            wrapping.

        """
        self._state[top_row:bottom_row + 1, left_column:right_column + 1] = 0
        self._sinks[top_row:bottom_row + 1, left_column:right_column + 1] = True

    @property
    def state(self) -> np.ndarray:
        """Get the current concentration state of the grid.

        Returns:
            2D numpy array with concentration values at each grid point.

        """
        return self._state

    @property
    def grid_size(self) -> int:
        """Get the size of the square grid."""
        return self._grid_size


class GrayScott(BaseGrid):
    def __init__(
        self,
        grid_size: int,
        f: float,
        k: float,
        square_size: int,
        Du=0.16,
        Dv=0.08,
    ) -> None:
        super().__init__(grid_size)
        self._new_state = np.zeros((grid_size, grid_size, 2))
        self._old_state = np.zeros((grid_size, grid_size, 2))
        # Concentration of U is 0.5 everywhere
        self._old_state[:, :, 0] = 0.5
        # Concentration of V is 0 everywhere,
        # except for a small square in the middle
        middle = grid_size // 2
        half_square = square_size // 2
        self._old_state[
            middle - half_square:middle + half_square,
            middle - half_square:middle + half_square,
            1,
        ] = 0.25 + np.random.rand(square_size, square_size) * 0.1
        self._Du = Du
        self._Dv = Dv
        self._f = f
        self._k = k

    @property
    def state(self) -> np.ndarray:
        return self._new_state

    def step(self) -> float:
        diff = self.numba_step(
            self._old_state,
            self._new_state,
            self._Du,
            self._Dv,
            self._f,
            self._k,
            1.0,
            self.grid_size,
        )
        self._old_state, self._new_state = self._new_state, self._old_state

        return diff

    @staticmethod
    @numba.njit(cache=True, fastmath=True, parallel=True)
    def numba_step(
        old_state: np.ndarray,
        new_state: np.ndarray,
        Du: float,
        Dv: float,
        f: float,
        k: float,
        dt: float,
        grid_size: int,
    ) -> float:
        max_diff = 0.0
        for i in numba.prange(grid_size):
            local_max = 0.0
            for j in range(grid_size):
                _step_cell(
                    old_state,
                    new_state,
                    i,
                    j,
                    Du,
                    Dv,
                    f,
                    k,
                    dt,
                    grid_size,
                )
                diff_u = abs(new_state[i, j, 0] - old_state[i, j, 0])
                diff_v = abs(new_state[i, j, 1] - old_state[i, j, 1])
                cell_diff = diff_u if diff_u > diff_v else diff_v
                if cell_diff > local_max:
                    local_max = cell_diff
            if local_max > max_diff:
                max_diff = local_max
        return max_diff


def make_update(
    fig,
    simulations_by_f,
    images_by_f,
    steps_per_frame: int,
    snapshot_step: int,
):
    snapshot_saved = False

    def update(frame_index: int):
        nonlocal snapshot_saved

        for _ in range(steps_per_frame):
            for simulations in simulations_by_f:
                for simulation in simulations:
                    simulation.step()

        for images, simulations in zip(images_by_f, simulations_by_f):
            for image, simulation in zip(images, simulations):
                image.set_data(simulation.state[:, :, 1])

        current_step = (frame_index + 1) * steps_per_frame
        fig.suptitle(
            f"Gray-Scott Model: V Concentration (Step {current_step})"
        )

        if not snapshot_saved and current_step >= snapshot_step:
            fig.savefig(
                f"snapshot_t_{snapshot_step}.png",
                dpi=300,
                bbox_inches="tight",
            )
            snapshot_saved = True

        return tuple(image for images in images_by_f for image in images)

    return update


if __name__ == "__main__":
    grid_size = 200

    fs = [0.02, 0.04, 0.06, 0.08]
    ks = [np.linspace(0.045, 0.055, 4)]  # f = 0.02
    ks += [np.linspace(0.058, 0.063, 4)]  # f = 0.04
    ks += [np.linspace(0.06, 0.065, 4)]  # f = 0.06
    ks += [np.linspace(0.06, 0.0616, 4)]  # f = 0.08

    np.random.seed(43)

    square_size = 10
    steps_per_frame = 100
    snapshot_step = 50000
    total_steps = 1000000
    total_frames = total_steps // steps_per_frame

    simulations_by_f = [
        [GrayScott(grid_size, f, k, square_size) for k in ks_small]
        for f, ks_small in zip(fs, ks)
    ]
    shared_norm = plt.Normalize(vmin=0, vmax=0.8)

    fig, axes = plt.subplots(4, 4, figsize=(10, 10), layout="compressed")

    images_by_f = []

    for i, (f, simulations, ks_small) in enumerate(
        zip(fs, simulations_by_f, ks)
    ):
        images = []

        for ax, k, simulation in zip(axes[i, :], ks_small, simulations):
            image = ax.imshow(
                simulation.state[:, :, 1],
                cmap="inferno",
                norm=shared_norm,
                animated=True,
            )
            ax.set_title(
                f"k = {k:.5f}",
                fontsize=14,
                fontweight="bold",
                pad=10,
            )
            images.append(image)

        axes[i, 0].set_ylabel(
            f"f = {f:.2f}",
            fontsize=14,
            fontweight="bold",
            rotation=0,
            labelpad=32,
            va="center",
        )
        images_by_f.append(images)

    cbar = fig.colorbar(
        images_by_f[0][0],
        ax=axes,
        shrink=0.9,
        location="right",
    )
    cbar.set_label("V concentration")

    fig.suptitle("Gray-Scott Model: V Concentration (Step 0)")

    update = make_update(
        fig,
        simulations_by_f,
        images_by_f,
        steps_per_frame,
        snapshot_step,
    )

    anim = FuncAnimation(
        fig,
        update,
        frames=total_frames,
        interval=16,
        blit=False,
    )

    plt.show()
    plt.close(fig)
