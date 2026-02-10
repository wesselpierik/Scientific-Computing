"""TODO.

Group:        10
Course:       Scientific Computing

Description:
-----------
TODO:
"""

import numpy as np
import numpy.typing as npt
from numba import njit


class DLA:
    def __init__(self, grid_size: int, *, omega: float = 1) -> None:
        if grid_size <= 2:  # noqa: PLR2004
            size_error = "Grid size is too small for any meaningful calculation."
            raise ValueError(size_error)

        self._growths = np.zeros((grid_size, grid_size))
        self._candidates = np.zeros((grid_size, grid_size))
        self._nutrients = np.zeros((grid_size, grid_size))
        self._grid_size = grid_size
        self._omega = omega

    def stabilize_nutrients(self) -> None:
        pass

    def grow_candidate(self) -> None:
        pass

    def step(self) -> float:
        """Perform one iteration step of the SOR diffusion solver.

        Must be implemented by subclasses.

        Returns:
            Maximum difference between old and new values in the iteration,
            used for convergence detection.

        """
        return self._step(self._nutrients, self._growths, self._omega)

    @staticmethod
    @njit
    def _step(
        nutrients: npt.NDArray,
        growths: npt.NDArray,
        omega: float,
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
        for row in range(1, grid_size - 1):
            for column in range(grid_size):
                if not growths[row, column]:
                    up = nutrients[row - 1, column]
                    down = nutrients[row + 1, column]
                    left = nutrients[row, column - 1]
                    right = nutrients[row, (column + 1) % grid_size]
                    current = nutrients[row, column]

                    new = (
                        0.25 * omega * (up + down + left + right)
                        + (1 - omega) * current
                    )

                    diff = abs(current - new)
                    max_diff = max(max_diff, diff)

                    nutrients[row, column] = new
        return max_diff

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
        return self._candidates

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
