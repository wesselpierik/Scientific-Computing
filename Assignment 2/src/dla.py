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
    def __init__(
        self, grid_size: int, eta: float, *, omega: float = 1, seed: int = 43
    ) -> None:
        if grid_size <= 2:  # noqa: PLR2004
            size_error = "Grid size is too small for any meaningful calculation."
            raise ValueError(size_error)

        self._gen = np.random.Generator(np.random.PCG64(seed))

        self._growths = np.zeros((grid_size, grid_size), dtype=np.bool)
        self._candidate_list = []
        self._candidate_array = np.zeros((grid_size, grid_size), dtype=np.bool)
        self._nutrients = np.zeros((grid_size, grid_size))

        self._grid_size = grid_size
        self._omega = omega
        self._eta = eta

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
            new_column = column + column_offset % self._grid_size

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
        epsilon = 1e-12
        while self.step_nutrients() > epsilon:
            pass

    def grow_candidate(self) -> None:
        # Calculate probabilites for each candidate dependent on nutrient concentration
        concentrations = np.array(
            [self._nutrients[row, column] for row, column in self._candidate_list]
        )
        concentrations = np.pow(concentrations, self._eta)
        total_concentration = np.sum(concentrations)
        probabilities = concentrations / total_concentration

        # Grow out new candidate
        index = self._gen.choice(range(len(self._candidate_list)), 1, p=probabilities)
        row, column = self._candidate_list[index]
        self._candidate_array[row, column] = False
        self._growths[row, column] = True
        self._candidate_list[index] = self._candidate_list[-1]
        self._candidate_list.pop()

    def step(self) -> None:
        self.stabilize_nutrients()
        self.grow_candidate()

    def step_nutrients(self) -> float:
        """Perform one iteration step of the SOR diffusion solver.

        Returns:
            Maximum difference between old and new values in the iteration,
            used for convergence detection.

        """
        return self._step_nutrients(self._nutrients, self._growths, self._omega)

    @staticmethod
    @njit
    def _step_nutrients(
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
