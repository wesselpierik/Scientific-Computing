"""Main iteration logic for time independent concentration iteration.

Group:        10
Course:       Scientific Computing

Description:
-----------
TODO:

AI usage:
--------
Rewrote the code to go from regular to staticmethods to allow for easy just in
time compilation with numba.

Generated the docstrings for each function and class below, followed by manual checking.
"""

import numpy as np
import numpy.typing as npt
from numba import njit


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
            size_error = "Grid size is too small for any meaningful calculation."
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
        self._state[top_row : (bottom_row + 1), left_column : (right_column + 1)] = 0
        self._sinks[top_row : (bottom_row + 1), left_column : (right_column + 1)] = True

    @property
    def state(self) -> npt.NDArray:
        """Get the current concentration state of the grid.

        Returns:
            2D numpy array with concentration values at each grid point.

        """
        return self._state

    @property
    def grid_size(self) -> int:
        """Get the size of the square grid."""
        return self._grid_size


class Jacobi(BaseGrid):
    """Jacobi iteration method for solving steady-state diffusion equations.

    Implements simultaneous updates where each grid point is updated based on
    the values of neighbors from the previous iteration.
    """

    def step(self) -> float:
        """Perform one Jacobi iteration.

        Updates all interior grid points simultaneously using the arithmetic
        mean of their four neighbors. Sink regions remain at zero.

        Returns:
            Maximum change in any grid point, used for convergence detection.

        Note:
            Uses periodic boundary conditions in the horizontal direction.

        """
        self._result = self._state.copy()
        max_diff = self._step_cell(
            self._state, self._result, self._sinks, self._grid_size
        )
        self._state = self._result.copy()
        return max_diff

    @staticmethod
    @njit
    def _step_cell(
        state: npt.NDArray, result: npt.NDArray, sinks: npt.NDArray, grid_size: int
    ) -> float:
        """Compute one Jacobi iteration step on the grid (JIT-compiled).

        Updates each non-sink interior point as the average of its four neighbors
        using values from the current state only. Boundary values are preserved.

        Args:
            state: Current concentration grid values.
            result: Output array for new concentration values.
            sinks: Boolean mask of absorbing sink regions.
            grid_size: Dimension of the square grid.

        Returns:
            Maximum absolute change between old and new values across the grid.

        Note:
            - Compiled with Numba JIT for performance.
            - Uses periodic boundary conditions (horizontal wrapping).
            - Skips updates for sink regions (value stays 0).

        """
        max_diff = 0.0
        for row in range(1, grid_size - 1):
            for column in range(grid_size):
                if not sinks[row, column]:
                    up = state[row - 1, column]
                    down = state[row + 1, column]
                    left = state[row, column - 1]
                    right = state[row, (column + 1) % grid_size]

                    result[row, column] = 0.25 * (up + down + left + right)

                    diff = abs(result[row, column] - state[row, column])
                    max_diff = max(max_diff, diff)
        return max_diff


class GaussSeidel(BaseGrid):
    """Gauss-Seidel iteration method for solving steady-state diffusion equations.

    Implements sequential updates where each grid point uses the most recently
    computed values of neighbors.
    """

    def __init__(self, grid_size: int) -> None:
        """Initialize Gauss-Seidel solver with a grid.

        Args:
            grid_size: Dimension of the square grid. Must be > 2.

        Raises:
            ValueError: If grid_size <= 2.

        """
        super().__init__(grid_size)

    def step(self) -> float:
        """Perform one Gauss-Seidel iteration.

        Updates all interior grid points sequentially, using the most recent
        values available for neighbors. Sink regions remain at zero.

        Returns:
            Maximum change in any grid point, used for convergence detection.

        Note:
            Uses periodic boundary conditions in the horizontal direction.

        """
        return self._step_cell(self._state, self._sinks, self._grid_size)

    @staticmethod
    @njit
    def _step_cell(state: npt.NDArray, sinks: npt.NDArray, grid_size: int) -> float:
        """Compute one Gauss-Seidel iteration step on the grid (JIT-compiled).

        Updates each non-sink interior point as the average of its four neighbors,
        using newly computed values immediately after they are calculated. Operates
        in-place on the state array.

        Args:
            state: Current concentration grid values (grid_size × grid_size).
                Modified in-place with new iteration results.
            sinks: Boolean mask of absorbing sink regions (grid_size × grid_size).
            grid_size: Dimension of the square grid.

        Returns:
            Maximum absolute change between old and new values across the grid.

        Note:
            - Uses periodic boundary conditions (horizontal wrapping).
            - Skips updates for sink regions (value stays 0).
            - Modifies state array in-place.

        """
        max_diff = 0.0
        for row in range(1, grid_size - 1):
            for column in range(grid_size):
                if not sinks[row, column]:
                    up = state[row - 1, column]
                    down = state[row + 1, column]
                    left = state[row, column - 1]
                    right = state[row, (column + 1) % grid_size]

                    new = 0.25 * (up + down + left + right)

                    diff = abs(state[row, column] - new)
                    max_diff = max(max_diff, diff)

                    state[row, column] = new
        return max_diff


class SOR(BaseGrid):
    """Successive Over-Relaxation (SOR) method for solving steady-state diffusion.

    Implements an accelerated iterative method with relaxation parameter omega.
    """

    def __init__(self, grid_size: int, omega: float) -> None:
        """Initialize SOR solver with a grid and relaxation parameter.

        Args:
            grid_size: Dimension of the square grid. Must be > 2.
            omega: Relaxation parameter. Values close to 1 are conservative,
                values > 1 accelerate convergence (typically 1 < omega < 2).

        Raises:
            ValueError: If grid_size <= 2.

        """
        super().__init__(grid_size)
        self._omega = omega

    def step(self) -> float:
        """Perform one SOR iteration.

        Updates all interior grid points sequentially using relaxation,
        blending between the Gauss-Seidel update and the current value.
        Sink regions remain at zero.

        Returns:
            Maximum change in any grid point, used for convergence detection.

        Note:
            Uses periodic boundary conditions in the horizontal direction.

        """
        return self._step_cell(self._state, self._sinks, self._omega, self._grid_size)

    @staticmethod
    @njit
    def _step_cell(
        state: npt.NDArray,
        sinks: npt.NDArray,
        omega: float,
        grid_size: int,
    ) -> float:
        """Compute one SOR iteration step on the grid (JIT-compiled).

        Updates each non-sink interior point as a weighted combination of its
        current value and the Gauss-Seidel update. Operates in-place on state.

        Args:
            state: Current concentration grid values.
                Modified in-place with new iteration results.
            sinks: Boolean mask of absorbing sink regions.
            omega: Relaxation parameter for acceleration/deceleration.
            grid_size: Dimension of the square grid.

        Returns:
            Maximum absolute change between old and new values across the grid.

        Note:
            - Uses periodic boundary conditions (horizontal wrapping).
            - Skips updates for sink regions (value stays 0).
            - Modifies state array in-place.

        """
        max_diff = 0.0
        for row in range(1, grid_size - 1):
            for column in range(grid_size):
                if not sinks[row, column]:
                    up = state[row - 1, column]
                    down = state[row + 1, column]
                    left = state[row, column - 1]
                    right = state[row, (column + 1) % grid_size]
                    current = state[row, column]

                    new = (
                        0.25 * omega * (up + down + left + right)
                        + (1 - omega) * current
                    )

                    diff = abs(current - new)
                    max_diff = max(max_diff, diff)

                    state[row, column] = new
        return max_diff
