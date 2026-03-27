"""Lattice Boltzman implementation of the incompressible Navier Stokes equations.

Group:        10
Course:       Scientific Computing

Description:
Uses Lattice Boltzman to simulate the navier stokes flow through a karman street. The
street has bounce back no slip boundaries on the top and bottom walls, a free flow boundary on the right wall and a Zou He boundary on the left side.

AI usage:
Used a combination of GPT5.1 and Haiku4.5 to rewrite functions from numpy to
numba based optimizations and write comments for the functions.
"""

from functools import reduce

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from numba import njit

# Karman street
dim_x = 2.2
rohr_x = 0.2
rohr_y = 0.2
rohr_rad = 0.05
dim_y = 0.41

# Discretization parameters
ds = 0.01
nx = int(dim_x / ds) + 1
ny = int(dim_y / ds) + 1

re = 100
u = 0.12
nu_phys = 0.005
u_phys = re * nu_phys / (rohr_rad * 2)
dt = u * ds / u_phys

n = (rohr_rad * 2) // ds
nu = n * u / re
tau = 3 * nu + 0.5

w = np.array(
    [4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36],
    dtype=np.float64,
)

e = np.array(
    [
        [0, 0],
        [1, 0],
        [0, 1],
        [-1, 0],
        [0, -1],
        [1, 1],
        [-1, 1],
        [-1, -1],
        [1, -1],
    ],
)

opposite = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])
wall_opposite = np.array([0, 1, 4, 3, 2, 8, 7, 6, 5])


def _create_rohr_optimized() -> np.ndarray:
    """Create a circular obstacle (rohr) in the domain using vectorized operations.

    Generates a 2D grid representing a circular obstacle at the specified location.
    Uses numpy meshgrid and vectorized operations for efficient computation.

    Returns
    -------
    np.ndarray
        A 2D array of shape (nx, ny) with values 0.0 where the obstacle exists
        and 1.0 elsewhere. The obstacle is a circle with radius rohr_rad
        centered at (rohr_x, rohr_y).

    """
    x = np.arange(nx) * ds - rohr_x
    y = np.arange(ny) * ds - rohr_y
    xx, yy = np.meshgrid(x, y, indexing="ij")
    dist_sq = xx**2 + yy**2
    return np.where(dist_sq <= rohr_rad**2, 0.0, 1.0)


rohr = _create_rohr_optimized()
rohr_bottom = int(
    reduce(
        lambda acc, x: x[0] if x[1] != ny and acc == 0 else acc,  # pyright: ignore[reportIndexIssue]
        enumerate(np.sum(rohr, axis=1)),
        0,
    ),
)
rohr_top = int(
    reduce(
        lambda acc, x: x[0] if x[1] != ny else acc,  # pyright: ignore[reportIndexIssue]
        enumerate(np.sum(rohr, axis=1)),
        0,
    ),
)
rohr_left = int(
    reduce(
        lambda acc, x: x[0] if x[1] != nx and acc == 0 else acc,  # pyright: ignore[reportIndexIssue]
        enumerate(np.sum(rohr, axis=0)),
        0,
    ),
)
rohr_right = int(
    reduce(
        lambda acc, x: x[0] if x[1] != nx else acc,  # pyright: ignore[reportIndexIssue]
        enumerate(np.sum(rohr, axis=0)),
        0,
    ),
)


@njit(cache=True, fastmath=True)
def feq(f: np.ndarray) -> np.ndarray:
    """Compute the equilibrium distribution function using fastmath optimization.

    Calculates the equilibrium distribution for each lattice velocity direction
    based on local density and velocity. This is the target distribution toward
    which non-equilibrium distributions relax.

    Parameters
    ----------
    f : np.ndarray
        The current distribution function array of shape (nx, ny, 9) containing
        the distribution values for each of the 9 lattice directions.

    Returns
    -------
    np.ndarray
        The equilibrium distribution function array of shape (nx, ny, 9).
        Computed using the D2Q9 lattice Boltzmann equilibrium formula.

    """
    eq = np.zeros((nx, ny, 9))

    for i in range(nx):
        for j in range(ny):
            rho_val = 0.0
            ux_val = 0.0
            uy_val = 0.0

            for direction in range(9):
                f_val = f[i, j, direction]
                rho_val += f_val
                ux_val += f_val * e[direction, 0]
                uy_val += f_val * e[direction, 1]

            ux_val = ux_val / rho_val
            uy_val = uy_val / rho_val
            u_sqr = ux_val * ux_val + uy_val * uy_val

            for direction in range(9):
                uxe = ux_val * e[direction, 0]
                uye = uy_val * e[direction, 1]
                ue = uxe + uye

                eq[i, j, direction] = (
                    3.0
                    * w[direction]
                    * rho_val
                    * (1.0 / 3.0 + ue + 1.5 * ue * ue - 0.5 * u_sqr)
                )

    return eq


@njit(cache=True)
def reflections(f: np.ndarray, f_new: np.ndarray, rohr: np.ndarray) -> None:
    """Apply bounce-back boundary conditions for the circular obstacle.

    Implements no-slip boundary conditions at the rohr obstacle surface by
    reversing the velocity components of populations that attempt to enter
    the obstacle region.

    Parameters
    ----------
    f : np.ndarray
        The current distribution function array of shape (nx, ny, 9).
    f_new : np.ndarray
        The post-streaming distribution function array of shape (nx, ny, 9)
        to be modified in-place with bounce-back reflections.
    rohr : np.ndarray
        The obstacle mask array of shape (nx, ny) where 0 indicates obstacle
        and 1 indicates fluid region.

    Returns
    -------
    None
        Modifies f_new in-place with reflected distributions.

    """
    for column in range(rohr_left - 1, rohr_right + 2):
        for row in range(rohr_bottom - 1, rohr_top + 2):
            for direction in range(9):
                new_row = row + e[direction, 1]
                new_column = column + e[direction, 0]
                if rohr[new_column, new_row] == 0:
                    f_new[new_column, new_row, opposite[direction]] = f[
                        column,
                        row,
                        direction,
                    ]


@njit(cache=True, fastmath=True)
def stream(f: np.ndarray, f_new: np.ndarray) -> None:
    """Perform the streaming step with periodic boundary conditions.

    Moves distribution functions in their lattice velocity directions using
    modulo arithmetic to enforce periodic boundaries. Updates f in-place.

    Parameters
    ----------
    f : np.ndarray
        The distribution function array of shape (nx, ny, 9) to be updated
        with streamed values.
    f_new : np.ndarray
        The intermediate post-collision distribution array of shape (nx, ny, 9)
        containing the sources for streaming.

    Returns
    -------
    None
        Updates f in-place with streamed distributions.

    """
    for d in range(9):
        ex, ey = e[d, 0], e[d, 1]
        for i in range(nx):
            for j in range(ny):
                src_i = (i - ex) % nx
                src_j = (j - ey) % ny
                f[i, j, d] = f_new[src_i, src_j, d]


@njit(cache=True)
def inflow(f: np.ndarray) -> None:
    """Implement Zou-He boundary condition for inlet flow.

    Applies the Zou-He boundary condition at the left boundary (x=0) to
    maintain a specified inlet velocity profile. Reconstructs non-known
    distributions based on known distributions and the desired velocity.

    Parameters
    ----------
    f : np.ndarray
        The distribution function array of shape (nx, ny, 9) to be modified
        in-place with inlet boundary values.

    Returns
    -------
    None
        Updates f at x=0 in-place with prescribed inlet conditions.

    """
    u_profile = u
    rho_in = (
        (f[0, :, 0] + f[0, :, 2] + f[0, :, 4])
        + 2.0 * (f[0, :, 3] + f[0, :, 6] + f[0, :, 7])
    ) / (1.0 - u_profile)

    f[0, :, 1] = f[0, :, 3] + (2.0 / 3.0) * rho_in * u_profile
    f[0, :, 5] = (
        f[0, :, 7] - 0.5 * (f[0, :, 2] - f[0, :, 4]) + (1.0 / 6.0) * rho_in * u_profile
    )
    f[0, :, 8] = (
        f[0, :, 6] + 0.5 * (f[0, :, 2] - f[0, :, 4]) + (1.0 / 6.0) * rho_in * u_profile
    )


@njit(cache=True)
def outflow(f: np.ndarray) -> None:
    """Implement zero-gradient outflow boundary condition at the exit.

    Applies a zero-gradient (Neumann) boundary condition at the right boundary
    (x=nx-1) to allow flow to exit smoothly without reflections. Uses the
    distribution from the neighboring interior cell.

    Parameters
    ----------
    f : np.ndarray
        The distribution function array of shape (nx, ny, 9) to be modified
        in-place with outflow boundary values.

    Returns
    -------
    None
        Updates f at x=nx-1 in-place with zero-gradient boundary values.

    """
    f[-1, :, :] = f[-2, :, :]


@njit(cache=True, fastmath=True)
def bounce_back_walls(f: np.ndarray, f_new: np.ndarray) -> None:
    """Apply bounce-back boundary conditions at top and bottom walls.

    Implements no-slip boundary conditions at y=0 (bottom) and y=ny-1 (top)
    walls by reversing the normal velocity components of populations.

    Parameters
    ----------
    f : np.ndarray
        The current distribution function array of shape (nx, ny, 9).
    f_new : np.ndarray
        The post-collision distribution array of shape (nx, ny, 9) to be
        modified in-place with wall boundary conditions.

    Returns
    -------
    None
        Updates f_new in-place with reflected distributions at walls.

    """
    for i in range(nx):
        # Bottom wall (y = 0)
        f_new[i, 0, 2] = f[i, 0, 4]
        f_new[i, 0, 5] = f[i, 0, 7]
        f_new[i, 0, 6] = f[i, 0, 8]

        # Top wall (y = ny - 1)
        f_new[i, -1, 4] = f[i, -1, 2]
        f_new[i, -1, 7] = f[i, -1, 5]
        f_new[i, -1, 8] = f[i, -1, 6]


def step(f: np.ndarray) -> None:
    """Perform a single time step of the LBM simulation.

    Executes one complete lattice Boltzmann iteration: collision, boundary
    conditions (reflections and walls), streaming, and inlet/outlet conditions.

    Parameters
    ----------
    f : np.ndarray
        The distribution function array of shape (nx, ny, 9) representing the
        populations at each lattice point and direction. Modified in-place.

    Returns
    -------
    None
        Updates f in-place by advancing one time step.

    """
    f_new = f - (f - feq(f)) / tau

    # Reflections against the object
    reflections(f, f_new, rohr)
    bounce_back_walls(f, f_new)

    # Streaming
    stream(f, f_new)

    # Boundary conditions
    inflow(f)
    outflow(f)


def animate_flow(num_frames: int = 100, interval: int = 1) -> None:  # pyright: ignore[reportUnusedFunction]
    """Animate the incompressible Navier-Stokes flow simulation using LBM.

    Creates an interactive animation of the flow field around a circular
    obstacle showing velocity magnitude using color. Runs the simulation
    for a specified number of frames with specified update intervals.

    Parameters
    ----------
    num_frames : int, optional
        Number of animation frames to display (default is 100).
    interval : int, optional
        Delay between frames in milliseconds (default is 1).

    Returns
    -------
    None
        Displays an interactive matplotlib animation window.

    """
    directions = 9
    f = feq(np.ones((nx, ny, directions)))

    x = np.arange(nx) * ds
    y = np.arange(ny) * ds
    x, y = np.meshgrid(x, y)

    fig, ax = plt.subplots(figsize=(12, 4))
    time = 0

    def update_animation(frame: int):
        """Update the animation frame with new simulation data.

        Advances the simulation by 10 time steps and updates the visualization
        with the current velocity magnitude field and streamlines.

        Parameters
        ----------
        frame : int
            The current frame number in the animation sequence.

        Returns
        -------
        None
            Updates the matplotlib axes in-place with new visualization.

        """
        nonlocal time
        for _ in range(10):
            time += dt
            step(f)

        # Clear and redraw streamplot since StreamplotSet cannot be updated directly
        ax.clear()  # type: ignore[reportAttributeAccessIssue]
        rho = np.sum(f, axis=2)
        ux = np.sum(f * e[:, 0], axis=2) / (rho + (rho == 0))
        uy = np.sum(f * e[:, 1], axis=2) / (rho + (rho == 0))
        speed = np.sqrt(ux**2 + uy**2)
        ax.imshow(
            speed.T,
            origin="lower",
            extent=(0, dim_x, 0, dim_y),
        )

        ax.set_xlim(0, dim_x)  # type: ignore[reportAttributeAccessIssue]
        ax.set_ylim(0, dim_y)  # type: ignore[reportAttributeAccessIssue]
        ax.set_xlabel("x")  # type: ignore[reportAttributeAccessIssue]
        ax.set_ylabel("y")  # type: ignore[reportAttributeAccessIssue]
        ax.set_title(f"LBM after {time:.3f} seconds")  # type: ignore[reportAttributeAccessIssue]

    anim = animation.FuncAnimation(
        fig,
        update_animation,  # pyright: ignore[reportArgumentType]
        frames=num_frames,
        interval=interval,
        repeat=True,
    )
    plt.show()


def main() -> None:
    """Execute the main lattice Boltzmann simulation and visualization.

    Prints simulation parameters (Reynolds number and kinematic viscosity)
    and launches the interactive animation of the flow field.

    Returns
    -------
    None
        Displays simulation parameters and opens animation window.

    """
    print(f"Reynolds: {re}\nNu: {nu_phys}\nds: {ds}")
    animate_flow()


if __name__ == "__main__":
    main()
