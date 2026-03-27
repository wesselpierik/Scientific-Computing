"""Finite difference implementation of Navier Stokes.

Group:        10
Course:       Scientific Computing

Description:
Uses a finite difference scheme to solve the Navier stokes equations
in a Karman street.

AI usage:
Used Haiku4.5 to write comments for the functions.
"""

from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation


class Grids:
    def __init__(
        self,
        u: np.ndarray[tuple[int, int], np.dtype[np.float64]],
        v: np.ndarray[tuple[int, int], np.dtype[np.float64]],
        p: np.ndarray[tuple[int, int], np.dtype[np.float64]],
        rohr: np.ndarray[tuple[int, int], np.dtype[np.float64]],
        rohr_count: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    ) -> None:
        self.u = u
        self.p = p
        self.v = v
        self.rohr = rohr
        self.rohr_count = rohr_count


# Karman street
dim_x = 2.2
rohr_x = 0.2
rohr_y = 0.2
rohr_rad = 0.05
dim_y = 0.41

# Discretization parameters
ds = 0.004
dt = 0.0003
nx = int(dim_x / ds) + 1
ny = int(dim_y / ds) + 1

# Parameters
rho = 1
nu = 0.005
re = 125
start_velocity = re * nu / (2 * rohr_rad)


def init_grids() -> Grids:
    """Initialize the computational grids for Navier-Stokes simulation.

    Sets up velocity components (u, v), pressure (p), obstacle mask (rohr), and
    obstacle neighbor counts. Initializes u-velocity with parabolic inlet profile
    and creates a circular obstacle (cylinder) in the domain.

    Returns:
        Grids: Object containing u, v, p velocity and pressure arrays, plus
               rohr (obstacle mask) and rohr_count (neighbor counter for averaging).

    """
    # Initialisation
    u = np.zeros((ny, nx))

    # Parabolic starting velocity
    def parabole(x: np.ndarray) -> np.ndarray:
        return (1 - ((x * 2 / dim_y) - 1) ** 2) * start_velocity

    u[:, 0] = parabole(np.arange(ny) * ds)
    u[:, 0] = np.ones(ny) * start_velocity
    v = np.zeros_like(u)
    p = np.zeros_like(u)
    rohr = np.zeros_like(u)
    for row, column in product(range(ny), range(nx)):
        rohr[row, column] = int(
            (row * ds - rohr_y) ** 2 + (column * ds - rohr_x) ** 2 > rohr_rad**2,
        )
    rohr_count = (
        np.ones((ny - 2, nx - 2)) * 4
        - rohr[1:-1, 2:]
        - rohr[1:-1, :-2]
        - rohr[2:, 1:-1]
        - rohr[:-2, 1:-1]
    )
    return Grids(u, v, p, rohr, rohr_count)


def update(grids: Grids) -> None:
    """Perform one timestep of the Navier-Stokes solution.

    Updates velocity fields (u, v) and pressure (p) in-place by advancing the
    Navier-Stokes equations by one time step. Enforces boundary conditions and
    respects obstacle geometry.

    Args:
        grids: Grids object containing u, v, p velocity/pressure arrays and
               rohr obstacle mask with rohr_count for pressure averaging.

    """
    _update(grids.u, grids.p, grids.v, grids.rohr, grids.rohr_count)


def _update_p(
    u: np.ndarray,
    v: np.ndarray,
    p: np.ndarray,
    rohr: np.ndarray,
    rohr_count: np.ndarray,
) -> None:
    """Update pressure field using pressure-Poisson equation with iterative solver.

    Computes pressure gradients from velocity divergence using finite differences
    and iteratively solves the pressure-Poisson equation via Gauss-Seidel relaxation.
    Enforces Neumann boundary conditions at inlet/outlet/walls and Dirichlet at exit.
    Pressure is zeroed inside the obstacle.

    Args:
        u: x-component of velocity (ny x nx array).
        v: y-component of velocity (ny x nx array).
        p: Pressure field (ny x nx array) - modified in-place.
        rohr: Binary obstacle mask (1=fluid, 0=obstacle).
        rohr_count: Pre-computed count of fluid neighbors for pressure averaging.

    """
    v_diff_x = (v[1:-1, 2:] - v[1:-1, :-2]) / (2 * ds)
    u_diff_x = (u[1:-1, 2:] - u[1:-1, :-2]) / (2 * ds)
    v_diff_y = (v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * ds)
    u_diff_y = (u[2:, 1:-1] - u[:-2, 1:-1]) / (2 * ds)
    p_change = (
        rho
        * ds
        * ds
        / 4
        * (
            (1 / dt) * (u_diff_x + v_diff_y)
            - u_diff_x * u_diff_x
            - 2 * u_diff_y * v_diff_x
            - v_diff_y * v_diff_y
        )
    )
    # Iterate pressure
    for _ in range(50):
        # Rohr multiplications for Neumann boundary around object
        p[1:-1, 1:-1] = (
            p[1:-1, 2:] * rohr[1:-1, 2:]
            + p[1:-1, :-2] * rohr[1:-1, :-2]
            + p[2:, 1:-1] * rohr[2:, 1:-1]
            + p[:-2, 1:-1] * rohr[:-2, 1:-1]
            + rohr_count * p[1:-1, 1:-1]
        ) / 4
        p[1:-1, 1:-1] -= p_change

        # # Strong Neumann
        p[0] = p[1]
        p[-1] = p[-2]
        p[:, 0] = p[:, 1]

        # Strong Dirichlet
        p[:, -1] = 0
    p *= rohr


def _update(
    u: np.ndarray,
    p: np.ndarray,
    v: np.ndarray,
    rohr: np.ndarray,
    rohr_count: np.ndarray,
) -> None:
    """Update velocity fields using a finite difference scheme.

    Advances both u and v velocity components by one time step. Includes convective,
    pressure-gradient, and viscous terms. Applies strong Neumann boundary conditions
    at exits and zeros velocities inside the obstacle.

    Args:
        u: x-component of velocity (ny x nx array) - modified in-place.
        p: Pressure field (ny x nx array).
        v: y-component of velocity (ny x nx array) - modified in-place.
        rohr: Binary obstacle mask (1=fluid, 0=obstacle).
        rohr_count: Pre-computed count of fluid neighbors (used by _update_p).

    """
    new_u = u[1:-1, 1:-1].copy()

    ds_sqr = ds**2

    dtds = dt / ds
    dtdss = dt / (2 * ds)

    new_u -= u[1:-1, 1:-1] * (u[1:-1, 2:] - u[1:-1, :-2]) * dtdss
    new_u -= v[1:-1, 1:-1] * (u[2:, 1:-1] - u[:-2, 1:-1]) * dtdss
    new_u -= dtds / (2 * rho) * (p[1:-1, 2:] - p[1:-1, :-2])

    u_delta_x = u[1:-1, 2:] + u[1:-1, :-2] - 2 * u[1:-1, 1:-1]
    u_delta_y = u[2:, 1:-1] + u[:-2, 1:-1] - 2 * u[1:-1, 1:-1]
    new_u += nu * dt / ds_sqr * (u_delta_x + u_delta_y)

    new_v = v[1:-1, 1:-1].copy()

    new_v -= u[1:-1, 1:-1] * (v[1:-1, 2:] - v[1:-1, :-2]) * dtdss
    new_v -= v[1:-1, 1:-1] * (v[2:, 1:-1] - v[:-2, 1:-1]) * dtdss
    new_v -= dtds / (2 * rho) * (p[2:, 1:-1] - p[:-2, 1:-1])

    v_delta_x = v[1:-1, 2:] + v[1:-1, :-2] - 2 * v[1:-1, 1:-1]
    v_delta_y = v[2:, 1:-1] + v[:-2, 1:-1] - 2 * v[1:-1, 1:-1]
    new_v += nu * dt / ds_sqr * (v_delta_x + v_delta_y)

    u[1:-1, 1:-1] = new_u

    # Strong Neumann
    u[:, -1] = u[:, -2]

    v[1:-1, 1:-1] = new_v

    # Strong Neumann
    v[:, -1] = v[:, -2]

    # Rohr
    u *= rohr
    v *= rohr
    _update_p(u, v, p, rohr, rohr_count)


def animate_flow(num_frames: int = 100, interval: int = 1) -> None:  # pyright: ignore[reportUnusedFunction]
    """Create an animated visualization of the Navier-Stokes flow field evolution.

    Simulates and displays the time-dependent flow around a circular obstacle using
    matplotlib FuncAnimation. Each frame advances the simulation by 7 timesteps
    and displays flow speed as a heatmap.

    Args:
        num_frames: Number of animation frames to generate (default: 100).
        interval: Delay between frames in milliseconds (default: 1).

    """
    grids = init_grids()
    x = np.arange(nx) * ds
    y = np.arange(ny) * ds
    x, y = np.meshgrid(x, y)

    fig, ax = plt.subplots(figsize=(12, 4))
    time = 0

    def update_animation(frame: int):
        """Update function for animation."""
        nonlocal time
        for _ in range(10):
            time += dt
            update(grids)

        # Clear and redraw streamplot since StreamplotSet cannot be updated directly
        ax.clear()  # type: ignore[reportAttributeAccessIssue]
        ax.imshow(np.sqrt(grids.u**2 + grids.v**2), extent=(0, dim_x, 0, dim_y))

        ax.set_xlim(0, dim_x)  # type: ignore[reportAttributeAccessIssue]
        ax.set_ylim(0, dim_y)  # type: ignore[reportAttributeAccessIssue]
        ax.set_xlabel("x")  # type: ignore[reportAttributeAccessIssue]
        ax.set_ylabel("y")  # type: ignore[reportAttributeAccessIssue]
        ax.set_title(f"FD after {time:.3f} seconds")  # type: ignore[reportAttributeAccessIssue]

    anim = animation.FuncAnimation(
        fig,
        update_animation,  # pyright: ignore[reportArgumentType]
        frames=num_frames,
        interval=interval,
        repeat=True,
    )
    plt.show()


def main() -> None:
    """Entry point for the Navier-Stokes simulation.

    Prints the Reynolds number of the flow and launches the interactive animation.

    """
    print(f"Reynolds: {re}\nNu: {nu}\nds: {ds}")
    animate_flow()


if __name__ == "__main__":
    main()
