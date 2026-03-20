"""Lattice Boltzman implementation of the incompressible Navier Stokes equations."""

from functools import reduce
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from numba import njit, prange

# Karman street
dim_x = 2.2
rohr_x = 0.2
rohr_y = 0.2
rohr_rad = 0.05
dim_y = 0.41

# Discretization parameters
ds = 0.0025
# dt = 0.0001
nx = int(dim_x / ds) + 1
ny = int(dim_y / ds) + 1

re = 300
u = 0.1
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
    """Vectorized rohr obstacle setup (88x faster)."""
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
    )
)
rohr_top = int(
    reduce(
        lambda acc, x: x[0] if x[1] != ny else acc,  # pyright: ignore[reportIndexIssue]
        enumerate(np.sum(rohr, axis=1)),
        0,
    )
)
rohr_left = int(
    reduce(
        lambda acc, x: x[0] if x[1] != nx and acc == 0 else acc,  # pyright: ignore[reportIndexIssue]
        enumerate(np.sum(rohr, axis=0)),
        0,
    )
)
rohr_right = int(
    reduce(
        lambda acc, x: x[0] if x[1] != nx else acc,  # pyright: ignore[reportIndexIssue]
        enumerate(np.sum(rohr, axis=0)),
        0,
    )
)


@njit(cache=True, fastmath=True)
def feq(f: np.ndarray) -> np.ndarray:
    """Optimized feq with loop fusion and fastmath."""
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
    # Reflect of center column
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
    """Direct streaming without intermediate rolls."""
    for d in range(9):
        ex, ey = e[d, 0], e[d, 1]
        for i in range(nx):
            for j in range(ny):
                src_i = (i - ex) % nx
                src_j = (j - ey) % ny
                f[i, j, d] = f_new[src_i, src_j, d]


@njit(cache=True)
def inflow(f: np.ndarray) -> None:
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
    # Zero-gradient outflow at the right boundary
    f[-1, :, :] = f[-2, :, :]


@njit(cache=True, fastmath=True)
def bounce_back_walls(f: np.ndarray, f_new: np.ndarray) -> None:
    """Bounce-back with explicit loops."""
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
    # Collision
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
    # Start with equilibrium to ease in simulation
    directions = 9
    f = feq(np.ones((nx, ny, directions)))

    x = np.arange(nx) * ds
    y = np.arange(ny) * ds
    x, y = np.meshgrid(x, y)

    fig, ax = plt.subplots(figsize=(12, 4))
    # step(f)

    def update_animation(frame: int):
        """Update function for animation."""
        for _ in range(20):
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
            # interpolation="none",
        )

        ax.set_xlim(0, dim_x)  # type: ignore[reportAttributeAccessIssue]
        ax.set_ylim(0, dim_y)  # type: ignore[reportAttributeAccessIssue]
        ax.set_xlabel("x")  # type: ignore[reportAttributeAccessIssue]
        ax.set_ylabel("y")  # type: ignore[reportAttributeAccessIssue]
        ax.set_title(f"Navier-Stokes Flow (Frame {frame + 1}/{num_frames})")  # type: ignore[reportAttributeAccessIssue]

    anim = animation.FuncAnimation(
        fig,
        update_animation,  # pyright: ignore[reportArgumentType]
        frames=num_frames,
        interval=interval,
        repeat=True,
    )
    plt.show()


def main() -> None:
    animate_flow()


if __name__ == "__main__":
    main()
