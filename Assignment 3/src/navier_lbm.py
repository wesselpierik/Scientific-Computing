"""Lattice Boltzman implementation of the incompressible Navier Stokes equations."""

from functools import reduce
from itertools import product

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
# dt = 0.0001
nx = int(dim_x / ds) + 1
ny = int(dim_y / ds) + 1

re = 200
u = 0.12
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


cs_sqr = 1 / 3

rohr = np.zeros((ny, nx))
for row, column in product(range(ny), range(nx)):
    rohr[row, column] = int(
        (row * ds - rohr_y) ** 2 + (column * ds - rohr_x) ** 2 > rohr_rad**2,
    )
no_slip = rohr.copy()
no_slip[0] = 0
no_slip[-1] = 0

rohr_bottom = int(
    reduce(
        lambda acc, x: x[0] if x[1] != nx and acc == 0 else acc,  # pyright: ignore[reportIndexIssue]
        enumerate(np.sum(rohr, axis=1)),
        0,
    )
)
rohr_top = int(
    reduce(
        lambda acc, x: x[0] if x[1] != nx else acc,  # pyright: ignore[reportIndexIssue]
        enumerate(np.sum(rohr, axis=1)),
        0,
    )
)
rohr_left = int(
    reduce(
        lambda acc, x: x[0] if x[1] != ny and acc == 0 else acc,  # pyright: ignore[reportIndexIssue]
        enumerate(np.sum(rohr, axis=0)),
        0,
    )
)
rohr_right = int(
    reduce(
        lambda acc, x: x[0] if x[1] != ny else acc,  # pyright: ignore[reportIndexIssue]
        enumerate(np.sum(rohr, axis=0)),
        0,
    )
)


def feq(f: np.ndarray) -> np.ndarray:
    eq = np.zeros((ny, nx, 9))

    rho = np.sum(f, axis=2)
    ux = np.sum(f * e[:, 1], axis=2) / rho
    uy = np.sum(f * e[:, 0], axis=2) / rho

    u_sqr = ux**2 + uy**2
    for direction in range(9):
        uxe = ux * e[direction, 1]
        uye = uy * e[direction, 0]
        ue = uxe + uye

        first = 1 / 3
        second = ue
        third = ue * ue * 3 / 2
        last = u_sqr / 2

        eq[:, :, direction] = 3 * w[direction] * rho * (first + second + third - last)
    return eq


@njit(cache=True)
def reflections(f: np.ndarray, f_new: np.ndarray) -> None:
    # Reflect of center column
    for row in range(rohr_bottom - 1, rohr_top + 2):
        for column in range(rohr_left - 1, rohr_right + 2):
            for direction in range(9):
                new_row = row + e[direction, 0]
                new_column = column + e[direction, 1]
                if rohr[new_row, new_column] == 0:
                    f_new[new_row, new_column, opposite[direction]] = f[
                        row,
                        column,
                        direction,
                    ]

    # Reflect of walls
    for row in [1, ny - 2]:
        for column in range(nx):
            for direction in range(9):
                # for direction in [3]:
                new_row = row + e[direction, 0]
                new_column = column + e[direction, 1]
                if (new_row == 0 or new_row == ny - 1) and (
                    new_column >= 0 and new_column < nx
                ):
                    f_new[new_row, new_column, opposite[direction]] = f[
                        row, column, direction
                    ]


def stream(f: np.ndarray, f_new: np.ndarray) -> None:
    # Principal axes
    f[1:-1, :-1, 1] = f_new[:-2, :-1, 1]
    f[1:-1, 1:-1, 2] = f_new[1:-1, :-2, 2]
    f[1:-1, :-1, 3] = f_new[2:, :-1, 3]
    f[1:-1, :-1, 4] = f_new[1:-1, 1:, 4]

    # Diagonals
    f[1:-1, 1:-1, 5] = f_new[:-2, :-2, 5]
    f[1:-1, 1:-1, 6] = f_new[2:, :-2, 6]
    f[1:-1, :-1, 7] = f_new[2:, 1:, 7]
    f[1:-1, :-1, 8] = f_new[:-2, 1:, 8]

    f[:, -1] = f[:, -2]


# @njit(cache=True)
def inflow(f: np.ndarray) -> None:
    top = ny - 2
    bottom = 1
    u_parabolic = (1 - (np.linspace(-1, 1, (top - bottom + 1)) ** 2)) * u
    rho_in = (
        (
            f[bottom : top + 1, 0, 0]
            + f[bottom : top + 1, 0, 1]
            + f[bottom : top + 1, 0, 3]
        )
        + 2.0
        * (
            f[bottom : top + 1, 0, 4]
            + f[bottom : top + 1, 0, 7]
            + f[bottom : top + 1, 0, 8]
        )
    ) / (1.0 - u_parabolic)

    f[bottom : top + 1, 0, 2] = (
        f[bottom : top + 1, 0, 4] + (2.0 / 3.0) * rho_in * u_parabolic
    )
    f[bottom : top + 1, 0, 5] = (
        f[bottom : top + 1, 0, 7]
        - 0.5 * (f[bottom : top + 1, 0, 1] - f[bottom : top + 1, 0, 3])
        + (1.0 / 6.0) * rho_in * u_parabolic
    )
    f[bottom : top + 1, 0, 6] = (
        f[bottom : top + 1, 0, 8]
        + 0.5 * (f[bottom : top + 1, 0, 1] - f[bottom : top + 1, 0, 3])
        + (1.0 / 6.0) * rho_in * u_parabolic
    )


def step(f: np.ndarray) -> None:
    # Collision
    f_new = f - (f - feq(f)) / tau

    # Reflections against the object and top and bottom walls
    reflections(f, f_new)

    # Streaming
    stream(f, f_new)

    # Input boundary conditions
    inflow(f)


def animate_flow(num_frames: int = 100, interval: int = 1) -> None:  # pyright: ignore[reportUnusedFunction]
    # Start with equilibrium to ease in simulation
    directions = 9
    f = feq(np.ones((ny, nx, directions)))
    x = np.arange(nx) * ds
    y = np.arange(ny) * ds
    x, y = np.meshgrid(x, y)

    fig, ax = plt.subplots(figsize=(12, 4))

    def update_animation(frame: int):
        """Update function for animation."""
        step(f)
        step(f)
        step(f)
        step(f)
        step(f)
        step(f)
        step(f)
        step(f)
        step(f)
        step(f)
        step(f)
        step(f)
        step(f)

        # Clear and redraw streamplot since StreamplotSet cannot be updated directly
        ax.clear()  # type: ignore[reportAttributeAccessIssue]
        rho = np.sum(f, axis=2)
        ux = np.sum(f * e[:, 1], axis=2) / (rho + (rho == 0))
        uy = np.sum(f * e[:, 0], axis=2) / (rho + (rho == 0))
        speed = np.sqrt(ux**2 + uy**2)
        ax.imshow(
            # f[:, :, 5],
            # np.sum(f, axis=2),
            speed * rohr,
            origin="lower",
            extent=(0, dim_x, 0, dim_y),
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
