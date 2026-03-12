"""Finite difference implementation of Navier Stokes.

Group:        10
Course:       Scientific Computing

Description:
TODO:
"""

from dataclasses import dataclass
from itertools import product

import matplotlib.pyplot as plt
import numba
import numpy as np
from numba import float64
from numba.experimental import jitclass

grid_spec = [
    ("u", float64[:, :]),
    ("v", float64[:, :]),
    ("p", float64[:, :]),
    ("rohr", float64[:, :]),
]


@jitclass(grid_spec)  # pyright: ignore[reportCallIssue]
class Grids:
    def __init__(
        self,
        u: np.ndarray[tuple[int, int], np.dtype[np.float64]],
        v: np.ndarray[tuple[int, int], np.dtype[np.float64]],
        p: np.ndarray[tuple[int, int], np.dtype[np.float64]],
        rohr: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    ) -> None:
        self.u = u
        self.p = p
        self.v = v
        self.rohr = rohr


# Karman street
dim_x = 2.2
rohr_x = 0.2
rohr_y = 0.2
rohr_rad = 0.05
dim_y = 0.41
start_velocity = 5

# Discretization parameters
ds = 0.01
dt = 0.1
nx = int(dim_x / ds) + 1
ny = int(dim_y / ds) + 1

# Parameters
rho = 1
nu = 1


def init_grids() -> Grids:
    # Initialisation
    u = np.zeros((ny, nx))
    v = np.zeros_like(u)
    p = np.zeros_like(u)
    rohr = np.zeros_like(u)
    for row, column in product(range(ny), range(nx)):
        rohr[row, column] = int(
            (row * ds - rohr_y) ** 2 + (column * ds - rohr_x) ** 2 > rohr_rad**2,
        )
    return Grids(u, v, p, rohr)


def update(grids: Grids) -> None:
    _update(grids)


# @numba.njit(cache=True)
def _update(grids: Grids) -> None:
    u = grids.u
    v = grids.v
    p = grids.p
    rohr = grids.rohr
    new_u = u[1:-1, 1:-1].copy()

    ds_sqr = ds**2

    dtds = dt / ds
    new_u -= u[1:-1, 1:-1] * (u[1:-1, 1:-1] - u[1:-1, :-2]) * dtds
    new_u -= v[1:-1, 1:-1] * (u[1:-1, 1:-1] - u[:-2, 1:-1]) * dtds
    new_u -= dtds / (2 * rho) * (p[1:-1, 2:] - p[1:-1, :-2])
    new_u += (
        nu
        * dt
        / ds_sqr
        * (-4 * u[1:-1, 1:-1] + u[1:-1, 2:] + u[1:-1, :-2] + u[2:, 1:-1] + u[:-2, 1:-1])
    )

    new_v = v[1:-1, 1:-1].copy()

    new_v -= u[1:-1, 1:-1] * (v[1:-1, 1:-1] - v[1:-1, :-2]) * dtds
    new_v -= v[1:-1, 1:-1] * (v[1:-1, 1:-1] - v[:-2, 1:-1]) * dtds
    new_v -= dtds / (2 * rho) * (p[2:, 1:-1] - p[:-2, 1:-1])
    new_v += (
        nu
        * dt
        / ds_sqr
        * (-4 * v[1:-1, 1:-1] + v[1:-1, 2:] + v[1:-1, :-2] + v[2:, 1:-1] + v[:-2, 1:-1])
    )

    u[1:-1, 1:-1] = new_u
    # Strong Dirichlet
    u[0] = 0
    u[-1] = 0
    u[:, 0] = start_velocity

    # Strong Neumann
    u[:, -1] = u[:, -2]

    v[1:-1, 1:-1] = new_v

    # Strong Dirichlet
    v[0] = 0
    v[-1] = 0
    v[:, 0] = 0

    # Strong Neumann
    v[:, -1] = v[:, -2]

    # Rohr
    u *= rohr
    v *= rohr

    v_diff_x = (v[1:-1, 2:] - v[1:-1, :-2]) / (2 * ds)
    u_diff_x = (u[1:-1, 2:] - u[1:-1, :-2]) / (2 * ds)
    v_diff_y = (v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * ds)
    u_diff_y = (u[2:, 1:-1] - u[:-2, 1:-1]) / (2 * ds)
    p_change = (
        rho
        * ds_sqr
        * ds_sqr
        / (4 * ds_sqr)
        * (
            (1 / dt) * (u_diff_x + v_diff_y)
            - u_diff_x * u_diff_x
            - 2 * u_diff_y * v_diff_x
            - v_diff_y * v_diff_y
        )
    )
    # Iterate pressure
    for _ in range(0):
        p[1:-1, 1:-1] = (
            ds_sqr * (p[1:-1, 2:] + p[1:-1, :-2])
            + (p[2:, 1:-1] + p[:-2, 1:-1]) / 4 * ds_sqr
        )
        p[1:-1, 1:-1] -= p_change

        # Strong Neumann
        p[0] = p[1]
        p[-1] = p[-2]
        p[:, 0] = p[:, 1]

        # Strong Dirichlet
        p[:, -1] = 0


def main() -> None:
    grids = init_grids()
    update(grids)
    x = np.arange(nx) * ds
    y = np.arange(ny) * ds
    x, y = np.meshgrid(x, y)
    plt.streamplot(x, y, grids.u, grids.v)
    plt.show()


if __name__ == "__main__":
    main()
