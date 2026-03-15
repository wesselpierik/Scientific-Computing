"""Finite difference implementation of Navier Stokes.

Group:        10
Course:       Scientific Computing

Description:
TODO:
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
start_velocity = 50

# Discretization parameters
ds = 0.01
dt = 0.0001
nx = int(dim_x / ds) + 1
ny = int(dim_y / ds) + 1

# Parameters
rho = 1
nu = 0.005


def init_grids() -> Grids:
    # Initialisation
    u = np.zeros((ny, nx))

    # Parabolic starting velocity
    def parabole(x: np.ndarray) -> np.ndarray:
        return (1 - ((x * 2 / dim_y) - 1) ** 2) * start_velocity

    u[:, 0] = parabole(np.arange(ny) * ds)
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
    _update(grids.u, grids.p, grids.v, grids.rohr, grids.rohr_count)


def _update_p(
    u: np.ndarray,
    v: np.ndarray,
    p: np.ndarray,
    rohr: np.ndarray,
    rohr_count: np.ndarray,
) -> None:
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
    new_u = u[1:-1, 1:-1].copy()

    ds_sqr = ds**2
    dt_sqr = dt**2
    u_sqr = u[1:-1, 1:-1] * u[1:-1, 1:-1]
    v_sqr = v[1:-1, 1:-1] * v[1:-1, 1:-1]

    dtds = dt / ds
    dtdss = dt / (2 * ds)

    new_u -= u[1:-1, 1:-1] * (u[1:-1, 2:] - u[1:-1, :-2]) * dtdss
    new_u -= v[1:-1, 1:-1] * (u[2:, 1:-1] - u[:-2, 1:-1]) * dtdss
    new_u -= dtds / (2 * rho) * (p[1:-1, 2:] - p[1:-1, :-2])

    u_delta_x = u[1:-1, 2:] + u[1:-1, :-2] - 2 * u[1:-1, 1:-1]
    u_delta_y = u[2:, 1:-1] + u[:-2, 1:-1] - 2 * u[1:-1, 1:-1]
    new_u += nu * dt / ds_sqr * (u_delta_x + u_delta_y)
    new_u += u_sqr * (dt_sqr / (2 * ds_sqr)) * (u_delta_x)
    new_u += v_sqr * (dt_sqr / (2 * ds_sqr)) * (u_delta_y)

    new_v = v[1:-1, 1:-1].copy()

    new_v -= u[1:-1, 1:-1] * (v[1:-1, 2:] - v[1:-1, :-2]) * dtdss
    new_v -= v[1:-1, 1:-1] * (v[2:, 1:-1] - v[:-2, 1:-1]) * dtdss
    new_v -= dtds / (2 * rho) * (p[2:, 1:-1] - p[:-2, 1:-1])

    v_delta_x = v[1:-1, 2:] + v[1:-1, :-2] - 2 * v[1:-1, 1:-1]
    v_delta_y = v[2:, 1:-1] + v[:-2, 1:-1] - 2 * v[1:-1, 1:-1]
    new_v += nu * dt / ds_sqr * (v_delta_x + v_delta_y)
    new_v += u_sqr * (dt_sqr / (2 * ds_sqr)) * (v_delta_x)
    new_v += v_sqr * (dt_sqr / (2 * ds_sqr)) * (v_delta_y)

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
    """Create an animated streamplot of the Navier-Stokes flow using FuncAnimation.

    Args:
        num_frames: Number of animation frames to generate
        interval: Delay between frames in milliseconds

    """
    grids = init_grids()
    x = np.arange(nx) * ds
    y = np.arange(ny) * ds
    x, y = np.meshgrid(x, y)

    fig, ax = plt.subplots(figsize=(12, 4))

    def update_animation(frame: int):
        """Update function for animation."""
        update(grids)
        update(grids)
        update(grids)
        update(grids)
        update(grids)
        update(grids)
        update(grids)

        # Clear and redraw streamplot since StreamplotSet cannot be updated directly
        ax.clear()  # type: ignore[reportAttributeAccessIssue]
        # ax.contourf(x, y, np.sqrt(grids.u**2 + grids.v**2), alpha=0.5, cmap=cm.viridis)  # pyright: ignore[reportAttributeAccessIssue]

        # ax.contourf(x, y, grids.v, alpha=0.5, cmap=cm.viridis)  # pyright: ignore[reportAttributeAccessIssue]
        ax.imshow(np.sqrt(grids.u**2 + grids.v**2), extent=(0, dim_x, 0, dim_y))

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
    print(f"Reynolds: {start_velocity * rohr_rad * 2 / nu}")
    animate_flow()
    # grids = init_grids()
    # while True:
    #     for _ in range(10):
    #         update(grids)
    #     x = np.linspace(0, 2.2, nx)
    #     y = np.linspace(0, 0.41, ny)
    #     X, Y = np.meshgrid(x, y)
    #     fig = plt.figure(figsize=(11, 7), dpi=100)
    #     # plotting the pressure field as a contour
    #     plt.contourf(X, Y, grids.p, alpha=0.5, cmap=cm.viridis)  # pyright: ignore[reportAttributeAccessIssue]
    #     plt.colorbar()
    #     # plotting the pressure field outlines
    #     plt.contour(X, Y, grids.p, cmap=cm.viridis)  # pyright: ignore[reportAttributeAccessIssue]
    #     # plotting velocity field
    #     plt.quiver(X[::2, ::2], Y[::2, ::2], grids.u[::2, ::2], grids.v[::2, ::2])
    #     plt.xlabel("X")
    #     plt.ylabel("Y")
    #     plt.show()


if __name__ == "__main__":
    main()
