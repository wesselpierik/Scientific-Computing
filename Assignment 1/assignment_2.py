from functools import partial
import math
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.image import AxesImage
from matplotlib import animation
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import argparse

def stability_condition(delta_t, delta_x, D):
    """Function to check whether the stability condition is satisfied for the specified time and space step sizes.

    Args:
        delta_t (float): time step size
        delta_x (float): space step size (x and y directions have the same step size)
        D (float): diffusion coefficient

    Returns:
        bool: True if the stability condition is satisfied, False otherwise
    """    
    stability_check = (4 * delta_t * D) / (delta_x**2)
    if stability_check <= 1:
        print(f"The stability condition is satisfied, value is: {stability_check}")
        return True

    print(f"The stability condition is not satisfied, value is: {stability_check}")
    return False


def concentration_timestep(c, delta_x, delta_t, D, tolerance):
    """Function that updates the spatial grid for a single step size.

    Args:
        c (npt.NDArray): array containing the concentration values (on x and y grid) at the current time step
        delta_x (float): space step size (x and y directions have the same step size)
        delta_t (float): time step size
        D (float): diffusion coefficient
        tolerance (float): tolerance for the stopping criterion

    Returns:
        npt.NDArray: updated array containing the concentration values after one time step
    """    
    N = c.shape[0]
    c_new = np.copy(c)
    tolerance_counter = 0
    tolerance_reached = False

    for x in range(N):
        for y in range(1, N - 1):
            c_k = c[y, x]
            # Boundary condition
            if x == N - 1:
                c_new[y, x] = c[y, x] + delta_t * D / (delta_x**2) * (
                    c[y, 1] + c[y, x - 1] + c[y + 1, x] + c[y - 1, x] - 4 * c[y, x]
                )

            elif x == 0:
                c_new[y, x] = c[y, x] + delta_t * D / (delta_x**2) * (
                    c[y, x + 1] + c[y, -2] + c[y + 1, x] + c[y - 1, x] - 4 * c[y, x]
                )

            else:
                c_new[y, x] = c[y, x] + delta_t * D / (delta_x**2) * (
                    c[y, x + 1] + c[y, x - 1] + c[y + 1, x] + c[y - 1, x] - 4 * c[y, x]
                )

            delta_c = abs(c_new[y, x] - c_k)
            if delta_c < tolerance:
                tolerance_counter += 1

    print(f"Tolerance: {tolerance_counter} out of {(N - 2) * N} ({round(100 * tolerance_counter / ((N - 2) * N), 1)}%)")

    if tolerance_counter == (N - 2) * N:
        print("Tolerance reached for all grid points.")
        tolerance_reached = True

    return c_new, tolerance_reached


def plot_analytic(t: float, D: float, N: int, *, axes: Axes | None = None):
    """Function to plot the analytical solution of the diffusion equation for a given time t, diffusion coefficient D and number of grid points N.

    Args:
        t (float): time
        D (float): diffusion coefficient
        N (int): number of grid points
        axes (Axes | None, optional): matplotlib axes object. Defaults to None.
    """    
    if axes is None:
        fig = plt.figure()
        axes = fig.subplots(nrows=1, ncols=1)

    y_range = np.linspace(0, 1, N)
    i_max = 100
    c = [
        sum(
            [
                math.erfc((1 - y + 2 * i) / 2 * math.sqrt(D * t))
                - math.erfc((1 + y + 2 * i) / 2 * math.sqrt(D * t))
                for i in range(i_max)
            ]
        )
        for y in y_range
    ]
    axes.plot(y_range, c)


# Animated grid functions:
def show_diffusion_step(
    frame: int,
    grid: list[npt.NDArray],
    im: AxesImage,
    delta_x: float,
    delta_t: float,
    D: float,
    tolerance: float,
) -> list[Artist]:
    """Function to update the grid for a single time step and update the image for the animation.

    Args:
        frame (int): current frame number
        grid (list[npt.NDArray]): list containing the concentration grid
        im (AxesImage): matplotlib image object
        delta_x (float): space step size
        delta_t (float): time step size
        D (float): diffusion coefficient
        tolerance (float): tolerance for the stopping criterion
    Returns:
        list[Artist]: list containing the updated image artist
    """   
    grid[0], _ = concentration_timestep(
        grid[0],
        delta_x,
        delta_t,
        D,
        tolerance
    )

    im.set_data(grid[0])

    return [im]


def show_diffusion(c: npt.NDArray, delta_x: float, delta_t: float, D: float, tolerance: float) -> None:
    """Function to show the diffusion process as an animation.

    Args:
        c (npt.NDArray): initial concentration grid
        delta_x (float): space step size
        delta_t (float): time step size
        D (float): diffusion coefficient
        tolerance (float): tolerance for the stopping criterion
    """    
    fig = plt.figure()
    axis = fig.subplots(nrows=1)
    im = axis.imshow(c)
    _anim = animation.FuncAnimation(
        fig,
        partial(
            show_diffusion_step, grid=[c], im=im, delta_x=delta_x, delta_t=delta_t, D=D, tolerance=tolerance
        ),
        100,
        interval=0.01,
        blit=True,
    )
    plt.show()


def parse_args() -> argparse.Namespace:
    """Function to parse the command line arguments.

    Returns:
        argparse.Namespace: namespace containing the parsed arguments
    """    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "option", help="Determine the code to run", choices=["animated", "tolerance"]
    )
    return parser.parse_args()


def main():
    parser = parse_args()
    option = parser.option

    # boundary conditions
    c_y0 = 0
    c_y1 = 1

    # interval lengths
    N = 50
    t0 = 0
    tN = 5000
    delta_x = 1 / N
    delta_t = 0.0001

    # parameters
    D = 1

    # tolerance for the stopping criterion
    p = 6
    tolerance = 10 ** (-p)
    tolerance_reached = False

    # array with x and y
    c = np.zeros((N, N))
    c[0,] = c_y1

    stability = stability_condition(delta_t, delta_x, D)

    if option == "animated":
        show_diffusion(c, delta_x, delta_t, D, tolerance)
    elif option == "tolerance":
        iteration_step = 0
        while tolerance_reached == False:
            c, tolerance_reached = concentration_timestep(c, delta_x, delta_t, D, tolerance)
            iteration_step += 1
        print(f"Tolerance reached after {iteration_step} iteration steps.")
            

if __name__ == "__main__":
    main()
