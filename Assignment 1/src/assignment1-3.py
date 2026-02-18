import argparse
from functools import partial

from matplotlib.lines import lineStyles
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.image import AxesImage
from matplotlib.patches import Rectangle
from time_independent import SOR, BaseGrid, GaussSeidel, Jacobi, SOR_Insulating


def concentration_curve(
    grid: BaseGrid, *, axis: Axes | None = None, column: int = 0
) -> None:
    if axis is None:
        axis = plt.subplot()

    checkpoints = 10 ** np.arange(0, 5)
    steps = 0
    axis.set_xlabel("y")
    axis.set_ylabel("c")
    axis.grid()
    for checkpoint in checkpoints:
        while steps < checkpoint:
            grid.step()
            steps += 1
        axis.plot(
            np.linspace(1, 0, grid.grid_size),
            grid.state[:, column],
            label=f"{checkpoint}",
        )
    axis.legend()


def assignment_h() -> None:
    grid_size = 50
    fig = plt.figure(figsize=(10, 3))
    axes = fig.subplots(ncols=3)

    jacobi = Jacobi(grid_size)
    axes[0].set_title("Jacobi")
    concentration_curve(jacobi, axis=axes[0])

    axes[1].set_title("Gauss-Seidel")
    gauss = GaussSeidel(grid_size)
    concentration_curve(gauss, axis=axes[1])

    axes[2].set_title("Successive Over Relaxation")
    sor = SOR(grid_size, 1.7)
    concentration_curve(sor, axis=axes[2])

    plt.show()


def plot_deltas(grid: BaseGrid, axis: Axes, label: str):
    x = np.arange(30000)
    deltas = [grid.step() for _ in x]
    axis.plot(x, deltas, label=label)


def assignment_i() -> None:
    fig = plt.figure()
    axis = fig.subplots(nrows=1)
    axis.set_yscale("log")

    grid_size = 50

    plot_deltas(Jacobi(grid_size), axis, "Jacobi")
    plot_deltas(GaussSeidel(grid_size), axis, "Gauss-Seidel")

    plot_deltas(SOR(grid_size, 0.75), axis, r"SOR with $\omega = 0.75$")
    plot_deltas(SOR(grid_size, 1.5), axis, r"SOR with $\omega = 1.5$")
    plot_deltas(SOR(grid_size, 2), axis, r"SOR with $\omega = 2$")

    axis.grid()
    axis.set_xlabel("k (steps)")
    axis.set_ylabel(r"$\delta$")
    axis.legend()
    plt.show()


def run_till_condition(grid: BaseGrid, epsilon: float, max_steps: int):
    steps = 0
    while grid.step() > epsilon and steps < max_steps:
        steps += 1
    return steps


def optimize_w(
    grid_size: int, *, sinks: list[tuple[int, int, int, int]] | None = None
) -> float:
    """Perform an objectively worse implementation of golden section search."""
    lower = 1
    upper = 2
    max_error = 1e-3
    epsilon = 1e-8
    max_steps = 20000

    if sinks is None:
        sinks = []

    while upper - lower > max_error:
        left = 0.49 * (upper - lower) + lower
        right = upper - left + lower

        grid = SOR(grid_size, left)
        for sink in sinks:
            grid.add_sink(*sink)
        steps_left = run_till_condition(grid, epsilon, max_steps)

        grid = SOR(grid_size, right)
        for sink in sinks:
            grid.add_sink(*sink)
        steps_right = run_till_condition(grid, epsilon, max_steps)

        if steps_left > steps_right:
            lower = left
        else:
            upper = right
    return (upper + lower) / 2


def assignment_j() -> None:
    """Plot the optimal omega for differing grid_sizes between 10 and 150."""
    grid_sizes = np.arange(10, 150, 4, dtype=int)
    optima = [optimize_w(grid_size) for grid_size in grid_sizes]
    plt.plot(grid_sizes, optima)
    plt.grid()
    plt.xlabel("Grid size")
    plt.ylabel(r"$\omega$")
    plt.xlim(5, 160)
    plt.ylim(1.6, 2)
    plt.title(r"Optimal $\omega$ over grid size")
    plt.show()


def show_sinks(grid: BaseGrid, axis: Axes, title: str) -> int:
    epsilon = 1e-10
    max_steps = 100000
    steps = run_till_condition(grid, epsilon, max_steps)
    axis.imshow(grid.state, cmap="viridis")
    axis.set_title(title)
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    return steps


def assignment_k() -> None:
    grid_size = 50
    fig = plt.figure(figsize=(20, 3))
    axes = fig.subplots(ncols=5)

    # Large sink heatmap
    grid = SOR(grid_size, 1.9)
    sink_large = (10, 5, 15, 35)
    grid.add_sink(*sink_large)
    steps_large = show_sinks(grid, axes[0], "Concentration large sink")

    # Random sinks heatmap
    grid = SOR(grid_size, 1.9)
    seed = 43
    gen = np.random.Generator(np.random.PCG64(seed))
    sinks_random = []
    for _ in range(10):
        column, row = gen.integers(1, grid_size - 1, 2)
        sink = (row, row, column, column)
        sinks_random.append(sink)
        grid.add_sink(*sink)
    steps_random = show_sinks(grid, axes[1], "Concentration random sinks")
    fig.colorbar(axes[0].get_images()[0], ax=axes)

    # Different locations for sinks heatmap
    ## Top
    grid = SOR(grid_size, 1.9)
    sink_top = (10, 5, 25, 25)
    grid.add_sink(*sink_top)
    steps_top = show_sinks(grid, axes[2], "Concentration top sink")

    ## Middle
    grid = SOR(grid_size, 1.9)
    sink_middle = (25, 20, 25, 25)
    grid.add_sink(*sink_middle)
    steps_middle = show_sinks(grid, axes[3], "Concentration middle sink")

    ## Bottom
    grid = SOR(grid_size, 1.9)
    sink_bottom = (45, 40, 25, 25)
    grid.add_sink(*sink_bottom)
    steps_bottom = show_sinks(grid, axes[4], "Concentration bottom sink")

    plt.show()

    # Sink analysis
    fig = plt.figure()
    axes = fig.subplots(nrows=1, ncols=2)

    # Bar plots showing required number of steps
    grid = SOR(grid_size, 1.9)
    steps_normal = run_till_condition(grid, 1e-10, 50000)
    sink_types = ["Normal", "Large", "Random", "Top", "Middle", "Bottom"]
    steps_data = [
        steps_normal,
        steps_large,
        steps_random,
        steps_top,
        steps_middle,
        steps_bottom,
    ]
    axes[0].bar(sink_types, steps_data)
    axes[0].set_ylabel("Steps to convergence")
    axes[0].set_title("Convergence speed by sink type")
    axes[0].grid(axis="y")

    # Bar plots showing optimum omega
    sink_types = ["Normal", "Large", "Random", "Top", "Middle", "Bottom"]
    normal_w = optimize_w(grid_size)
    steps_data = [
        normal_w,
        optimize_w(grid_size, sinks=[sink_large]),
        optimize_w(grid_size, sinks=sinks_random),
        optimize_w(grid_size, sinks=[sink_top]),
        optimize_w(grid_size, sinks=[sink_middle]),
        optimize_w(grid_size, sinks=[sink_bottom]),
    ]
    axes[1].set_ylim(1.5, 2)
    axes[1].hlines(normal_w, -10, 10, color="red", linestyles="dashed")
    axes[1].bar(sink_types, steps_data)
    axes[1].set_ylabel(r"optimal $\omega$")
    axes[1].set_title("Convergence speed by sink type")
    yticks = [*axes[1].get_yticks(), normal_w]
    yticklabels = [*axes[1].get_yticklabels(), f"{normal_w:.2f}"]
    axes[1].set_yticks(yticks, labels=yticklabels)
    axes[1].grid(axis="y")
    plt.show()


def optimize_w_insulating(
    grid_size: int, *, insulators: list[tuple[int, int, int, int]] | None = None
) -> float:
    """Perform an objectively worse implementation of golden section search.

    Unline the regular optimize_w, this one uses the SOR_Insulating class with the given
    grid_size and insulator corner points.
    """
    lower = 1
    upper = 2
    max_error = 1e-3
    epsilon = 1e-8
    max_steps = 20000

    if insulators is None:
        insulators = []

    while upper - lower > max_error:
        left = 0.49 * (upper - lower) + lower
        right = upper - left + lower

        grid = SOR_Insulating(grid_size, left)
        for sink in insulators:
            grid.add_sink(*sink)
        steps_left = run_till_condition(grid, epsilon, max_steps)

        grid = SOR_Insulating(grid_size, right)
        for sink in insulators:
            grid.add_sink(*sink)
        steps_right = run_till_condition(grid, epsilon, max_steps)

        if steps_left > steps_right:
            lower = left
        else:
            upper = right
    return (upper + lower) / 2


def assignment_l() -> None:
    grid_size = 50
    fig = plt.figure()
    axes = fig.subplots(ncols=2)

    # Two split heatmap
    grid = SOR_Insulating(grid_size, 1.9)
    grid.add_sink(5, 5, 0, 19)
    grid.add_sink(5, 5, 22, 28)
    grid.add_sink(5, 5, 31, 49)
    show_sinks(grid, axes[0], "Concentration two-slit")

    # Highlighting which row is used for concentration
    axes[0].add_patch(
        Rectangle(
            (19.5, -0.5), 1, 50, fill=False, edgecolor="crimson", lw=1, clip_on=False
        )
    )

    # Random insulators heatmap
    grid = SOR_Insulating(grid_size, 1.9)
    seed = 43
    gen = np.random.Generator(np.random.PCG64(seed))
    sinks_random = []
    for _ in range(100):
        column, row = gen.integers(1, grid_size - 1, 2)
        sink = (row, row, column, column)
        sinks_random.append(sink)
        grid.add_sink(*sink)
    show_sinks(grid, axes[1], "Concentration random obstructions")

    # Highlighting which row is used for concentration
    axes[1].add_patch(
        Rectangle(
            (19.5, -0.5), 1, 50, fill=False, edgecolor="crimson", lw=1, clip_on=False
        )
    )
    fig.colorbar(axes[0].get_images()[0], ax=axes)
    plt.show()

    fig = plt.figure()
    axes = fig.subplots(ncols=2)

    # Concentration curves
    grid = SOR_Insulating(grid_size, 1.9)
    grid.add_sink(5, 5, 0, 19)
    grid.add_sink(5, 5, 22, 28)
    grid.add_sink(5, 5, 31, 49)
    axes[0].plot(np.linspace(0, 1, 2), np.linspace(0, 1, 2), "--", label="Target")
    axes[0].set_title("Concentration curve two-slit")
    concentration_curve(grid, axis=axes[0], column=20)

    grid = SOR_Insulating(grid_size, 1.9)
    for sink in sinks_random:
        grid.add_sink(*sink)
    axes[1].plot(np.linspace(0, 1, 2), np.linspace(0, 1, 2), "--", label="Target")
    axes[1].set_title("Concentration curve random obstructions")
    concentration_curve(grid, axis=axes[1], column=20)
    plt.show()


# Animated grid functions:
def show_diffusion_step(frame: int, grid: BaseGrid, im: AxesImage) -> list[Artist]:
    im.set_data(grid.state)
    grid.step()
    return [im]


def show_diffusion(grid: BaseGrid) -> None:
    fig = plt.figure()
    axis = fig.subplots(nrows=1)
    im = axis.imshow(grid.state)
    _anim = animation.FuncAnimation(
        fig,
        partial(show_diffusion_step, grid=grid, im=im),
        100,
        interval=1000,
        blit=True,
    )
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "assignment",
        help="The identifier of the relevant sub-assignment.",
        type=str,
        choices=["H", "I", "J", "K", "L"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assignment = args.assignment
    match assignment:
        case "H":
            assignment_h()
        case "I":
            assignment_i()
        case "J":
            assignment_j()
        case "K":
            assignment_k()
        case "L":
            assignment_l()


if __name__ == "__main__":
    main()
