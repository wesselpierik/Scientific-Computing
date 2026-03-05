"""Assignment 2: DLA Analysis and Visualization.

Group:         10
Course:        Scientific Computing

Description:
-----------
This module provides tools for analyzing and visualizing Diffusion-Limited
Aggregation (DLA) simulations across different eta and omega parameters.
It includes functions for parameter sweeps, performance benchmarking, and
visualization of growth patterns. Supports multiple parallelization backends
(Numba, manual multiprocessing, or single-threaded).
"""

import argparse
import csv
import time

import dla
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from tqdm import tqdm
from utils import data_dir

parallelization = None


def create_dla(
    grid_size: int,
    eta: float,
    omega: float,
    epsilon: float,
    seed: int = 43,
) -> dla.DLA:
    """Create a DLA instance with the specified parameters.

    Args:
        grid_size: Size of the simulation grid.
        eta: Nutrient diffusion bias parameter.
        omega: Over-relaxation parameter for the SOR solver.
        epsilon: Convergence tolerance for the SOR solver.
        seed: Random seed for reproducibility (default 43).

    Returns:
        A configured DLA instance based on the global parallelization setting.

    """
    if parallelization == "numba":
        return dla.DLA(
            grid_size,
            eta,
            omega=omega,
            epsilon=epsilon,
            seed=seed,
            numba_parallel=True,
        )
    if parallelization == "manual":
        return dla.DLA(
            grid_size,
            eta,
            omega=omega,
            epsilon=epsilon,
            seed=seed,
            workers=16,
        )
    return dla.DLA(grid_size, eta, omega=omega, epsilon=epsilon, seed=seed)


def plot_omega_steps(
    start_grid: dla.DLA,
    iterations: int,
    max_steps: int,
    all_omega: np.ndarray,
    axes: Axes,
    label: str,
) -> None:
    """Plot the number of steps required for convergence across omega values.

    Args:
        start_grid: Initial DLA grid configuration to use as baseline.
        iterations: Number of grow_candidate iterations per omega test.
        max_steps: Maximum number of nutrient solver steps per iteration.
        all_omega: Array of omega values to test.
        axes: Matplotlib axes object to plot results on.
        label: Legend label for the plotted data.

    """
    all_steps = []
    for omega in tqdm(all_omega):
        # A lot of half legal variable accesses here.
        grid = create_dla(start_grid._grid_size, 1, omega, start_grid._epsilon)  # noqa: SLF001
        grid._growths = start_grid._growths.copy()  # noqa: SLF001
        grid._nutrients = start_grid._nutrients.copy()  # noqa: SLF001
        grid._candidate_array = start_grid._candidate_array.copy()  # noqa: SLF001
        grid._candidate_list = list(start_grid._candidate_list)  # noqa: SLF001
        steps = 0
        for _ in range(iterations):
            while steps < max_steps and grid.step_nutrients() > start_grid._epsilon:  # noqa: SLF001
                steps += 1
            grid.grow_candidate()
        all_steps.append(steps)
    axes.plot(all_omega, all_steps, label=label)


def plot_omega() -> None:
    """Plot the influence of optimizing omega for the SOR solver.

    Tests a range of omega values (1.75 to 2.0) at different iteration
    counts to demonstrate SOR convergence behavior. Results are plotted
    with multiple iteration ranges for comparison.
    """
    fig = plt.figure()
    axes = fig.subplots(ncols=1)
    axes.set_xlabel(r"$\omega$")
    axes.set_ylabel("Steps")

    all_iterations = [100, 500, 1000]
    grid_size = 100
    iterations = 50
    max_steps = 100000
    epsilon = 1e-8
    all_omega = np.linspace(1.75, 2, 20)

    start_grid = create_dla(grid_size, 1, 1.95, epsilon)
    prev_iterations = 0
    for start_iterations in all_iterations:
        for _ in tqdm(range(start_iterations - prev_iterations)):
            start_grid.step()
        plot_omega_steps(
            start_grid,
            iterations,
            max_steps,
            all_omega,
            axes,
            f"{start_iterations}-{start_iterations + iterations}",
        )
        prev_iterations = start_iterations
    plt.legend()
    plt.show()


def plot_eta_path() -> None:
    """Plot the growth path with different eta bias values.

    Visualizes the time-dependent growth patterns for eta values of 0, 1,
    and 2. Data is cached to avoid recomputation. Shows how nutrient bias
    affects the spatial distribution of growth over time.
    """
    grid_size = 100
    fig = plt.figure()
    axes = fig.subplots(ncols=3)

    axis: Axes = axes[0]
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_title(r"Growth path with $\eta = 0$")

    axis = axes[1]
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_title(r"Growth path with $\eta = 1$")

    axis = axes[2]
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_title(r"Growth path with $\eta = 2$")

    path = np.zeros((3, grid_size, grid_size))

    # Use cached version or compute anew
    cached_path = data_dir / "eta_time_data.npy"
    if cached_path.exists():
        cached = np.load(cached_path)
        for i, avg in enumerate(cached):
            axes[i].imshow(avg, vmin=0, vmax=1)
    else:
        for i, eta in enumerate(np.linspace(0, 2, 3)):
            runs = 10
            for j in range(runs):
                grid = create_dla(grid_size, eta, 1.9, 1e-5, j)
                iterations = 1000
                for iteration in tqdm(range(iterations)):
                    grid.step()
                    path[i] = np.maximum(
                        path[i],
                        grid.growths * (1 - iteration / iterations),
                    )
            axes[i].imshow(path[i], vmin=0, vmax=10)
        np.save(cached_path, path)

    # Add a single colorbar with legend labels
    cbar = fig.colorbar(
        axes[1].images[0],
        ax=axes,
        orientation="horizontal",
        pad=0.1,
        aspect=30,
    )
    cbar.ax.set_xticks([1, 0])
    cbar.ax.set_xticklabels(["Start", "End"])

    plt.show()


def plot_eta() -> None:
    """Plot the average growth with different eta bias values.

    Visualizes the average growth patterns for eta values of 0, 1, and 2
    over 1000 iterations with 10 runs each. Data is cached to avoid
    recomputation. Shows how nutrient bias affects the overall growth
    distribution.
    """
    grid_size = 100
    fig = plt.figure()
    axes = fig.subplots(ncols=3)

    axis: Axes = axes[0]
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_title(r"Average growth with $\eta = 0$")

    axis = axes[1]
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_title(r"Average growth with $\eta = 1$")

    axis = axes[2]
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_title(r"Average growth with $\eta = 2$")

    avg = np.zeros((3, grid_size, grid_size))

    cached_path = data_dir / "eta_data.npy"
    if cached_path.exists():
        cached = np.load(cached_path)
        for i, avg in enumerate(cached):
            axes[i].imshow(avg, vmin=0, vmax=10)
    else:
        for i, eta in enumerate(np.linspace(0, 2, 3)):
            runs = 10
            for j in range(runs):
                grid = create_dla(grid_size, eta, 1.9, 1e-5, j)
                iterations = 1000
                for _ in tqdm(range(iterations)):
                    grid.step()
                avg[i] += grid.growths
            axes[i].imshow(avg[i], vmin=0, vmax=10)
        np.save(cached_path, avg)

    # Add a single colorbar with legend labels
    cbar = fig.colorbar(
        axes[1].images[0],
        ax=axes,
        orientation="horizontal",
        pad=0.1,
        aspect=30,
    )
    cbar.ax.set_xticks([10, 0])
    cbar.ax.set_xticklabels(["Always reached", "Never reached"])

    plt.show()


def gather_small() -> None:
    """Benchmark small grid (100x100) performance over 100 steps.

    Measures execution time for 100 steps on a 100x100 grid over 100
    iterations. Results are written in the data directory to 'no_mp.csv',
    "numba_mp.csv" or "manual_mp.csv" dependent on the type of parallelization.
    """
    if parallelization == "numba":
        data_path = data_dir / "numba_mp.csv"
    elif parallelization == "manual":
        data_path = data_dir / "manual_mp.csv"
    else:
        data_path = data_dir / "no_mp.csv"
    grid_size = 100
    epsilon = 1e-8
    steps = 100
    iterations = 100
    latencies = []
    for i in tqdm(range(iterations)):
        grid = create_dla(grid_size, 1, 1.95, epsilon, i)
        begin_time = time.time()
        for _ in range(steps):
            grid.step()
        latencies.append([time.time() - begin_time])
    with data_path.open("w") as f:
        writer = csv.writer(f)
        writer.writerows(latencies)


def gather_large() -> None:
    """Benchmark large grid (1000x1000) performance over 100 steps.

    Measures execution time for 100 steps on a 1000x1000 grid over 10
    iterations. Results are written in the data directory to 'no_mp_large.csv',
    "numba_mp_large.csv" or "manual_mp_large.csv" dependent on the type of
    parallelization.
    """
    if parallelization == "numba":
        data_path = data_dir / "numba_mp_large.csv"
    elif parallelization == "manual":
        data_path = data_dir / "manual_mp_large.csv"
    else:
        data_path = data_dir / "no_mp_large.csv"
    grid_size = 1000
    epsilon = 1e-6
    steps = 100
    iterations = 10
    latencies = []
    for i in tqdm(range(iterations)):
        grid = create_dla(grid_size, 1, 1.96, epsilon, i)
        begin_time = time.time()
        for _ in range(steps):
            grid.step()
        latencies.append([time.time() - begin_time])
    with data_path.open("w") as f:
        writer = csv.writer(f)
        writer.writerows(latencies)


def plot_small() -> None:
    """Plot performance comparison for small grids (100x100).

    Reads timing data from CSV files (no_mp.csv, numba_mp.csv,
    manual_mp.csv) and displays a bar chart with error bars comparing
    the three parallelization approaches for 100-step benchmarks.
    """
    # Data reading
    with (data_dir / "no_mp.csv").open("r") as f:
        reader = csv.reader(f)
        data = [float(x[0]) for x in reader]
        no_mp_mean = np.mean(data)
        no_mp_std = np.std(data)

    with (data_dir / "numba_mp.csv").open("r") as f:
        reader = csv.reader(f)
        data = [float(x[0]) for x in reader]
        numba_mp_mean = np.mean(data)
        numba_mp_std = np.std(data)

    with (data_dir / "manual_mp.csv").open("r") as f:
        reader = csv.reader(f)
        data = [float(x[0]) for x in reader]
        manual_mp_mean = np.mean(data)
        manual_mp_std = np.std(data)

    # Plotting
    fig = plt.figure(figsize=(12, 6))
    axes = fig.subplots(ncols=1)
    labels = ["Control", "Multiprocessing", "Numba"]
    means = [no_mp_mean, manual_mp_mean, numba_mp_mean]
    stds = [no_mp_std, manual_mp_std, numba_mp_std]

    # Create bar chart with nice colors
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]
    axes.bar(labels, means, color=colors, alpha=0.8, edgecolor="black", linewidth=1.5)

    # Add error bars
    axes.errorbar(
        labels,
        means,
        stds,
        fmt="none",
        ecolor="black",
        capsize=5,
        capthick=2,
        elinewidth=2,
    )

    # Styling
    axes.grid(axis="y", alpha=0.3, linestyle="--")
    axes.set_title(
        "Average computation time for 100 DLA steps",
        fontsize=14,
    )
    axes.set_ylabel("Time (seconds)", fontsize=12)

    plt.tight_layout()
    plt.show()


def plot_large() -> None:
    """Plot performance comparison for large grids (1000x1000).

    Reads timing data from CSV files (no_mp_large.csv, numba_mp_large.csv,
    manual_mp_large.csv) and displays a bar chart with error bars comparing
    the three parallelization approaches for 100-step benchmarks.
    """
    # Data reading
    with (data_dir / "no_mp_large.csv").open("r") as f:
        reader = csv.reader(f)
        data = [float(x[0]) for x in reader]
        no_mp_mean = np.mean(data)
        no_mp_std = np.std(data)

    with (data_dir / "numba_mp_large.csv").open("r") as f:
        reader = csv.reader(f)
        data = [float(x[0]) for x in reader]
        numba_mp_mean = np.mean(data)
        numba_mp_std = np.std(data)

    with (data_dir / "manual_mp_large.csv").open("r") as f:
        reader = csv.reader(f)
        data = [float(x[0]) for x in reader]
        manual_mp_mean = np.mean(data)
        manual_mp_std = np.std(data)

    # Plotting
    fig = plt.figure(figsize=(12, 6))
    axes = fig.subplots(ncols=1)
    labels = ["Control", "Multiprocessing", "Numba"]
    means = [no_mp_mean, manual_mp_mean, numba_mp_mean]
    stds = [no_mp_std, manual_mp_std, numba_mp_std]

    # Create bar chart with nice colors
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]
    axes.bar(labels, means, color=colors, alpha=0.8, edgecolor="black", linewidth=1.5)

    # Add error bars
    axes.errorbar(
        labels,
        means,
        stds,
        fmt="none",
        ecolor="black",
        capsize=5,
        capthick=2,
        elinewidth=2,
    )

    # Styling
    axes.grid(axis="y", alpha=0.3, linestyle="--")
    axes.set_title(
        "Average computation time for 100 DLA steps",
        fontsize=14,
    )
    axes.set_ylabel("Time (seconds)", fontsize=12)

    plt.tight_layout()
    plt.show()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for assignment selection.

    Returns:
        Namespace containing 'assignment' (plot/gather function name) and
        'parallelization' (backend: 'none', 'numba', or 'manual').

    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "assignment",
        help="The part of the assignment to plot",
        choices=[
            "omega",
            "eta",
            "eta_path",
            "gather_small",
            "gather_large",
            "plot_small",
            "plot_large",
            "all",
        ],
    )
    parser.add_argument(
        "parallelization",
        help="Which type of parallelization is applied",
        choices=["none", "numba", "manual"],
    )
    return parser.parse_args()


def main() -> None:
    """Execute the selected assignment analysis or benchmarking task.

    Parses command-line arguments to determine which analysis to run
    (omega/eta plots, eta path visualization, or performance gathering/
    plotting) and applies the selected parallelization backend.
    """
    global parallelization  # noqa: PLW0603

    args = parse_args()
    parallelization = args.parallelization
    assignment = args.assignment
    if assignment == "omega":
        plot_omega()
    elif assignment == "eta":
        plot_eta()
    elif assignment == "eta_path":
        plot_eta_path()
    elif assignment == "gather_small":
        gather_small()
    elif assignment == "gather_large":
        gather_large()
    elif assignment == "plot_small":
        plot_small()
    elif assignment == "plot_large":
        plot_large()
    elif assignment == "all":
        print(
            "Warning some of these plots may take long to compute, ",
            "calling the individual plotting functions is recommended.",
        )
        print("Omega")
        plot_omega()
        print("Eta")
        plot_eta()
        print("Eta path")
        plot_eta_path()
        print("Timing small")
        plot_small()
        print("Timing Large")
        plot_large()


if __name__ == "__main__":
    main()
