import argparse
import csv
import time
from pathlib import Path
from typing import TYPE_CHECKING

import dla
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

if TYPE_CHECKING:
    from matplotlib.axes import Axes

local_dir = Path(__file__).parent


def plot_omega() -> None:
    """Plot the influence of optimizing omega for the SOR solver."""
    grid_size = 100
    iterations = 50
    epsilon = 1e-8
    max_steps = 100000
    all_steps = []
    all_omega = np.linspace(1.75, 2, 20)
    for omega in tqdm(all_omega):
        grid = dla.DLA(grid_size, eta=1, omega=omega)
        steps = 0
        while steps < max_steps and grid.step_nutrients() > epsilon:
            steps += 1
        steps = 0
        for _ in range(iterations):
            while steps < max_steps and grid.step_nutrients() > epsilon:
                steps += 1
            grid.grow_candidate()
        all_steps.append(steps)
    fig = plt.figure()
    axes = fig.subplots(ncols=1)
    axes.set_title("Step count with SOR scheme")
    axes.set_xlabel(r"$\omega$")
    axes.set_ylabel("Steps")
    axes.plot(all_omega, all_steps)
    plt.show()


def plot_eta() -> None:
    """Plot the influence of choosing a higher or lower eta value."""
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

    cached_path = local_dir / "eta_data.npy"
    if cached_path.exists():
        cached = np.load(cached_path)
        for i, avg in enumerate(cached):
            axes[i].imshow(avg)
    else:
        for i, eta in enumerate(np.linspace(0, 2, 3)):
            runs = 10
            for j in range(runs):
                grid = dla.DLA(
                    grid_size,
                    eta,
                    epsilon=1e-5,
                    workers=1,
                    seed=j,
                    omega=1.8,
                )
                iterations = 1000
                for _ in tqdm(range(iterations)):
                    grid.step()
                avg[i] += grid.growths
            axes[i].imshow(avg[i])
        np.save(cached_path, avg)
    plt.show()


def gather_small() -> None:
    """Gather the time it takes for the grid to complete 100 steps.

    Due to the simplicity of this function, it requires some hardcoding to
    try out different DLA configurations
    """
    data_path = local_dir / "small_data.csv"
    grid_size = 100
    epsilon = 1e-8
    steps = 100
    iterations = 100
    latencies = []
    for i in tqdm(range(iterations)):
        grid = dla.DLA(grid_size, 1, epsilon=epsilon, omega=1.9, seed=i, workers=1)
        begin_time = time.time()
        for _ in range(steps):
            grid.step()
        latencies.append([time.time() - begin_time])
    with data_path.open("w") as f:
        writer = csv.writer(f)
        writer.writerows(latencies)


def plot_small() -> None:
    """Plot the time data from the gather_small function.

    Requires the no_mp.csv, numba_mp.csv and manual_mp.csv to be present.
    """
    # Data reading
    with (local_dir / "no_mp.csv").open("r") as f:
        reader = csv.reader(f)
        data = [float(x[0]) for x in reader]
        no_mp_mean = np.mean(data)
        no_mp_std = np.std(data)

    with (local_dir / "numba_mp.csv").open("r") as f:
        reader = csv.reader(f)
        data = [float(x[0]) for x in reader]
        numba_mp_mean = np.mean(data)
        numba_mp_std = np.std(data)

    with (local_dir / "manual_mp.csv").open("r") as f:
        reader = csv.reader(f)
        data = [float(x[0]) for x in reader]
        manual_mp_mean = np.mean(data)
        manual_mp_std = np.std(data)

    # Plotting
    fig = plt.figure()
    axes = fig.subplots(ncols=1)
    labels = ["Control", "Multiprocessing", "Numba"]
    means = [no_mp_mean, manual_mp_mean, numba_mp_mean]
    axes.bar(labels, means)
    # TODO: Error bars


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "assignment",
        help="The part of the assignment to plot",
        choices=["omega", "eta", "gather_small"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assignment = args.assignment
    if assignment == "omega":
        plot_omega()
    elif assignment == "eta":
        plot_eta()
    elif assignment == "gather_small":
        gather_small()


if __name__ == "__main__":
    main()
