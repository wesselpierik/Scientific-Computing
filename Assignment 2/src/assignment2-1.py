import argparse

import dla
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def plot_omega() -> None:
    """Plot the influence of optimizing omega for the SOR solver."""
    pass


def plot_eta() -> None:
    """Plot the influence of choosing a higher or lower eta value."""
    for eta in np.linspace(0, 2, 3):
        avg = np.zeros((100, 100))
        runs = 10
        for i in range(runs):
            grid = dla.DLA(100, eta, epsilon=1e-5, workers=1, seed=i)
            iterations = 1000
            for _ in tqdm(range(iterations)):
                grid.step()
            avg += grid.growths
    plt.imshow(grid.growths)
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "assignment",
        help="The part of the assignment to plot",
        choices=["omega", "eta"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assignment = args.assignment
    if assignment == "omega":
        plot_omega()
    elif assignment == "eta":
        plot_eta()


if __name__ == "__main__":
    main()
