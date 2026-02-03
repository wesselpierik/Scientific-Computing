import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from time_independent import SOR, BaseGrid, GaussSeidel, Jacobi


def concentration_curve(grid: BaseGrid, *, axis: Axes | None = None) -> None:
    if axis is None:
        axis = plt.subplot()

    checkpoints = 10 ** np.arange(0, 5)
    steps = 0
    for checkpoint in checkpoints:
        while steps < checkpoint:
            grid.step()
            steps += 1
        axis.plot(np.linspace(1, 0, grid.grid_size), grid.state[:, 0])


def assignment_h() -> None:
    grid_size = 50
    jacobi = Jacobi(grid_size)
    concentration_curve(jacobi)
    plt.show()


def assignment_i() -> None:
    pass


def assignment_j() -> None:
    pass


def assignment_k() -> None:
    pass


def assignment_l() -> None:
    pass


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
