import argparse


def plot_omega() -> None:
    """Plot the influence of optimizing omega for the SOR solver."""
    pass


def plot_eta() -> None:
    """Plot the influence of choosing a higher or lower eta value."""
    pass


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
