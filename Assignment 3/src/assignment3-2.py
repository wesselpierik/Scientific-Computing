"""Assignment 3: Wi-Fi router placement optimization

Group:         10
Course:        Scientific Computing

Description:
-----------
This module provides tools for simulating Wi-Fi signal strength in a 2D floor
plan and optimizing router placement. It uses the finite difference
method to solve the Helmholtz equation, implemented in the `helmholtz` module.

To run a single simulatioin for a specific router location, use the command
line arguments `--x` and `--y` to specify the router's position in meters.
If these arguments are not provided, the module will run simulations for all
valid router locations and save the results to a CSV file.
"""

import argparse

import helmholtz


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the Helmholtz simulation.

    Returns:
        argparse.Namespace: Parsed argument namespace.

    """
    parser = argparse.ArgumentParser(
        description="Simulate WiFi signal strength for router placements"
    )
    parser.add_argument(
        "--wave-number",
        type=float,
        default=50.3,
        help="Wave number k (default: 50.3 rad/m)",
    )
    parser.add_argument(
        "--domain-x",
        type=float,
        default=10.0,
        help="Domain size in x-direction (default: 10 meters)",
    )
    parser.add_argument(
        "--domain-y",
        type=float,
        default=8.0,
        help="Domain size in y-direction (default: 8 meters)",
    )
    parser.add_argument(
        "--amplitude",
        type=float,
        default=1e4,
        help="Source amplitude (default: 1e4)",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.2,
        help="Source width (standard deviation) in meters (default: 0.2)",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=100,
        help="Scale factor for grid resolution (default: 100). "
        "Larger values increase resolution and computation time.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help=(
            "Directory where PNG files are saved "
            "(default: current directory)"
        ),
    )
    parser.add_argument(
        "--results-file",
        type=str,
        default="results.txt",
        help="Path to output results file (default: results.txt)",
    )
    parser.add_argument(
        "--x",
        type=float,
        default=None,
        help="Optional router x-position for running one location",
    )
    parser.add_argument(
        "--y",
        type=float,
        default=None,
        help="Optional router y-position for running one location",
    )

    return parser.parse_args()


def main() -> None:
    """Parse CLI args and run the Helmholtz simulation module."""
    args = parse_arguments()

    if (args.x is None) != (args.y is None):
        raise ValueError("Provide both --x and --y, or neither.")

    helmholtz.main(
        k=args.wave_number,
        Nx=int(10 * args.scale),
        Ny=int(8 * args.scale),
        Lx=args.domain_x,
        Ly=args.domain_y,
        amplitude=args.amplitude,
        sigma=args.sigma,
        scale=args.scale,
        output_dir=args.output_dir,
        results_file=args.results_file,
        router_x=args.x,
        router_y=args.y,
    )


if __name__ == "__main__":
    main()
