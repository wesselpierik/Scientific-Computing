"""Assignment 2: Gray-Scott Simulation and Visualization

Group:         10
Course:        Scientific Computing

Description:
-----------
This module provides tools for simulating and visualizing the Gray-Scott
reaction-diffusion systems. It utilizes the `GrayScott` class to represent
and evolve the concentration of two chemicals (U and V) on a 2D grid.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from gray_scott import GrayScott


def make_update(
    fig,
    simulations_by_f,
    images_by_f,
    steps_per_frame: int,
    snapshot_step: int,
):
    """Create the animation callback used by ``FuncAnimation``.

    Args:
        fig: Matplotlib figure used for title updates and snapshot export.
        simulations_by_f: Nested list of ``GrayScott`` simulations grouped by
            feed-rate row.
        images_by_f: Nested list of image artists corresponding to
            ``simulations_by_f``.
        steps_per_frame: Number of simulation steps computed per animation
            frame.
        snapshot_step: First simulation step at which a PNG snapshot is saved.

    Returns:
        A frame update function compatible with ``FuncAnimation``.

    """
    snapshot_saved = False

    def update(frame_index: int):
        """Advance simulations, refresh artists, and optionally save snapshot.

        Args:
            frame_index: Zero-based animation frame index.

        Returns:
            Tuple of updated image artists for Matplotlib redraw.

        """
        nonlocal snapshot_saved

        for _ in range(steps_per_frame):
            for simulations in simulations_by_f:
                for simulation in simulations:
                    simulation.step()

        for images, simulations in zip(images_by_f, simulations_by_f):
            for image, simulation in zip(images, simulations):
                image.set_data(simulation.state[:, :, 1])

        current_step = (frame_index + 1) * steps_per_frame
        fig.suptitle(
            f"Gray-Scott Model: V Concentration (Step {current_step})",
            fontsize=16,
            fontweight="bold",
        )

        if not snapshot_saved and current_step >= snapshot_step:
            fig.savefig(
                f"snapshot_t_{snapshot_step}.png",
                dpi=300,
                bbox_inches="tight",
            )
            snapshot_saved = True

        return tuple(image for images in images_by_f for image in images)

    return update


def assignment_e():
    grid_size = 200

    fs = [0.02, 0.04, 0.06, 0.08]
    ks = [np.linspace(0.045, 0.055, 4)]  # f = 0.02
    ks += [np.linspace(0.058, 0.063, 4)]  # f = 0.04
    ks += [np.linspace(0.06, 0.065, 4)]  # f = 0.06
    ks += [np.linspace(0.06, 0.0616, 4)]  # f = 0.08

    np.random.seed(43)

    square_size = 10
    steps_per_frame = 100
    snapshot_step = 50000
    total_steps = 1000000
    total_frames = total_steps // steps_per_frame

    simulations_by_f = [
        [GrayScott(grid_size, f, k, square_size) for k in ks_small]
        for f, ks_small in zip(fs, ks)
    ]
    shared_norm = plt.Normalize(vmin=0, vmax=0.8)

    fig, axes = plt.subplots(4, 4, figsize=(10, 10), layout="compressed")

    images_by_f = []

    for i, (f, simulations, ks_small) in enumerate(
        zip(fs, simulations_by_f, ks)
    ):
        images = []

        for ax, k, simulation in zip(axes[i, :], ks_small, simulations):
            image = ax.imshow(
                simulation.state[:, :, 1],
                cmap="inferno",
                norm=shared_norm,
                animated=True,
            )
            ax.set_title(
                f"k = {k:.5f}",
                fontsize=14,
                pad=10,
            )
            images.append(image)

        axes[i, 0].set_ylabel(
            f"f = {f:.2f}",
            fontsize=14,
            rotation=0,
            labelpad=32,
            va="center",
        )
        images_by_f.append(images)

    cbar = fig.colorbar(
        images_by_f[0][0],
        ax=axes,
        shrink=0.9,
        location="right",
    )
    cbar.set_label("V concentration")

    fig.suptitle(
        "Gray-Scott Model: V Concentration (Step 0)",
        fontsize=16,
        fontweight="bold",
    )

    update = make_update(
        fig,
        simulations_by_f,
        images_by_f,
        steps_per_frame,
        snapshot_step,
    )

    _ = FuncAnimation(
        fig,
        update,
        frames=total_frames,
        interval=16,
        blit=False,
    )

    plt.show()
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Function to parse the command line arguments.

    Returns:
        argparse.Namespace: namespace containing the parsed arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "option", help="Determine the code to run", choices=["E"]
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    option = args.option

    if option == "E":
        assignment_e()
