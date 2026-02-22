import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
import numba
import argparse

L = 10
N = 1000
c = 1
dx = L / N
dt = 0.999 * dx / c
T = 10
num_ts = int(T / dt)


@numba.njit(cache=True)
def update_euler(us, vs, prev_t, t, c_local, dx_local, dt_local):
    constant = c_local**2 / dx_local**2 * dt_local
    n_points = us.shape[0]
    for i in numba.prange(1, n_points - 1):
        # Velocity update
        acceleration = constant * (
            us[i - 1, prev_t] - 2 * us[i, prev_t] + us[i + 1, prev_t]
        )
        vs[i] = vs[i] + acceleration

        # Position update
        us[i, t] = us[i, prev_t] + vs[i] * dt_local

    us[0, t] = 0
    us[-1, t] = 0


@numba.njit(cache=True, fastmath=True)
def update_leapfrog(us, vs, prev_t, t, c_local, dx_local, dt_local):
    constant = c_local**2 / dx_local**2 * dt_local / 2
    n_points = us.shape[0]

    for i in numba.prange(1, n_points - 1):
        # Update velocities
        acceleration = constant * (
            us[i - 1, prev_t] - 2 * us[i, prev_t] + us[i + 1, prev_t]
        )
        vs[i] = vs[i] + acceleration

        # Update positions
        us[i, t] = us[i, prev_t] + vs[i] * dt_local

    for i in numba.prange(1, n_points - 1):
        # Update velocities
        acceleration = constant * (us[i - 1, t] - 2 * us[i, t] + us[i + 1, t])
        vs[i] = vs[i] + acceleration

    us[0, t] = 0
    us[-1, t] = 0


@numba.njit(cache=True, fastmath=True)
def compute_energy(us, vs, t):
    """
    Compute total discrete energy at time index t
    """
    kinetic = 0.0
    potential = 0.0

    # Kinetic energy
    kinetic = 0.5 * np.sum(vs[1:-1] * vs[1:-1])

    # Potential energy (strain)
    for i in range(0, N - 1):
        du = (us[i + 1, t] - us[i, t]) / dx
        potential += 0.5 * c * c * du * du

    return kinetic + potential


def init_function(xs, a, start=0, end=1) -> np.ndarray:
    return np.sin(a * np.pi * xs) * (xs < end) * (xs > start)


def initialize(L, N, num_ts: int, init_fun, a, start=0, end=1):
    xs = np.linspace(0, L, N)
    arr = np.empty((*xs.shape, num_ts))
    arr[:, 0] = init_fun(xs, a, start, end)
    arr[:, 1] = init_fun(xs, a, start, end)
    return arr


def initialize_small(L, N, num_ts: int, init_fun, a, start=0, end=1):
    xs = np.linspace(0, L, N)
    arr = np.empty((*xs.shape, 2))
    arr[:, 0] = init_fun(xs, a, start, end)
    return arr


def animation(ax, arr, color="g"):
    print(arr.shape)
    artists = [ax.plot(state, color) for state in arr.T]

    return artists


def multi_animation(ax, arr1, arr2, xs=None):
    artists = []
    legend = None
    for state1, state2 in zip(arr1.T, arr2.T):
        if xs is None:
            (line_1,) = ax.plot(
                state1, color="g", linewidth=2, label="Leapfrog"
            )
            (line_2,) = ax.plot(
                state2,
                color="b",
                linestyle="--",
                alpha=0.8,
                label="Euler",
            )
        else:
            (line_1,) = ax.plot(
                xs, state1, color="g", linewidth=2, label="Leapfrog"
            )
            (line_2,) = ax.plot(
                xs,
                state2,
                color="b",
                linestyle="--",
                alpha=0.8,
                label="Euler",
            )
        if legend is None:
            legend = ax.legend(loc="upper right")
        if legend is None:
            artists.append([line_1, line_2])
        else:
            artists.append([line_1, line_2, legend])

    return artists


def run_case(
    initial_state, max_step, snapshots_needed, c_local, dx_local, dt_local
):
    snapshots = {0: initial_state.copy()}

    us = np.empty((initial_state.shape[0], 2), dtype=np.float64)
    us[:, 0] = initial_state
    us[:, 1] = initial_state
    us[0, 0] = 0.0
    us[-1, 0] = 0.0
    us[0, 1] = 0.0
    us[-1, 1] = 0.0

    vs = np.zeros(initial_state.shape[0], dtype=np.float64)
    prev_t, next_t = 0, 1
    for step in range(1, max_step + 1):
        update_euler(us, vs, prev_t, next_t, c_local, dx_local, dt_local)

        if step in snapshots_needed:
            snapshots[step] = us[:, next_t].copy()

        prev_t, next_t = next_t, prev_t

    return snapshots


def assignment_b():
    c_local = 1
    dt_local = 0.001
    x_min, x_max = 0, 1
    nx = 1000
    xs = np.linspace(x_min, x_max, nx)
    dx_local = xs[1] - xs[0]

    # Time snapshots to show in each panel
    plot_times = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    snapshot_indices = np.round(plot_times / dt_local).astype(np.int64)
    max_step = int(snapshot_indices.max())

    num_ts_local = max_step + 1
    ic_1 = initialize(1, nx, num_ts_local, init_function, 2)[:, 0]
    ic_2 = initialize(1, nx, num_ts_local, init_function, 5)[:, 0]
    ic_3 = initialize(
        1,
        nx,
        num_ts_local,
        init_function,
        5,
        start=0.2,
        end=0.4,
    )[:, 0]

    initial_conditions = [
        ("$\\Psi(x,0)=\\sin(2\\pi x)$", ic_1),
        ("$\\Psi(x,0)=\\sin(5\\pi x)$", ic_2),
        (
            "$\\Psi(x,0)=\\sin(5\\pi x)$ if "
            "$\\frac{1}{5}<x<\\frac{2}{5}$, else 0",
            ic_3,
        ),
    ]

    snapshots_needed = set(snapshot_indices.tolist())
    cmap = plt.cm.viridis(np.linspace(0.1, 0.95, len(plot_times)))

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    for ax, (title, ic) in zip(axes, initial_conditions):
        snapshots = run_case(
            ic,
            max_step,
            snapshots_needed,
            c_local,
            dx_local,
            dt_local,
        )
        for idx, color, t_plot in zip(snapshot_indices, cmap, plot_times):
            ax.plot(
                xs,
                snapshots[int(idx)],
                color=color,
                label=f"t = {t_plot:.1f}",
            )

        ax.set_title(title)
        ax.set_ylabel("$\\Psi(x,t)$")
        ax.grid(alpha=0.25)

    axes[-1].set_xlabel("x")
    axes[0].legend(ncol=3, fontsize=9, loc="upper right")
    fig.suptitle("Wave equation time development (c=1, dt=0.001)")
    fig.tight_layout()
    fig.savefig("B.png", dpi=300, bbox_inches="tight")
    plt.show()


def simulate_method(
    update_method, initial_state, num_steps, c_local, dx_local, dt_local
):
    n_points = initial_state.shape[0]
    arr = np.empty((n_points, num_steps + 1), dtype=np.float64)
    arr[:, 0] = initial_state
    arr[0, 0] = 0.0
    arr[-1, 0] = 0.0

    us = np.empty((n_points, 2), dtype=np.float64)
    us[:, 0] = initial_state
    us[:, 1] = initial_state
    us[0, 0] = 0.0
    us[-1, 0] = 0.0
    us[0, 1] = 0.0
    us[-1, 1] = 0.0

    vs = np.zeros(n_points, dtype=np.float64)
    prev_t, next_t = 0, 1
    for step in range(1, num_steps + 1):
        update_method(us, vs, prev_t, next_t, c_local, dx_local, dt_local)
        arr[:, step] = us[:, next_t]
        prev_t, next_t = next_t, prev_t

    return arr


def assignment_c(save_animation=False):
    c_local = 1.0
    dt_local = 0.001
    x_min, x_max = 0.0, 1.0
    nx = 1000
    t_end = 2.0
    frame_stride = 5

    xs = np.linspace(x_min, x_max, nx)
    dx_local = xs[1] - xs[0]
    num_steps = int(t_end / dt_local)

    initial_state = initialize(
        1.0,
        nx,
        num_steps + 1,
        init_function,
        5,
        start=1.0 / 5.0,
        end=2.0 / 5.0,
    )[:, 0]

    arr_euler = simulate_method(
        update_euler,
        initial_state,
        num_steps,
        c_local,
        dx_local,
        dt_local,
    )
    arr_leapfrog = simulate_method(
        update_leapfrog,
        initial_state,
        num_steps,
        c_local,
        dx_local,
        dt_local,
    )

    arr_euler_frames = arr_euler[:, ::frame_stride]
    arr_leapfrog_frames = arr_leapfrog[:, ::frame_stride]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(x_min, x_max)
    y_bound = 1.1 * max(
        np.max(np.abs(arr_euler_frames)),
        np.max(np.abs(arr_leapfrog_frames)),
    )
    ax.set_ylim(-y_bound, y_bound)
    ax.set_title("Time development (Leapfrog vs Euler)")
    ax.set_xlabel("x")
    ax.set_ylabel("$\\Psi(x,t)$")

    artists = multi_animation(ax, arr_leapfrog_frames, arr_euler_frames, xs=xs)
    ani = ArtistAnimation(fig, artists=artists, interval=20, blit=True)

    if save_animation:
        ani.save("C.mp4", dpi=120)

    plt.show()


def energy_trace(
    update_method, initial_state, c_local, dx_local, dt_local, t_end
):
    num_steps = int(t_end / dt_local)
    n_points = initial_state.shape[0]

    us = np.empty((n_points, 2), dtype=np.float64)
    us[:, 0] = initial_state
    us[:, 1] = initial_state
    us[0, 0] = 0.0
    us[-1, 0] = 0.0
    us[0, 1] = 0.0
    us[-1, 1] = 0.0

    vs = np.zeros(n_points, dtype=np.float64)
    energies = np.empty(num_steps + 1, dtype=np.float64)
    energies[0] = compute_energy(us, vs, 0)

    prev_t, next_t = 0, 1
    for step in range(1, num_steps + 1):
        update_method(us, vs, prev_t, next_t, c_local, dx_local, dt_local)
        energies[step] = compute_energy(us, vs, next_t)
        prev_t, next_t = next_t, prev_t

    times = np.linspace(0.0, num_steps * dt_local, num_steps + 1)
    return times, energies


def assignment_optional():
    c_local = 1.0
    x_min, x_max = 0.0, 1.0
    nx = 1000
    t_end = 2.0
    dt_values = [0.0005, 0.001, 0.0010010025]

    xs = np.linspace(x_min, x_max, nx)
    dx_local = xs[1] - xs[0]
    initial_state = initialize(
        1,
        nx,
        2,
        init_function,
        5,
        start=1 / 5,
        end=2 / 5,
    )[:, 0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    method_specs = [
        ("Euler", update_euler, axes[0]),
        ("Leapfrog", update_leapfrog, axes[1]),
    ]

    for method_name, method, ax in method_specs:
        for dt_local in dt_values:
            times, energies = energy_trace(
                method,
                initial_state,
                c_local,
                dx_local,
                dt_local,
                t_end,
            )
            ax.plot(times, energies, label=f"dt = {dt_local}")

        ax.set_title(method_name)
        ax.set_xlabel("t")
        ax.set_yscale("log")
        ax.set_ylim(1e-6, 1e10)
        ax.grid(alpha=0.25)
        ax.legend()

    axes[0].set_ylabel("Total energy")
    fig.suptitle("Total energy vs time for different dt")
    fig.tight_layout()
    fig.savefig("optional.png", dpi=300, bbox_inches="tight")
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "assignment",
        help="The identifier of the relevant sub-assignment.",
        type=str,
        choices=["B", "C", "Optional"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assignment = args.assignment
    match assignment:
        case "B":
            assignment_b()
        case "C":
            assignment_c()
        case "Optional":
            assignment_optional()


if __name__ == "__main__":
    main()
