import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
import numba
import tqdm


L = 1
N = 1000
c = 1
dx = L / N
dt = 0.001
T = 10
num_ts = int(T / dt)


@numba.njit(cache=True)
def update(us, vs, prev_t, t):
    constant = c**2 / dx**2 * dt
    for i in numba.prange(1, N - 1):
        # Velocity update
        acceleration = constant * (
            us[i - 1, prev_t] - 2 * us[i, prev_t] + us[i + 1, prev_t]
        )
        vs[i] = vs[i] + acceleration

        # Position update
        us[i, t] = us[i, prev_t] + vs[i] * dt

    us[0, t] = 0
    us[-1, t] = 0


@numba.njit(cache=True, fastmath=True)
def update_leapfrog(us, vs, prev_t, t):
    constant = c**2 / dx**2 * dt / 2

    for i in numba.prange(1, N - 1):
        # Update velocities
        acceleration = constant * (
            us[i - 1, prev_t] - 2 * us[i, prev_t] + us[i + 1, prev_t]
        )
        vs[i] = vs[i] + acceleration

        # Update positions
        us[i, t] = us[i, prev_t] + vs[i] * dt

    for i in numba.prange(1, N - 1):
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


def multi_animation(ax, arr1, arr2):
    artists = []
    legend = None
    for state1, state2 in zip(arr1.T, arr2.T):
        (line_1,) = ax.plot(state1, color="g", linewidth=2, label="Leapfrog")
        (line_2,) = ax.plot(state2, color="b", linestyle="--", alpha=0.8, label="Euler")
        if legend is None:
            legend = ax.legend(loc="upper right")
        if legend is None:
            artists.append([line_1, line_2])
        else:
            artists.append([line_1, line_2, legend])

    return artists


@numba.njit(cache=True, fastmath=True)
def max_abs(arr, col):
    max = 0
    for i in range(arr.shape[0]):
        v = arr[i, col]
        if v < 0:
            v = -v
        if v > max:
            max = v

    return max


@numba.njit(cache=True, fastmath=True)
def run_research_euler(arr, vs, prev_t, next_t):
    update(arr, vs, prev_t, next_t)
    return compute_energy(arr, vs, next_t)


@numba.njit(cache=True, fastmath=True)
def run_research_leapfrog(arr, vs, prev_t, next_t):
    update_leapfrog(arr, vs, prev_t, next_t)
    return compute_energy(arr, vs, next_t)


def research_stability_euler():
    arr = initialize_small(L, N, num_ts, init_function, 5, start=0.2, end=0.4)
    vs = np.zeros((N,))
    energy = np.empty(num_ts)
    energy[0] = compute_energy(arr, vs, 0)
    energy[1] = compute_energy(arr, vs, 1)
    tk0 = tqdm.tqdm(range(2, num_ts), total=num_ts - 2, disable=None)
    prev, next = 0, 1
    for t in tk0:
        energy[t] = run_research_euler(arr, vs, prev, next)
        prev, next = next, prev

    return energy


def research_stability_leapfrog():
    arr = initialize_small(L, N, num_ts, init_function, 5, start=0.2, end=0.4)
    vs = np.zeros((N,))
    energy = np.empty(num_ts)
    energy[0] = compute_energy(arr, vs, 0)
    energy[1] = compute_energy(arr, vs, 1)
    tk0 = tqdm.tqdm(range(2, num_ts), total=num_ts - 2, disable=None)
    prev, next = 0, 1
    for t in tk0:
        energy[t] = run_research_leapfrog(arr, vs, prev, next)
        prev, next = next, prev

    return energy


if __name__ == "__main__":
    arr_leapfrog = initialize(L, N, num_ts, init_function, 2, start=0.2, end=0.4)
    vs = np.zeros((N,))
    tk0 = tqdm.tqdm(range(2, num_ts), total=num_ts - 2, disable=None)
    for t in tk0:
        update_leapfrog(arr_leapfrog, vs, t - 1, t)

    arr_euler = initialize(L, N, num_ts, init_function, 2, start=0.2, end=0.4)
    vs = np.zeros((N,))
    tk0 = tqdm.tqdm(range(2, num_ts), total=num_ts - 2, disable=None)
    for t in tk0:
        update(arr_euler, vs, t - 1, t)

    euler_energies = research_stability_euler()

    leapfrog_energies = research_stability_leapfrog()

    plt.figure()
    plt.title("Total Energy over time")
    plt.plot(euler_energies, label="Euler Total Energy")
    plt.plot(leapfrog_energies, label="Leapfrog Total Energy")
    plt.xlabel("Time steps")
    plt.ylabel("Total Energy")
    plt.legend()
    plt.tight_layout()
    plt.show()

    fig = plt.figure()
    ax = fig.subplots()

    artists = multi_animation(ax, arr_leapfrog, arr_euler)
    ani = ArtistAnimation(fig, artists=artists, interval=16, blit=True)

    print(np.max(arr_euler), np.max(arr_leapfrog))

    plt.show()
