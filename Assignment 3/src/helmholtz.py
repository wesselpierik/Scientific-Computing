import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import numba
from multiprocessing import cpu_count, get_context
from functools import partial

import scipy as sp
from scipy.sparse.linalg import LinearOperator, gmres, spilu

# Try to import tqdm for progress bars, but allow it to be optional
# Required since tqdm is not available on the supercomputer cluster
try:
    from tqdm import tqdm

    TQDM = True
except ImportError:
    TQDM = False

# Global variables for shared data across worker processes
SHARED_A_EQ = None
SHARED_PRECONDITIONER = None
SHARED_ROW_SCALE_INV = None
SHARED_COL_SCALE = None
SHARED_NX = None
SHARED_NY = None
SHARED_HX = None
SHARED_HY = None

# Measurement locations in the room (x, y) in meters
MEASURE_LOCATIONS = {
    "living_room": (1, 5),
    "kitchen": (2, 1),
    "bedroom": (9, 7),
    "bathroom": (9, 1),
}


class gmres_counter(object):
    def __init__(self, disp=True):
        self._disp = disp
        self.niter = 0

    def __call__(self, rk=None):
        self.niter += 1
        if self._disp:
            print(f"iter {self.niter:3d}\trk = {str(rk)}")


@numba.njit(cache=True)
def f(
    x: float, y: float, x_r: float, y_r: float, amplitude: float, sigma: float
) -> float:
    return amplitude * np.exp(-(((x - x_r)) ** 2 + ((y - y_r)) ** 2) / (2 * sigma**2))


@numba.njit(cache=True)
def idx(i: int, j: int, Nx: int) -> int:
    return i * Nx + j


def matrix_construction(
    A: sp.sparse.lil_matrix,
    n: np.ndarray,
    i: int,
    Nx: int,
    Ny: int,
    hx: float,
    hy: float,
    k: float,
) -> None:
    for j in range(Nx):
        p = idx(i, j, Nx)

        # Prefer 9-point Helmholtz stencil wherever all neighbors exist.
        if 0 < i < Ny - 1 and 0 < j < Nx - 1:
            # Isotropic 9-point Laplacian
            # (assumes near-uniform spacing hx ~= hy)
            inv_h2 = 1.0 / (hx * hy)
            A[p, p] = -20.0 / 6.0 * inv_h2 + k**2 * n[i, j] ** 2
            A[p, idx(i + 1, j, Nx)] = 4.0 / 6.0 * inv_h2
            A[p, idx(i - 1, j, Nx)] = 4.0 / 6.0 * inv_h2
            A[p, idx(i, j + 1, Nx)] = 4.0 / 6.0 * inv_h2
            A[p, idx(i, j - 1, Nx)] = 4.0 / 6.0 * inv_h2
            A[p, idx(i + 1, j + 1, Nx)] = 1.0 / 6.0 * inv_h2
            A[p, idx(i + 1, j - 1, Nx)] = 1.0 / 6.0 * inv_h2
            A[p, idx(i - 1, j + 1, Nx)] = 1.0 / 6.0 * inv_h2
            A[p, idx(i - 1, j - 1, Nx)] = 1.0 / 6.0 * inv_h2
        # Boundary conditions (Sommerfeld/Robin: du/dn = i*k*u)
        elif i == 0:
            if Ny > 2:
                # 2nd-order one-sided derivative: (-3u0 + 4u1 - u2)/(2*hy)
                A[p, p] = 3.0 / (2.0 * hy) - 1j * k
                A[p, idx(i + 1, j, Nx)] = -4.0 / (2.0 * hy)
                A[p, idx(i + 2, j, Nx)] = 1.0 / (2.0 * hy)
            else:
                # Fallback when a 3-point one-sided stencil is impossible.
                A[p, p] = 1.0 / hy - 1j * k
                A[p, idx(i + 1, j, Nx)] = -1.0 / hy
        elif i == Ny - 1:
            if Ny > 2:
                # 2nd-order one-sided derivative: (3uN - 4uN-1 + uN-2)/(2*hy)
                A[p, p] = 3.0 / (2.0 * hy) - 1j * k
                A[p, idx(i - 1, j, Nx)] = -4.0 / (2.0 * hy)
                A[p, idx(i - 2, j, Nx)] = 1.0 / (2.0 * hy)
            else:
                A[p, p] = 1.0 / hy - 1j * k
                A[p, idx(i - 1, j, Nx)] = -1.0 / hy
        elif j == 0:
            if Nx > 2:
                # 2nd-order one-sided derivative: (-3u0 + 4u1 - u2)/(2*hx)
                A[p, p] = 3.0 / (2.0 * hx) - 1j * k
                A[p, idx(i, j + 1, Nx)] = -4.0 / (2.0 * hx)
                A[p, idx(i, j + 2, Nx)] = 1.0 / (2.0 * hx)
            else:
                A[p, p] = 1.0 / hx - 1j * k
                A[p, idx(i, j + 1, Nx)] = -1.0 / hx
        elif j == Nx - 1:
            if Nx > 2:
                # 2nd-order one-sided derivative: (3uN - 4uN-1 + uN-2)/(2*hx)
                A[p, p] = 3.0 / (2.0 * hx) - 1j * k
                A[p, idx(i, j - 1, Nx)] = -4.0 / (2.0 * hx)
                A[p, idx(i, j - 2, Nx)] = 1.0 / (2.0 * hx)
            else:
                A[p, p] = 1.0 / hx - 1j * k
                A[p, idx(i, j - 1, Nx)] = -1.0 / hx


@numba.njit(cache=True)
def build_source_vector(
    x_r: float,
    y_r: float,
    Nx: int,
    Ny: int,
    hx: float,
    hy: float,
    amplitude: float,
    sigma: float,
) -> np.ndarray:
    """Build RHS for a given router location.

    Args:
        x_r: Router x-position.
        y_r: Router y-position.
        Nx: Grid size in x-direction.
        Ny: Grid size in y-direction.
        hx: Grid spacing in x-direction.
        hy: Grid spacing in y-direction.
        amplitude: Source amplitude.
        sigma: Source width.

    Returns:
        np.ndarray: Complex RHS vector with source values on interior nodes
            and zero boundary values.
    """
    b = np.zeros(Nx * Ny, dtype=np.complex128)
    for i in range(1, Ny - 1):
        y = i * hy
        for j in range(1, Nx - 1):
            x = j * hx
            p = idx(i, j, Nx)
            b[p] = f(x, y, x_r, y_r, amplitude, sigma)
    return b


@numba.njit(cache=True)
def measure_signal_strength(
    u_2d: np.ndarray, i: int, j: int, hx: float, hy: float
) -> float:
    """Measure signal strength within 5cm radius of measurement point.

    Args:
        u_2d: 2D field of solution values.
        i: Grid index in y-direction.
        j: Grid index in x-direction.
        hx: Grid spacing in x-direction.
        hy: Grid spacing in y-direction.

    Returns:
        float: Average signal strength in dB within 5cm radius.
    """
    strength = 10 * np.log10(np.abs(u_2d) ** 2 / np.max(np.abs(u_2d) ** 2) + 1e-20)

    summed_strength = 0.0
    num_points = 0

    # Radius in meters: 5 centimeters
    radius = 0.05
    radius_squared = radius**2

    # Determine search range in grid indices
    max_di = int(radius / hy) + 1
    max_dj = int(radius / hx) + 1

    for di in range(-max_di, max_di + 1):
        for dj in range(-max_dj, max_dj + 1):
            # Check physical distance from center point
            physical_dist_sq = (di * hy) ** 2 + (dj * hx) ** 2
            if physical_dist_sq <= radius_squared:
                ni = i + di
                nj = j + dj
                if 0 <= ni < u_2d.shape[0] and 0 <= nj < u_2d.shape[1]:
                    summed_strength += strength[ni, nj]
                    num_points += 1

    return summed_strength / num_points if num_points > 0 else 0.0


@numba.njit(cache=True)
def create_wall(
    x_start: float,
    x_end: float,
    y_start: float,
    y_end: float,
    n: np.ndarray,
    Ny: int,
    Nx: int,
    hy: float,
    hx: float,
) -> None:
    val = 2.5 + 0.5j

    # Convert to index space once
    i_start = max(0, int(y_start / hy))
    i_end = min(Ny - 1, int(y_end / hy))

    j_start = max(0, int(x_start / hx))
    j_end = min(Nx - 1, int(x_end / hx))

    # Loop only over the relevant region
    for i in range(i_start, i_end + 1):
        for j in range(j_start, j_end + 1):
            n[i, j] = val


@numba.njit(cache=True)
def room_formation(Nx: int, Ny: int, Ly: float, Lx: float, n: np.ndarray) -> None:
    hx = Lx / (Nx - 1)
    hy = Ly / (Ny - 1)

    val = 2.5 + 0.5j

    for i in range(Ny):
        y = i * hy
        y_edge = (y < 0.15) or (y > 7.85)

        for j in range(Nx):
            if y_edge:
                n[i, j] = val
            else:
                x = j * hx
                if (x < 0.15) or (x > 9.85):
                    n[i, j] = val

    create_wall(2.425, 2.575, 0.15, 2.0, n, Ny, Nx, hy, hx)
    create_wall(6.925, 7.075, 0.15, 1.5, n, Ny, Nx, hy, hx)
    create_wall(6.925, 7.075, 2.5, 3.0, n, Ny, Nx, hy, hx)
    create_wall(0.15, 3, 2.925, 3.075, n, Ny, Nx, hy, hx)
    create_wall(4, 6, 2.925, 3.075, n, Ny, Nx, hy, hx)
    create_wall(7, 10, 2.925, 3.075, n, Ny, Nx, hy, hx)
    create_wall(5.925, 6.075, 3, 7.925, n, Ny, Nx, hy, hx)


def build_preconditioner(
    A_csc: sp.sparse.csc_array, A_csr: sp.sparse.csr_array
) -> LinearOperator:
    A_work = A_csc.copy()
    A_work.sum_duplicates()
    A_work.eliminate_zeros()
    A_work.sort_indices()

    # Shift the matrix by a small imaginary value to improve
    # ILU stability if needed.
    identity = sp.sparse.eye(A_work.shape[0], format="csc", dtype=A_work.dtype) * 1j
    attempts = [
        ("ILU preconditioner", A_work, 5e-4, 20),
        (
            "shifted ILU preconditioner (shift=1e-8j)",
            A_work + 1e-8 * identity,
            1e-3,
            30,
        ),
        (
            "shifted ILU preconditioner (shift=1e-6j)",
            A_work + 1e-6 * identity,
            1e-3,
            30,
        ),
        (
            "shifted ILU preconditioner (shift=1e-4j)",
            A_work + 1e-4 * identity,
            1e-2,
            40,
        ),
    ]

    for label, matrix, drop_tol, fill_factor in attempts:
        try:
            ilu = spilu(
                matrix,
                drop_tol=drop_tol,
                fill_factor=fill_factor,
                permc_spec="COLAMD",
                diag_pivot_thresh=0.01,
            )
            print(f"Using {label}")
            return LinearOperator(A_work.shape, matvec=ilu.solve, dtype=A_work.dtype)
        except RuntimeError as error:
            print(f"{label} failed: {error}")

    diag = A_csr.diagonal().copy()
    diag[np.abs(diag) < 1e-12] = 1.0
    print("ILU failed for all attempts; using Jacobi preconditioner")
    return LinearOperator(A_csr.shape, matvec=lambda x: x / diag, dtype=A_csr.dtype)


def equilibrate_system(
    A_csr: sp.sparse.csr_array, eps: float = 1e-14
) -> tuple[sp.sparse.csr_array, np.ndarray, np.ndarray]:
    """Equilibrate rows and columns once and reuse scaling for all RHS vectors.

    Args:
        A_csr: System matrix in CSR format.
        eps: Small positive guard to avoid division by zero.

    Returns:
        tuple[sp.sparse.csr_array, np.ndarray, np.ndarray]:
            Equilibrated matrix, inverse row scaling, and column scaling
            factors.
    """
    abs_A = np.abs(A_csr)

    row_norm = np.sqrt(np.asarray(abs_A.power(2).sum(axis=1)).ravel())
    row_norm[row_norm < eps] = 1.0
    row_scale_inv = 1.0 / row_norm

    col_norm = np.sqrt(np.asarray(abs_A.power(2).sum(axis=0)).ravel())
    col_norm[col_norm < eps] = 1.0

    D_row = sp.sparse.diags(row_scale_inv, format="csr")
    D_col = sp.sparse.diags(1.0 / col_norm, format="csr")

    A_eq = (D_row @ A_csr @ D_col).tocsr()

    return A_eq, row_scale_inv, col_norm


def solve_GMRES(
    A_csr,
    rhs,
    preconditioner,
    counter,
    rtol=1e-8,
    atol=1e-12,
    maxiter=10000,
    restart=300,
) -> tuple[np.ndarray, int]:
    """Solve with GMRES.

    Args:
        A_csr: System matrix in CSR format.
        rhs: Right-hand side vector.
        preconditioner: LinearOperator preconditioner for GMRES.
        counter: Callback object tracking GMRES residual history.
        rtol: Relative tolerance for GMRES.
        atol: Absolute tolerance for GMRES.
        maxiter: Maximum GMRES iterations.
        restart: Restart parameter passed to GMRES.

    Returns:
        tuple[np.ndarray, int]: Solution vector and GMRES info code.
    """
    solution, info = gmres(
        A_csr,
        rhs,
        M=preconditioner,
        maxiter=maxiter,
        restart=restart,
        rtol=rtol,
        atol=atol,
        callback=counter,
    )

    return solution, info


def check_router_position(x_r: float, y_r: float, measure_locations: dict) -> bool:
    if (
        (0.15 <= x_r <= 2.425 and 0.15 <= y_r <= 2.0)
        or (6.925 <= x_r <= 7.075 and 0.15 <= y_r <= 1.5)
        or (6.925 <= x_r <= 7.075 and 2.5 <= y_r <= 3.0)
        or (0.15 <= x_r <= 3 and 2.925 <= y_r <= 3.075)
        or (4 <= x_r <= 6 and 2.925 <= y_r <= 3.075)
        or (7 <= x_r <= 10 and 2.925 <= y_r <= 3.075)
        or (5.925 <= x_r <= 6.075 and 3 <= y_r <= 7.925)
    ):
        return False

    for _, (x_m, y_m) in measure_locations.items():
        if (x_m - 0.5 <= x_r <= x_m + 0.5) and (y_m - 0.5 <= y_r <= y_m + 0.5):
            return False

    return True


def simulate_wifi_signal(
    x_r: float,
    y_r: float,
    amplitude: float,
    sigma: float,
    scale: int,
    output_dir: str = ".",
) -> tuple[float, float, float, float, float, float] | None:
    if not check_router_position(x_r, y_r, MEASURE_LOCATIONS):
        return None
    b = build_source_vector(
        x_r, y_r, SHARED_NX, SHARED_NY, SHARED_HX, SHARED_HY, amplitude, sigma
    )
    b_eq = SHARED_ROW_SCALE_INV * b

    counter = gmres_counter(disp=False)

    print("Created source vector, starting GMRES solve...")

    u_eq, info = solve_GMRES(
        SHARED_A_EQ,
        b_eq,
        preconditioner=SHARED_PRECONDITIONER,
        counter=counter,
        rtol=1e-7,
        atol=1e-12,
        maxiter=100,
    )

    print(f"GMRES finished with info={info} after {counter.niter} iterations")

    u = u_eq / SHARED_COL_SCALE

    u_2d = u.reshape((SHARED_NY, SHARED_NX))

    signal_living_room = measure_signal_strength(
        u_2d,
        int(MEASURE_LOCATIONS["living_room"][1] / SHARED_HY),
        int(MEASURE_LOCATIONS["living_room"][0] / SHARED_HX),
        SHARED_HX,
        SHARED_HY,
    )
    signal_bedroom = measure_signal_strength(
        u_2d,
        int(MEASURE_LOCATIONS["bedroom"][1] / SHARED_HY),
        int(MEASURE_LOCATIONS["bedroom"][0] / SHARED_HX),
        SHARED_HX,
        SHARED_HY,
    )
    signal_kitchen = measure_signal_strength(
        u_2d,
        int(MEASURE_LOCATIONS["kitchen"][1] / SHARED_HY),
        int(MEASURE_LOCATIONS["kitchen"][0] / SHARED_HX),
        SHARED_HX,
        SHARED_HY,
    )
    signal_bathroom = measure_signal_strength(
        u_2d,
        int(MEASURE_LOCATIONS["bathroom"][1] / SHARED_HY),
        int(MEASURE_LOCATIONS["bathroom"][0] / SHARED_HX),
        SHARED_HX,
        SHARED_HY,
    )

    plt.figure(figsize=(10, 8))
    plt.imshow(
        10 * np.log10(np.abs(u_2d) ** 2 / np.max(np.abs(u_2d) ** 2) + 1e-20),
        extent=[0, 10, 0, 8],
        origin="lower",
        cmap="jet",
        vmin=-40,
    )
    plt.colorbar(label="Signal Strength (dB)")
    plt.scatter(x_r, y_r, color="cyan", marker="x", s=100, label="Router")
    plt.title(f"WiFi Signal Strength (Router at ({x_r:.2f}, {y_r:.2f}))")
    plt.xlabel("x (meters)")
    plt.ylabel("y (meters)")
    plt.legend()
    output_path = f"{output_dir}/signal_{x_r:.2f}_{y_r:.2f}_{scale}.png"
    plt.savefig(output_path)
    plt.close()

    return (
        x_r,
        y_r,
        signal_living_room,
        signal_bedroom,
        signal_kitchen,
        signal_bathroom,
    )


def _worker_task(args, amplitude, sigma, scale, output_dir):
    """Worker task for multiprocessing pool.

    Args:
        args: Tuple of (x_r, y_r) coordinates.
        amplitude: Source amplitude.
        sigma: Source width.
        scale: Scale factor.
        output_dir: Output directory for PNG files.

    Returns:
        Result tuple or None if position invalid.
    """
    x_r, y_r = args
    return simulate_wifi_signal(x_r, y_r, amplitude, sigma, scale, output_dir)


def main(
    k: float,
    Nx: int,
    Ny: int,
    Lx: float,
    Ly: float,
    amplitude: float,
    sigma: float,
    scale: int,
    output_dir: str = ".",
    results_file: str = "results.txt",
) -> None:
    # Create output directory if it doesn't exist
    if output_dir != "." and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    n = np.ones((Ny, Nx), dtype=np.complex128)

    room_formation(Nx, Ny, Ly, Lx, n)

    hx = Lx / (Nx - 1)
    hy = Ly / (Ny - 1)

    N = Nx * Ny
    A = sp.sparse.lil_array((N, N), dtype=np.complex128)
    for i in range(Ny):
        matrix_construction(A, n, i, Nx, Ny, hx, hy, k)

    A_csr = A.tocsr()
    A_csr.sum_duplicates()
    A_csr.eliminate_zeros()
    A_eq, row_scale_inv, col_scale = equilibrate_system(A_csr)
    A_eq_csc = A_eq.tocsc()
    preconditioner = build_preconditioner(A_eq_csc, A_eq)

    global SHARED_A_EQ
    global SHARED_PRECONDITIONER
    global SHARED_ROW_SCALE_INV
    global SHARED_COL_SCALE
    global SHARED_NX
    global SHARED_NY
    global SHARED_HX
    global SHARED_HY

    SHARED_A_EQ = A_eq
    SHARED_PRECONDITIONER = preconditioner
    SHARED_ROW_SCALE_INV = row_scale_inv
    SHARED_COL_SCALE = col_scale
    SHARED_NX = Nx
    SHARED_NY = Ny
    SHARED_HX = hx
    SHARED_HY = hy

    # Generate all (x_r, y_r) coordinate pairs
    x_positions = np.linspace(0.3, 9.7, 95)
    y_positions = np.linspace(0.3, 7.7, 75)

    tasks = [(x_r, y_r) for x_r in x_positions for y_r in y_positions]

    num_workers = min(len(tasks), cpu_count())
    print(f"Using {num_workers} worker processes for {len(tasks)} tasks")

    worker = partial(
        _worker_task,
        amplitude=amplitude,
        sigma=sigma,
        scale=scale,
        output_dir=output_dir,
    )

    tk0 = (
        tqdm(range(len(tasks)), desc="Simulations", unit="sim")
        if TQDM
        else range(len(tasks))
    )

    with open(results_file, "w") as f:
        f.write(
            "x_r, y_r, living_room_signal, bedroom_signal, "
            "kitchen_signal, bathroom_signal\n"
        )

        with get_context("fork").Pool(processes=num_workers) as pool:
            results = pool.imap_unordered(worker, tasks, chunksize=4)
            completed = 0
            for tk0, result in zip(tk0, results):
                if result is not None:
                    x_r, y_r, sig_lr, sig_br, sig_kr, sig_ba = result
                    f.write(
                        f"{x_r:.2f}, {y_r:.2f}, "
                        f"{sig_lr:.2f}, {sig_br:.2f}, "
                        f"{sig_kr:.2f}, {sig_ba:.2f}\n"
                    )
                    completed += 1
                    if completed % 10 == 0:
                        print(f"Completed {completed}/{len(tasks)} simulations")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate WiFi signal strength for different" " router placements"
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
        help="Scale factor for grid resolution (default:" "100, higher is finer)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory where PNG files will be saved (default: current "
        "working directory)",
    )
    parser.add_argument(
        "--results-file",
        type=str,
        default="results.txt",
        help="Path to the results.txt file (default: results.txt in current "
        "working directory)",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_arguments()

    main(
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
    )
