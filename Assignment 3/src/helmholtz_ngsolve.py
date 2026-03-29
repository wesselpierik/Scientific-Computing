import argparse
import os
import ngsolve
from netgen.geom2d import CSG2d, Rectangle
import matplotlib.pyplot as plt
import numpy as np
import threading
import time


# ---------------- PARAMETERS ----------------
Lx, Ly = 10.0, 8.0
k = 50.3
sigma = 0.2
amplitude = 1e4

measure_locations = {
    "living_room": (1, 5),
    "kitchen": (2, 1),
    "bedroom": (9, 7),
    "bathroom": (9, 1),
}


# ---------------- GEOMETRY ----------------
def create_geometry() -> CSG2d:
    geo = CSG2d()

    # outer domain
    domain = Rectangle((0, 0), (Lx, Ly)).Maxh(5)

    # walls
    walls = [
        Rectangle((2.425, 0.15), (2.575, 2.0)).Maxh(0.05),
        Rectangle((6.925, 0.15), (7.075, 1.5)).Maxh(0.05),
        Rectangle((6.925, 2.5), (7.075, 3.0)).Maxh(0.05),
        Rectangle((0.15, 2.925), (3, 3.075)).Maxh(0.05),
        Rectangle((4, 2.925), (6, 3.075)).Maxh(0.05),
        Rectangle((7, 2.925), (10, 3.075)).Maxh(0.05),
        Rectangle((5.925, 3), (6.075, 7.925)).Maxh(0.05),
    ]

    # assign materials
    geo.Add(domain.Mat("air"))

    for i, w in enumerate(walls):
        geo.Add(w.Mat(f"wall{i}"))

    return geo


# ---------------- REFRACTIVE INDEX ----------------
def build_refractive_index(mesh: ngsolve.Mesh) -> ngsolve.CoefficientFunction:
    # default air
    n_air = 1.0
    n_wall = 2.5 + 1j

    regions = mesh.GetMaterials()

    values = []
    for mat in regions:
        if "wall" in mat:
            values.append(n_wall)
        else:
            values.append(n_air)

    return ngsolve.CoefficientFunction(values)


# ---------------- SOURCE ----------------
def gaussian_source(x_r: float, y_r: float) -> ngsolve.CoefficientFunction:
    x, y = ngsolve.specialcf.x, ngsolve.specialcf.y
    return amplitude * ngsolve.exp(-((x - x_r) ** 2 + (y - y_r) ** 2) / (2 * sigma**2))


# ---------------- SOLVER ----------------
def solve_wifi(x_r: float, y_r: float, fes: ngsolve.FESpace, a: ngsolve.BilinearForm, inv: ngsolve.Operator) -> ngsolve.GridFunction:

    v = fes.TestFunction()

    f = gaussian_source(x_r, y_r)

    rhs = ngsolve.LinearForm(fes)
    rhs += f * v * ngsolve.dx

    u_sol = ngsolve.GridFunction(fes)

    rhs.Assemble()

    # direct solver (fast for this size)
    u_sol.vec.data = inv * rhs.vec

    return u_sol


# ---------------- SIGNAL MEASUREMENT ----------------
def measure_signal(
    u_sol: ngsolve.GridFunction, mesh: ngsolve.Mesh, x: float, y: float
) -> float:
    val = u_sol(mesh(x, y))
    return 10 * np.log10(abs(val) ** 2 + 1e-20)


# ---------------- ROUTER VALIDATION ----------------
def valid_position(x_r: float, y_r: float) -> bool:
    walls = [
        (0.15, 2.425, 0.15, 2.0),
        (6.925, 7.075, 0.15, 1.5),
        (6.925, 7.075, 2.5, 3.0),
        (0.15, 3, 2.925, 3.075),
        (4, 6, 2.925, 3.075),
        (7, 10, 2.925, 3.075),
        (5.925, 6.075, 3, 7.925),
    ]

    for xmin, xmax, ymin, ymax in walls:
        if xmin <= x_r <= xmax and ymin <= y_r <= ymax:
            return False

    return True


def precompute_inverse(
    a: ngsolve.BilinearForm, fes: ngsolve.FESpace
) -> ngsolve.Operator:
    return a.mat.Inverse(fes.FreeDofs(), inverse="umfpack")


def precompute_fes(mesh: ngsolve.Mesh) -> ngsolve.FESpace:
    return ngsolve.H1(mesh, order=2, complex=True)


def precompute_a(mesh: ngsolve.Mesh, fes: ngsolve.FESpace) -> ngsolve.BilinearForm:
    u = fes.TrialFunction()
    v = fes.TestFunction()

    a = ngsolve.BilinearForm(fes)
    a += ngsolve.grad(u) * ngsolve.grad(v) * ngsolve.dx
    a += -(k**2) * build_refractive_index(mesh) ** 2 * u * v * ngsolve.dx
    a += -1j * k * u * v * ngsolve.ds
    a.Assemble()

    return a


def _mesh_worker(result: dict, geo: ngsolve.CSG2d):
    try:
        result["mesh"] = geo.GenerateMesh(quad_dominated=True)
    except Exception as exc:
        result["error"] = exc


def generate_mesh_with_progress(
    geo: ngsolve.CSG2d, maxh: float = 0.05, heartbeat_seconds: int = 5
):
    print(f"Generating mesh with maxh={maxh} ...")

    # Increase Netgen verbosity to show more meshing-stage details.
    try:
        ngsolve.ngsglobals.msg_level = max(ngsolve.ngsglobals.msg_level, 5)
        print(f"Netgen verbosity level: {ngsolve.ngsglobals.msg_level}")
    except Exception:
        print("Could not increase Netgen verbosity; " "using heartbeat logs only.")

    result = {"mesh": None, "error": None}

    start = time.perf_counter()
    worker = threading.Thread(target=_mesh_worker, args=(result, geo), daemon=True)
    worker.start()

    while worker.is_alive():
        worker.join(timeout=heartbeat_seconds)
        if worker.is_alive():
            elapsed = time.perf_counter() - start
            print(
                f"Mesh generation in progress... {elapsed:.1f}s elapsed",
                end="\r",
            )

    elapsed = time.perf_counter() - start

    if result["error"] is not None:
        raise result["error"]

    print(f"Mesh generation finished in {elapsed:.1f}s")
    return result["mesh"]


def save_solution_png(
    u_sol: ngsolve.GridFunction,
    mesh: ngsolve.Mesh,
    x_r: float,
    y_r: float,
    output_dir: str,
    resolution: int = 200,
) -> None:
    x_values = np.linspace(0.0, Lx, resolution)
    y_values = np.linspace(0.0, Ly, resolution)
    field_db = np.full((resolution, resolution), np.nan, dtype=np.float64)

    for i, y in enumerate(y_values):
        for j, x in enumerate(x_values):
            try:
                val = u_sol(mesh(float(x), float(y)))
                field_db[i, j] = 10 * np.log10(abs(val) ** 2 + 1e-20)
            except Exception:
                continue

    finite_vals = field_db[np.isfinite(field_db)]
    vmin = np.nanmin(finite_vals) if finite_vals.size else -120.0
    vmax = np.nanmax(finite_vals) if finite_vals.size else -40.0

    plt.figure(figsize=(10, 8))
    plt.imshow(
        field_db,
        extent=[0.0, Lx, 0.0, Ly],
        origin="lower",
        cmap="jet",
        vmin=vmin,
        vmax=vmax,
    )
    plt.colorbar(label="Signal Strength (dB)")
    plt.scatter(x_r, y_r, color="cyan", marker="x", s=100, label="Router")
    plt.title(f"WiFi Signal Strength (Router at ({x_r:.2f}, {y_r:.2f}))")
    plt.xlabel("x (meters)")
    plt.ylabel("y (meters)")
    plt.legend()
    plt.tight_layout()
    output_path = os.path.join(output_dir, f"signal_{x_r:.2f}_{y_r:.2f}.png")
    plt.savefig(output_path)
    plt.close()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate WiFi signal strength with NGSolve"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help=(
            "Directory where PNG files will be saved " "(default: current directory)"
        ),
    )
    parser.add_argument(
        "--results-file",
        type=str,
        default="results_ngsolve.txt",
        help="Path to the results file (default: results_ngsolve.txt)",
    )
    parser.add_argument(
        "--mesh-file",
        type=str,
        default="room.vol",
        help="Path to cached mesh file (default: room.vol)",
    )

    return parser.parse_args()


# ---------------- MAIN LOOP ----------------
def main(args: argparse.Namespace) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    results_parent = os.path.dirname(os.path.abspath(args.results_file))
    os.makedirs(results_parent, exist_ok=True)

    try:
        mesh = ngsolve.Mesh(args.mesh_file)
    except Exception as e:
        print(e)
        print("Mesh file not found, generating new mesh...")
        geo = create_geometry()
        ngmesh = generate_mesh_with_progress(geo, maxh=5)
        mesh = ngsolve.Mesh(ngmesh)  # MUCH coarser than FD
        mesh.Save(args.mesh_file)

    print("Mesh is ready")

    with open(args.results_file, "w") as f:
        f.write("x_r, y_r, living, bedroom, kitchen, bathroom\n")

        fes = precompute_fes(mesh)
        a = precompute_a(mesh, fes)
        inv = precompute_inverse(a, fes)
        print(f"System has {a.mat.Size()} DOFs")

        for x_r in np.linspace(2.5, 9.7, 40):  # fewer points needed
            for y_r in np.linspace(5.5, 7.7, 30):
                if not valid_position(x_r, y_r):
                    continue

                print(f"Router at ({x_r:.2f}, {y_r:.2f})")

                u_sol = solve_wifi(mesh, x_r, y_r, fes, a, inv)

                signals = {}
                for name, (x, y) in measure_locations.items():
                    signals[name] = measure_signal(u_sol, mesh, x, y)

                print(signals)

                f.write(
                    f"{x_r:.2f}, {y_r:.2f}, "
                    f"{signals['living_room']:.2f}, "
                    f"{signals['bedroom']:.2f}, "
                    f"{signals['kitchen']:.2f}, "
                    f"{signals['bathroom']:.2f}\n"
                )

                save_solution_png(u_sol, mesh, x_r, y_r, args.output_dir)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
