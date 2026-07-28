"""
Dimension sweep runner for DD-AADL experiments.

Usage
-----
Activate the project venv, then run from the repo root:

    python run_dim_sweep.py --experiments burgers_sinusoidal heat_cosine --dims 2 10 50 100

Or sweep all registered experiments:

    python run_dim_sweep.py --all --dims 2 10 50 100

Each (experiment, d) combination is run as a subprocess so that a crash in
one combination does not abort the rest.  Results land in
  results/<experiment>_d<D>_adam.npy
  results/<experiment>_d<D>_aadl.npy
  results/<experiment>_d<D>_ddaadl.npy
  results/<experiment>_d<D>_metrics.npz

and a summary CSV is written to results/sweep_summary.csv.

Design note
-----------
Each experiment script controls its own training loop, network size, and
hyperparameters.  The sweep runner simply passes the target dimension ``d``
via the environment variable ``DDAADL_DIM`` which each script reads at
startup.  To add a new experiment to the sweep, register it in
``EXPERIMENTS`` below and make the script honour ``DDAADL_DIM``.

Adapting an experiment script to honour DDAADL_DIM
---------------------------------------------------
Add the following near the parameter section (after the imports):

    import os
    _d_env = os.environ.get("DDAADL_DIM")
    if _d_env is not None:
        d = int(_d_env)
        layers = np.array([d, 50, 50, 50, 1])

The sweep runner sets the environment variable; if not set, the script uses
its built-in default (typically d=100).
"""

import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Registry: short name -> path relative to repo root
# ---------------------------------------------------------------------------
EXPERIMENTS = {
    # Burgers
    "burgers_parabola":        "experiments/burgers/test_HighD_Burgers_parabola.py",
    "burgers_boundary_layer":  "experiments/burgers/test_HighD_Burgers_boundary_layer.py",
    "burgers_sinusoidal":      "experiments/burgers/test_HighD_Burgers_sinusoidal.py",
    "burgers_gaussian":        "experiments/burgers/test_HighD_Burgers_gaussian.py",
    # Helmholtz
    "helmholtz_parabola":      "experiments/helmholtz/test_HighD_Helmholtz_parabola.py",
    "helmholtz_goniometric":   "experiments/helmholtz/test_HighD_Helmholtz_goniometric.py",
    "helmholtz_boundary_layer":"experiments/helmholtz/test_HighD_Helmholtz_boundary_layer.py",
    "helmholtz_cosine":        "experiments/helmholtz/test_HighD_Helmholtz_cosine.py",
    "helmholtz_gaussian":      "experiments/helmholtz/test_HighD_Helmholtz_gaussian.py",
    "helmholtz_cosine_product":"experiments/helmholtz/test_HighD_Helmholtz_cosine_product.py",
    "helmholtz_interior_spike":"experiments/helmholtz/test_HighD_Helmholtz_interior_spike.py",
    # Black-Scholes
    "bs_parabola":             "experiments/black_scholes/test_HighD_BlackScholes_parabola.py",
    "bs_gaussian":             "experiments/black_scholes/test_HighD_BlackScholes_gaussian.py",
    "bs_boundary_layer":       "experiments/black_scholes/test_HighD_BlackScholes_boundary_layer.py",
    # New PDE types
    "heat_cosine":             "experiments/heat/test_HighD_Heat_cosine.py",
    "poisson_sinusoidal":      "experiments/poisson/test_HighD_Poisson_sinusoidal.py",
    "allen_cahn_sinusoidal":   "experiments/allen_cahn/test_HighD_AllenCahn_sinusoidal.py",
    "advdiff_polynomial":      "experiments/advection_diffusion/test_HighD_AdvDiff_polynomial.py",
    "wave_cosine":             "experiments/wave/test_HighD_Wave_cosine.py",
}

DEFAULT_DIMS = [2, 10, 50, 100]


def run_one(script_path: str, d: int, results_dir: Path, python: str) -> dict:
    """Run a single (script, d) combination and return a result dict."""
    stem = Path(script_path).stem + f"_d{d}"
    env = os.environ.copy()
    env["DDAADL_DIM"] = str(d)
    # Ask the script to save outputs into results_dir
    env["DDAADL_OUTDIR"] = str(results_dir)

    print(f"\n{'='*60}")
    print(f"  Running: {Path(script_path).name}  d={d}")
    print(f"{'='*60}")

    t0 = time.time()
    result = subprocess.run(
        [python, script_path],
        env=env,
        cwd=str(Path(script_path).parent),   # run from the experiment's own dir
        capture_output=False,                 # stream output to terminal
        text=True,
    )
    elapsed = time.time() - t0

    success = result.returncode == 0
    print(f"  → {'OK' if success else 'FAILED (exit ' + str(result.returncode) + ')'}  "
          f"in {elapsed:.1f}s")

    return {
        "experiment": Path(script_path).stem,
        "d": d,
        "success": success,
        "returncode": result.returncode,
        "elapsed_s": round(elapsed, 1),
    }


def build_summary(rows: list, results_dir: Path):
    """Write sweep_summary.csv to results_dir."""
    import numpy as np

    summary_path = results_dir / "sweep_summary.csv"
    fieldnames = [
        "experiment", "d", "success", "returncode", "elapsed_s",
        "iters_adam_mean", "iters_aadl_mean", "iters_ddaadl_mean", "threshold",
    ]

    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            stem = row["experiment"] + f"_d{row['d']}"
            metrics_path = results_dir / f"{stem}_metrics.npz"
            if metrics_path.exists():
                try:
                    m = np.load(metrics_path)
                    row["iters_adam_mean"]    = float(m["iters_adam"].mean())
                    row["iters_aadl_mean"]    = float(m["iters_aadl"].mean())
                    row["iters_ddaadl_mean"]  = float(m["iters_ddaadl"].mean())
                    row["threshold"]          = float(m["threshold"][0])
                except Exception:
                    pass
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    print(f"\nSweep summary written to {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run DD-AADL experiments across multiple dimensions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--experiments", "-e", nargs="+", metavar="NAME",
        help="Space-separated list of experiment short names (see EXPERIMENTS dict).",
    )
    parser.add_argument(
        "--all", "-a", action="store_true",
        help="Run all registered experiments.",
    )
    parser.add_argument(
        "--dims", "-d", nargs="+", type=int, default=DEFAULT_DIMS, metavar="D",
        help=f"Space-separated list of dimensions (default: {DEFAULT_DIMS}).",
    )
    parser.add_argument(
        "--results-dir", "-o", default="results", metavar="DIR",
        help="Directory for .npy/.npz output files (default: results/).",
    )
    parser.add_argument(
        "--python", default=sys.executable, metavar="PYTHON",
        help="Python interpreter to use (default: current interpreter).",
    )
    args = parser.parse_args()

    # --- resolve experiments ---
    if args.all:
        selected = list(EXPERIMENTS.items())
    elif args.experiments:
        unknown = [e for e in args.experiments if e not in EXPERIMENTS]
        if unknown:
            parser.error(f"Unknown experiment(s): {unknown}\nKnown: {list(EXPERIMENTS)}")
        selected = [(name, EXPERIMENTS[name]) for name in args.experiments]
    else:
        parser.error("Specify --experiments NAME [NAME ...] or --all")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Sweeping {len(selected)} experiment(s) × {len(args.dims)} dimension(s) "
          f"= {len(selected) * len(args.dims)} runs")
    print(f"Dimensions : {args.dims}")
    print(f"Experiments: {[n for n, _ in selected]}")
    print(f"Results dir: {results_dir.resolve()}")

    rows = []
    total = len(selected) * len(args.dims)
    done = 0
    for name, script in selected:
        for d in args.dims:
            done += 1
            print(f"\n[{done}/{total}]", end="")
            row = run_one(script, d, results_dir, args.python)
            rows.append(row)

    build_summary(rows, results_dir)

    n_ok = sum(r["success"] for r in rows)
    n_fail = total - n_ok
    print(f"\nDone: {n_ok}/{total} OK, {n_fail} failed.")
    if n_fail:
        print("Failed runs:")
        for r in rows:
            if not r["success"]:
                print(f"  {r['experiment']}  d={r['d']}  exit={r['returncode']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
