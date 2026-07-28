#!/usr/bin/env python3
"""
Smoke test for all DD-AADL experiment scripts.

Each script is exec'd in-process with:
  niters = 5, num_repeats = 1, d = 10, N_f = 100, N_u = 50

This is enough to exercise imports, tensor shapes, autograd graph, and the
DD-AADL closure API without waiting hours for full training runs.

Usage
-----
    # From the repo root with the project venv active:
    python3 smoke_test.py                          # run all
    python3 smoke_test.py -s burgers_sinusoidal    # run one
    python3 smoke_test.py -s helmholtz_cosine wave_cosine
"""

import argparse
import os
import sys
import traceback
from pathlib import Path

# Force non-interactive matplotlib backend before any script imports it
os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parent

SCRIPTS = {
    "burgers_parabola":         "experiments/burgers/test_HighD_Burgers_parabola.py",
    "burgers_boundary_layer":   "experiments/burgers/test_HighD_Burgers_boundary_layer.py",
    "burgers_sinusoidal":       "experiments/burgers/test_HighD_Burgers_sinusoidal.py",
    "burgers_gaussian":         "experiments/burgers/test_HighD_Burgers_gaussian.py",
    "helmholtz_parabola":       "experiments/helmholtz/test_HighD_Helmholtz_parabola.py",
    "helmholtz_goniometric":    "experiments/helmholtz/test_HighD_Helmholtz_goniometric.py",
    "helmholtz_boundary_layer": "experiments/helmholtz/test_HighD_Helmholtz_boundary_layer.py",
    "helmholtz_cosine":         "experiments/helmholtz/test_HighD_Helmholtz_cosine.py",
    "helmholtz_gaussian":       "experiments/helmholtz/test_HighD_Helmholtz_gaussian.py",
    "helmholtz_cosine_product": "experiments/helmholtz/test_HighD_Helmholtz_cosine_product.py",
    "helmholtz_interior_spike": "experiments/helmholtz/test_HighD_Helmholtz_interior_spike.py",
    "bs_parabola":              "experiments/black_scholes/test_HighD_BlackScholes_parabola.py",
    "bs_gaussian":              "experiments/black_scholes/test_HighD_BlackScholes_gaussian.py",
    "bs_boundary_layer":        "experiments/black_scholes/test_HighD_BlackScholes_boundary_layer.py",
    "heat_cosine":              "experiments/heat/test_HighD_Heat_cosine.py",
    "poisson_sinusoidal":       "experiments/poisson/test_HighD_Poisson_sinusoidal.py",
    "allen_cahn_sinusoidal":    "experiments/allen_cahn/test_HighD_AllenCahn_sinusoidal.py",
    "advdiff_polynomial":       "experiments/advection_diffusion/test_HighD_AdvDiff_polynomial.py",
    "wave_cosine":              "experiments/wave/test_HighD_Wave_cosine.py",
    "burgers_2d_data":          "experiments/burgers/test_2D_burgers_with_data.py",
}

# Applied as literal string replacements before exec.
# Order matters: later patches may refine earlier ones.
PATCHES = [
    # Shrink training budget
    ("niters = 3000",    "niters = 5"),
    ("niters = 5000",    "niters = 5"),
    # Single repeat is enough to check shapes
    ("num_repeats = 5",  "num_repeats = 1"),
    # Smaller problem — d=10 still exercises all dimensions but is much faster
    ("d = 100",          "d = 10"),
    # Fewer collocation / boundary points
    ("N_f = 4000",       "N_f = 100"),
    ("N_u = 400",        "N_u = 50"),
    # Disable mid-run resampling (would never trigger in 5 iters anyway)
    ("resample = 500",   "resample = 9999"),
    # Wave equation: fewer IC-velocity sample points
    ("n_ic = 200",       "n_ic = 10"),
    # Suppress file output (no .npy/.npz/.jpg during smoke test)
    ("save_records(__file__,", "# save_records(__file__,"),
    ("fig.savefig(",     "# fig.savefig("),
    ("fig2.savefig(",    "# fig2.savefig("),
]

# Prepended to every script to ensure Agg backend is active
_HEADER = "import matplotlib\nmatplotlib.use('Agg')\n"


def run_smoke(name: str, rel_path: str) -> tuple:
    """Exec the script with smoke-test patches.  Returns (passed, error_text)."""
    path = ROOT / rel_path
    if not path.exists():
        return False, f"File not found: {path}"

    with open(path) as f:
        code = f.read()

    for old, new in PATCHES:
        code = code.replace(old, new)

    code = _HEADER + code

    script_dir = str(path.parent)
    old_cwd = os.getcwd()
    os.chdir(script_dir)
    try:
        ns = {"__file__": str(path), "__name__": "__main__", "__spec__": None}
        exec(compile(code, str(path), "exec"), ns)
        return True, ""
    except Exception:
        return False, traceback.format_exc()
    finally:
        os.chdir(old_cwd)


def main():
    parser = argparse.ArgumentParser(
        description="Smoke-test all DD-AADL experiment scripts (5 iters, d=10).",
    )
    parser.add_argument(
        "--scripts", "-s", nargs="+", metavar="NAME",
        help="Short names of scripts to run (default: all).  "
             f"Available: {list(SCRIPTS)}",
    )
    args = parser.parse_args()

    to_run = args.scripts if args.scripts else list(SCRIPTS.keys())
    unknown = [s for s in to_run if s not in SCRIPTS]
    if unknown:
        parser.error(f"Unknown script(s): {unknown}\nAvailable: {list(SCRIPTS)}")

    results = {}
    for i, name in enumerate(to_run, 1):
        print(f"\n[{i}/{len(to_run)}] {'─'*55}")
        print(f"  {name}")
        print(f"  {SCRIPTS[name]}")
        print(f"  {'─'*53}")
        ok, err = run_smoke(name, SCRIPTS[name])
        results[name] = ok
        if ok:
            print(f"  PASS")
        else:
            print(f"  FAIL")
            # Print only the last 20 lines of the traceback to keep output tidy
            lines = err.strip().splitlines()
            for line in lines[-20:]:
                print(f"    {line}")

    print(f"\n{'='*60}")
    passed = sum(results.values())
    total  = len(results)
    print(f"  {passed}/{total} passed")
    failed = [n for n, ok in results.items() if not ok]
    if failed:
        print(f"  Failed: {failed}")
    else:
        print("  All smoke tests passed.")
    print(f"{'='*60}")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
