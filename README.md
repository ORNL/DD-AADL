# DD-AADL — Data-Driven Anderson Acceleration for Deep Learning

A PyTorch implementation of **Type-2 (data-driven) Anderson Acceleration** applied to
Physics-Informed Neural Networks (PINNs).  Unlike standard (Type-1) Anderson acceleration,
which mixes iterates based on parameter differences, DD-AADL computes mixing coefficients
directly from the **PDE residuals**, exploiting the structure of the physics loss.

---

## Requirements

| Dependency | Tested version |
|---|---|
| Python | 3.10 – 3.14 |
| PyTorch | ≥ 2.0 |
| NumPy | ≥ 1.24 |
| Matplotlib | ≥ 3.7 |
| AADL (baseline) | git HEAD |

---

## Installation

### 1 — Clone the repository

```bash
git clone https://github.com/ORNL/DD-AADL.git
cd DD-AADL
```

### 2 — Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows
```

### 3 — Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **GPU (CUDA) users** — replace the PyTorch line in `requirements.txt` with the
> appropriate wheel from [pytorch.org/get-started](https://pytorch.org/get-started/locally/),
> e.g. for CUDA 12.4:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu124
> ```
> then install the remaining requirements:
> ```bash
> pip install numpy matplotlib
> pip install AADL @ git+https://github.com/allaffa/AADL.git
> ```

### 4 — Verify the installation

```bash
python3 -c "
import torch, numpy, matplotlib, AADL
from src import MLP, accelerate
print('torch', torch.__version__)
print('cuda available:', torch.cuda.is_available())
print('All imports OK.')
"
```

---

## Repository structure

```
DD-AADL/
├── src/
│   ├── anderson_acceleration.py   # DD-AADL core optimizer wrapper
│   ├── NN_models.py               # MLP and ResMLP architectures
│   ├── experiment_utils.py        # val_metrics(), save_records()
│   └── utils.py
├── experiments/
│   ├── burgers/                   # Viscous Burgers equation
│   ├── helmholtz/                 # Nonlinear Helmholtz equation
│   ├── black_scholes/             # Black-Scholes (financial PDE)
│   ├── heat/                      # Heat equation (parabolic)
│   ├── poisson/                   # Poisson equation (elliptic)
│   ├── allen_cahn/                # Allen-Cahn phase-field equation
│   ├── advection_diffusion/       # Advection-diffusion equation
│   └── wave/                      # Wave equation (hyperbolic)
├── Data/                          # Reference data for 2D Burgers
├── run_dim_sweep.py               # CLI runner for dimension-scaling studies
└── requirements.txt
```

---

## Running an experiment

Each script in `experiments/` is self-contained.  Run from the repo root:

```bash
cd experiments/burgers
python3 test_HighD_Burgers_sinusoidal.py
```

The script trains three models (Adam baseline, Adam + AADL, Adam + DD-AADL),
prints validation metrics, saves training curves as `.npy` files, and writes a
`*_metrics.npz` file with iterations-to-threshold statistics.

### Dimension-scaling study

Use `run_dim_sweep.py` to sweep all experiments over multiple problem dimensions:

```bash
# Sweep selected experiments at d = 2, 10, 50, 100
python3 run_dim_sweep.py --experiments burgers_sinusoidal heat_cosine wave_cosine \
                         --dims 2 10 50 100

# Sweep all 19 registered experiments
python3 run_dim_sweep.py --all --dims 2 10 50 100

# See all options
python3 run_dim_sweep.py --help
```

Results land in `results/` alongside a `sweep_summary.csv` with
mean iterations-to-threshold for each method × dimension combination.

To make an experiment respond to the sweep runner, add near its parameter block:

```python
import os
_d_env = os.environ.get("DDAADL_DIM")
if _d_env is not None:
    d = int(_d_env)
    layers = np.array([d, 50, 50, 50, 1])
```

---

## Method overview

DD-AADL wraps any PyTorch optimizer and intercepts the `step()` call.
Every `frequency` iterations it collects recent PDE residual vectors into a
matrix **R** and solves the constrained least-squares problem

$$\min_{\gamma} \|\mathbf{R}\gamma\|^2 \quad \text{s.t.} \quad \mathbf{1}^T\gamma = 1$$

using QR decomposition with column equilibration for numerical stability.
The resulting coefficients mix the recent parameter iterates, producing an
accelerated update that accounts for the physics of the problem.

```python
from src import MLP, accelerate, clear_hist

net   = MLP([d, 50, 50, 50, 1])
optim = torch.optim.Adam(net.parameters(), lr=1e-2)
accelerate(optim, history_depth=10, frequency=5)

for itr in range(niters):
    def closure():
        optim.zero_grad()
        res, loss = loss_fn(...)   # must return (residual_vector, scalar_loss)
        loss.backward()
        return res, loss
    optim.step(closure)
```

The closure **must** return `(res, loss)` — the flat residual tensor is what
DD-AADL uses to compute the mixing coefficients.

---

## Citation

If you use this code, please cite the original DD-AADL paper (reference TBD).
