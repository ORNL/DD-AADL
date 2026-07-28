"""
Shared utilities for DD-AADL experiment scripts.

Provides:
  val_metrics(net, x_val, y_val)  -> (L1 error, relative L2 error)
  save_records(script, rec_adam, rec_aadl, rec_ddaadl, threshold=1e-4)
      -> saves *.npy training curves + *_metrics.npz threshold summary
"""

import os
import numpy as np
import torch


def val_metrics(net, x_val, y_val):
    """Return (L1 error, relative L2 error) on a held-out validation set.

    Both quantities are computed under torch.no_grad() and returned as plain
    Python floats, so they are safe to accumulate across repeats.
    """
    with torch.no_grad():
        y_pred = net(x_val)
    err_abs = torch.mean(torch.abs(y_val - y_pred)).item()
    denom = torch.sqrt(torch.mean(y_val ** 2)).clamp(min=1e-12)
    err_rel = (torch.sqrt(torch.mean((y_val - y_pred) ** 2)) / denom).item()
    return err_abs, err_rel


def _first_crossing(rec, threshold):
    """For each repeat column, return the first iteration index where loss
    drops below *threshold*.  Returns *niters* (last index) if never crossed.

    Parameters
    ----------
    rec : np.ndarray of shape [niters+1, num_repeats]
    threshold : float
    """
    crossed = rec < threshold                     # bool [niters+1, num_repeats]
    first = np.argmax(crossed, axis=0)            # index of first True per col
    never = ~np.any(crossed, axis=0)              # columns that never crossed
    first[never] = rec.shape[0] - 1              # sentinel: last iteration
    return first                                  # [num_repeats]


def save_records(script, rec_adam, rec_aadl, rec_ddaadl, threshold=1e-4):
    """Save training records and print an iterations-to-threshold summary.

    Files written (relative to the current working directory):
      <stem>_adam.npy      — loss curve, shape [niters+1, num_repeats]
      <stem>_aadl.npy      — same for AADL
      <stem>_ddaadl.npy    — same for DD-AADL
      <stem>_metrics.npz   — threshold, iters_adam, iters_aadl, iters_ddaadl

    Parameters
    ----------
    script : str   pass __file__ from the calling script
    rec_adam, rec_aadl, rec_ddaadl : np.ndarray  [niters+1, num_repeats]
    threshold : float  loss threshold for the crossing metric (default 1e-4)
    """
    stem = os.path.splitext(os.path.basename(script))[0]

    np.save(f"{stem}_adam.npy",    rec_adam)
    np.save(f"{stem}_aadl.npy",    rec_aadl)
    np.save(f"{stem}_ddaadl.npy",  rec_ddaadl)

    itt_adam    = _first_crossing(rec_adam,    threshold)
    itt_aadl    = _first_crossing(rec_aadl,    threshold)
    itt_ddaadl  = _first_crossing(rec_ddaadl,  threshold)

    np.savez(
        f"{stem}_metrics.npz",
        threshold=np.array([threshold]),
        iters_adam=itt_adam,
        iters_aadl=itt_aadl,
        iters_ddaadl=itt_ddaadl,
    )

    print(f"\n--- {stem} — saved *.npy / _metrics.npz ---")
    print(f"Iterations to threshold {threshold:.0e}:")
    print(f"  Adam:    mean={itt_adam.mean():.0f}  per-repeat={itt_adam}")
    print(f"  AADL:    mean={itt_aadl.mean():.0f}  per-repeat={itt_aadl}")
    print(f"  DD-AADL: mean={itt_ddaadl.mean():.0f}  per-repeat={itt_ddaadl}")
