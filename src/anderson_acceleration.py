from __future__ import print_function
from types import MethodType
from collections import deque

import torch
from torch.nn.utils import parameters_to_vector


def vector_to_parameters(vec, parameters):
    """Drop-in replacement for ``torch.nn.utils.vector_to_parameters`` that
    uses ``param.data.copy_()`` instead of ``param.data =`` so that the memory
    format of existing parameter tensors is preserved (fix ported from AADL)."""
    pointer = 0
    for param in parameters:
        num_param = param.numel()
        param.data.copy_(vec[pointer:pointer + num_param].view_as(param).data)
        pointer += num_param


def anderson_qr_fun(X, R, relaxation=1.0, regularization=0.0):
    # Anderson Acceleration type 2 (data-driven: PDE residuals drive the solve)
    # X: parameter history matrix  [n_params, history]
    # R: PDE residual history matrix [n_res,   history]
    # Returns the accelerated parameter vector.

    assert X.ndim == 2, "X must be a matrix"
    assert R.ndim == 2, "R must be a matrix"
    assert regularization >= 0.0, "regularization for least-squares must be >=0.0"

    DX = X[:, 1:] - X[:, :-1]   # DX[:,i] = X[:,i+1] - X[:,i]
    DR = R[:, 1:] - R[:, :-1]   # DR[:,i] = R[:,i+1] - R[:,i]
    b  = R[:, -1]                # target: last residual

    # --- Column equilibration (ported from AADL) ----------------------------
    # Scale each column of DR to unit L2 norm before the solve, then undo
    # after.  This improves the condition number of the least-squares problem
    # at negligible cost and is critical for deep history buffers.
    scale = DR.norm(dim=0)
    safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    A_s = DR / safe_scale

    # --- Build system (optionally augmented for Tikhonov regularization) ----
    if regularization == 0.0:
        A   = A_s
        rhs = b
    else:
        sqrt_reg = torch.sqrt(torch.tensor(regularization, device=A_s.device, dtype=A_s.dtype))
        eye      = torch.eye(A_s.size(1), device=A_s.device, dtype=A_s.dtype)
        zero_pad = torch.zeros(A_s.size(1), device=A_s.device, dtype=A_s.dtype)
        A   = torch.cat((A_s, sqrt_reg * eye), dim=0)
        rhs = torch.cat((b,   zero_pad))

    # --- QR + triangular solve (ported from AADL) ---------------------------
    # Faster and more numerically stable than lstsq for tall-skinny A.
    # Also fixes a shape inconsistency: the old regularised path returned
    # gamma with shape [k, 1] while the unconstrained path returned [k].
    Q, R_qr = torch.linalg.qr(A, mode='reduced')
    y = torch.linalg.solve_triangular(
        R_qr, (Q.mT @ rhs).unsqueeze(-1), upper=True
    ).squeeze(-1)

    # Undo column scaling to recover the true mixing vector
    gamma = y / safe_scale

    # --- Extrapolation ------------------------------------------------------
    extr = X[:, -1] - torch.matmul(DX, gamma)

    if relaxation != 1:
        assert relaxation > 0, "relaxation must be positive"
        extr = (1 - relaxation) * X[:, -1] + relaxation * extr

    return extr


def accelerate(
    optimizer,
    relaxation: float = 1.0,
    regularization: float = 0.0,
    history_depth: int = 15,
    store_each_nth: int = 10,
    frequency: int = 10,
):
    # acceleration options
    optimizer.acc = True
    optimizer.acc_relaxation = relaxation
    optimizer.acc_regularization = regularization

    optimizer.acc_history_depth = history_depth  # history size
    optimizer.acc_store_each_nth = store_each_nth  # frequency to update history
    optimizer.acc_frequency = frequency  # frequency to accelerate

    # TODO: add averaging or other methods

    # acceleration history
    optimizer.acc_call_counter = 0
    optimizer.acc_store_counter = 0
    optimizer.acc_param_hist = [
        deque([], maxlen=optimizer.acc_history_depth) for _ in optimizer.param_groups
    ]
    optimizer.res_hist = deque([], maxlen=optimizer.acc_history_depth)

    # redefine step of the optimizer
    optimizer.orig_step = optimizer.step

    optimizer.step = MethodType(accelerated_step, optimizer)

    return optimizer


def remove_acceleration(optimizer):
    if not getattr(optimizer, "acc", False):
        return optimizer
    optimizer.acc = False
    optimizer.step = optimizer.orig_step
    # clean up all dynamically attached attributes
    for attr in [
        "orig_step", "acc_relaxation", "acc_regularization",
        "acc_history_depth", "acc_store_each_nth", "acc_frequency",
        "acc_call_counter", "acc_store_counter", "acc_param_hist", "res_hist",
    ]:
        optimizer.__dict__.pop(attr, None)
    return optimizer


def clear_hist(optimizer):
    # clear history when resampling (e.g. after re-drawing collocation points)
    optimizer.acc_param_hist = [
        deque([], maxlen=optimizer.acc_history_depth) for _ in optimizer.param_groups
    ]
    optimizer.res_hist = deque([], maxlen=optimizer.acc_history_depth)


def accelerated_step(self, closure):
    if closure is None:
        raise RuntimeError(
            "DD-AADL requires a closure that returns (residual, loss). "
            "Call accelerated_step(closure) with a valid closure."
        )

    self.orig_step(closure)

    self.acc_store_counter += 1
    self.acc_call_counter += 1

    should_store = (self.acc_store_counter % self.acc_store_each_nth == 0)
    should_accelerate = (self.acc_call_counter % self.acc_frequency == 0)

    # avoid an extra closure call on steps where neither storage nor acceleration fires
    if not should_store and not should_accelerate:
        return

    res, loss = closure()  # calculate the residual once, only when needed

    # add current parameters to the history
    if should_store:
        for group, group_hist in zip(self.param_groups, self.acc_param_hist):
            group_hist.append(
                parameters_to_vector(group["params"]).detach()
            )  # network parameters

        self.res_hist.append(res.detach())  # residual from current network parameters

    # perform acceleration
    if should_accelerate:
        for group, group_hist in zip(self.param_groups, self.acc_param_hist):
            if len(group_hist) >= 3:
                # build history matrices and solve for the accelerated candidate
                # (pure tensor arithmetic — no gradient tracking needed)
                with torch.no_grad():
                    X = torch.stack(list(group_hist), dim=1)
                    R = torch.stack(list(self.res_hist), dim=1)
                    acc_param = anderson_qr_fun(
                        X, R, self.acc_relaxation, self.acc_regularization
                    )
                    vector_to_parameters(acc_param, group["params"])

                # evaluate candidate (closure must run with gradient tracking
                # because it calls loss.backward() internally)
                _, new_loss = closure()

                with torch.no_grad():
                    if new_loss < loss:
                        group_hist.pop()
                        group_hist.append(acc_param)
                    else:
                        # revert to non-accelerated params
                        vector_to_parameters(group_hist[-1], group["params"])

        final_res, final_loss = closure()
        if final_loss < loss:
            with torch.no_grad():
                self.res_hist.pop()
                self.res_hist.append(final_res.detach())