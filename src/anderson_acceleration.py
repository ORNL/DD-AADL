from __future__ import print_function
from itertools import count
import sys
from types import MethodType
from collections import deque


import torch
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector, vector_to_parameters


def anderson_qr_fun(X, R, relaxation=1.0, regularization=0.0):
    # Solve the least square problem with qr factorization
    # Anderson Acceleration type 2
    # Tinput both the parameters X and residual R
    # Return acceleration result

    assert X.ndim == 2, "X must be a matrix"
    assert R.ndim == 2, "R must be a matrix"
    assert regularization >= 0.0, "regularization for least-squares must be >=0.0"

    # Compute residuals
    DX = X[:, 1:] - X[:, :-1]  # DX[:,i] =  X[:,i+1] -  X[:,i]
    DR = R[:, 1:] - R[:, :-1]  # DR[:,i] =  R[:,i+1] -  R[:,i]

    if regularization == 0.0:
        # solve unconstrained least-squares problem
        gamma, _ = torch.lstsq(R[:, -1].unsqueeze(1), DR)
        gamma = gamma.squeeze(1)[: DR.size(1)]
    else:
        # solve augmented least-squares for Tykhonov regularization
        rhs = R[:, -1].unsqueeze(1)
        expanded_rhs = torch.cat((rhs, torch.zeros(DR.size(1), 1)))
        expanded_matrix = torch.cat(
            (DR, torch.sqrt(torch.tensor(regularization)) * torch.eye(DR.size(1)))
        )  # sqrt here?
        gamma, _ = torch.lstsq(expanded_rhs, expanded_matrix)
        gamma = gamma.squeeze(1)[: DR.size(1)]

    # compute acceleration
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


# TODO: add acceeleration removal
def remove_acceleration(optimizer):
    optimizer.acc = False
    optimizer.step = optimizer.orig_step
    return optimizer


def clear_hist(optimizer):
    # clear history when resampling
    print("clear history at ", optimizer.acc_store_counter)
    optimizer.acc_param_hist = [
        deque([], maxlen=optimizer.acc_history_depth) for _ in optimizer.param_groups
    ]
    optimizer.res_hist = deque([], maxlen=optimizer.acc_history_depth)


def accelerated_step(self, closure):
    # check for bugs
    if closure is None:
        print("Unable to perform acceleration without closure")
        sys.exit()

    self.orig_step(closure)

    res, loss = closure()  # calculate the residual
    # add current parameters to the history
    self.acc_store_counter += 1
    if self.acc_store_counter % self.acc_store_each_nth == 0:
        for group, group_hist in zip(self.param_groups, self.acc_param_hist):
            group_hist.append(
                parameters_to_vector(group["params"]).detach()
            )  # network parameters

        self.res_hist.append(res.detach())  # residual from current network parameters

    # perform acceleration
    self.acc_call_counter += 1
    if self.acc_call_counter % self.acc_frequency == 0:
        for group, group_hist in zip(self.param_groups, self.acc_param_hist):
            if len(group_hist) >= 3:
                # make matrix of updates from the history list
                X = torch.stack(list(group_hist), dim=1)
                R = torch.stack(list(self.res_hist), dim=1)
                acc_param = anderson_qr_fun(
                    X, R, self.acc_relaxation, self.acc_regularization
                )

                # check performance
                vector_to_parameters(acc_param, group["params"])
                _, new_loss = closure()
                if new_loss < loss:
                    group_hist.pop()
                    group_hist.append(acc_param)

                else:
                    # revert to non-accelerated params
                    vector_to_parameters(group_hist[-1], group["params"])

        final_res, final_loss = closure()
        if final_loss < loss:
            print(
                "acceleration working at step %5d, improve loss by %.5f"
                % (self.acc_call_counter, loss - final_loss)
            )

            self.res_hist.pop()
            self.res_hist.append(final_res.detach())