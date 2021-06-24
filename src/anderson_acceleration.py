from __future__ import print_function
from itertools import count
import sys
from types import MethodType



import torch
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector, vector_to_parameters


def anderson_qr_fun(X, R, relaxation=1.0, regularization = 0.0):
    # Solve the least square problem with qr factorization
    # Anderson Acceleration type 2
    # Tinput both the parameters X and residual R
    # Return acceleration result

    assert X.ndim==2, "X must be a matrix"
    assert R.ndim==2, "R must be a matrix"
    assert regularization >= 0.0, "regularization for least-squares must be >=0.0"

    # Compute residuals
    DX =  X[:,1:] -  X[:,:-1] # DX[:,i] =  X[:,i+1] -  X[:,i]
    DR =  R[:,1:] -  R[:,:-1] # DR[:,i] =  R[:,i+1] -  R[:,i]

    if regularization == 0.0:
       # solve unconstrained least-squares problem
       gamma, _ = torch.lstsq( R[:,-1].unsqueeze(1), DR )
       gamma = gamma.squeeze(1)[:DR.size(1)]
    else:
       # solve augmented least-squares for Tykhonov regularization
       rhs = R[:,-1].unsqueeze(1)
       expanded_rhs       = torch.cat( (rhs, torch.zeros(DR.size(1),1)) )
       expanded_matrix = torch.cat( (DR, torch.sqrt(torch.tensor(regularization)) * torch.eye(DR.size(1))) )  # sqrt here?
       gamma, _ = torch.lstsq( expanded_rhs, expanded_matrix )
       gamma = gamma.squeeze(1)[:DR.size(1)]

    # compute acceleration
    extr = X[:,-1] - torch.matmul(DX, gamma)

    if relaxation!=1:
        assert relaxation>0, "relaxation must be positive"
        extr = (1-relaxation)*X[:,-1] + relaxation*extr

    return extr


def accelerate(optimizer, relaxation: float = 1.0, regularization: float = 0.0, history_depth: int = 15,
               store_each_nth: int = 10, frequency: int = 10, resample_frequency: int = 400):
    # acceleration options
    optimizer.acc_relaxation = relaxation
    optimizer.acc_regularization = regularization

    optimizer.acc_history_depth = history_depth  # history size
    optimizer.acc_store_each_nth = store_each_nth  # frequency to update history
    optimizer.acc_frequency = frequency  # frequency to accelerate
    optimizer.resample_frequency = resample_frequency  # resample frequency important for tracking history

    # TODO: add averaging or other methods

    # acceleration history
    optimizer.acc_call_counter = 0
    optimizer.acc_store_counter = 0
    optimizer.acc_param_hist = []
    optimizer.res_hist = []

    # redefine step of the optimizer
    optimizer.orig_step = optimizer.step

    optimizer.step = MethodType(accelerated_step, optimizer)

    return optimizer



# TODO: add acceeleration removal

def clear_hist(optimizer):
    # clear history when resampling
    print('clear history at ', optimizer.acc_store_counter)
    optimizer.acc_param_hist = []
    optimizer.res_hist = []


def accelerated_step(self, closure):
    # test, single parameter group! currently doesn't support spliting parameters into multiple groups
    if closure is None:
        print('Unable to perform acceleration without closure')
        sys.exit()

    self.orig_step(closure)

    res, loss = closure()  # calculate the residual
    # add current parameters to the history
    self.acc_store_counter += 1
    if self.acc_store_counter % self.acc_store_each_nth == 0 and len(self.res_hist) < self.acc_history_depth:
        for group in self.param_groups:
            self.acc_param_hist.append(
                parameters_to_vector(group['params']).detach())  # network parameters, single group!!

        self.res_hist.append(res.detach())  # residual from current network parameters

    # perform acceleration
    self.acc_call_counter += 1
    if self.acc_call_counter % self.acc_frequency == 0:
        if len(self.acc_param_hist) >= 3:
            # make matrix of updates from the history list
            X = torch.stack(list(self.acc_param_hist), dim=1)
            R = torch.stack(list(self.res_hist), dim=1)
            acc_param = anderson_qr_fun(X, R, self.acc_relaxation, self.acc_regularization)

            # check performance
            for group in self.param_groups:
                vector_to_parameters(acc_param, group['params'])
            new_res, new_loss = closure()
            if new_loss < loss:
                self.acc_param_hist.pop()
                self.acc_param_hist.append(acc_param)
                self.res_hist.pop()
                self.res_hist.append(new_res.detach())

                print(
                    'acceleration working at step %5d, improve loss by %.5f' % (self.acc_call_counter, loss - new_loss))
            else:
                # revert to non-accelerated params
                vector_to_parameters(self.acc_param_hist[-1], group['params'])
