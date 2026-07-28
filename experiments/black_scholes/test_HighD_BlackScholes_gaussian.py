import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import time
import math
import torch
import torch.autograd as autograd
import torch.nn as nn

from src.NN_models import *
from src.anderson_acceleration import *
from src.utils import count_parameters

import AADL as AADL

# ## Problem Setup
#
# Consider Black-Scholes Equation
#
# $$ u_t + r (x \cdot \nabla_x u) + \frac{\sigma^2}{2} (x^2 \cdot \nabla^2_x u) - r u = f $$
#
# Exact solution (Method of Manufactured Solutions):
#
# $$ u = \exp\!\left(-t - \sum_{i=1}^{d-1} x_i^2\right) $$
#
# Domain: [-2, 2]^{d-1} x [0, 1]
# Boundary conditions: non-zero on all spatial faces (Gaussian).
#
# Key derivatives:
#   u_{x_i}     = -2*x_i * u
#   u_{x_i x_i} = (-2 + 4*x_i^2) * u
#   u_t         = -u
#
# Forcing term:
#   f = -u  +  r*sum_i(-2*x_i^2*u)  +  sigma^2/2*sum_i(x_i^2*(-2+4*x_i^2)*u)  -  r*u
#     = u * (-(1+r))  +  u * sum_i [ (-2r - sigma^2)*x_i^2  +  2*sigma^2*x_i^4 ]


def data_gen(x):
    # exact solution to Black-Scholes equation
    d = x.shape[1]
    xx = x[:, :d - 1]
    sol = torch.exp(-x[:, -1].view(-1, 1) - torch.sum(xx ** 2, dim=1).view(-1, 1))
    return sol


def forcing(x):
    # forcing term computed analytically via MMS
    d = x.shape[1]
    u = data_gen(x)
    xx = x[:, :d - 1]
    f = -(1 + rate) * u
    f = f + u * torch.sum((-2 * rate - sigma ** 2) * xx ** 2 + 2 * sigma ** 2 * xx ** 4, dim=1).view(-1, 1)
    return f


def bound_data(n, d):
    # sample on boundary of [-2,2]^{d-1} x [0,1]
    # spatial faces at ±2; terminal condition at t=1
    n0 = math.floor(n / d / 2)
    x = torch.empty(n, d)
    for i in range(d - 1):
        x0 = torch.cat((4 * torch.rand(n0, d - 1) - 2, torch.rand(n0, 1)), dim=1)
        x0[:, i] = -2.
        x[i * 2 * n0:i * 2 * n0 + n0, :] = x0
        x0 = torch.cat((4 * torch.rand(n0, d - 1) - 2, torch.rand(n0, 1)), dim=1)
        x0[:, i] = 2.
        x[i * 2 * n0 + n0:(i + 1) * 2 * n0, :] = x0
    n1 = n - 2 * n0 * (d - 1)
    x0 = torch.cat((4 * torch.rand(n1, d - 1) - 2, torch.rand(n1, 1)), dim=1)
    x0[:, -1] = 1.
    x[n - n1:, :] = x0
    return x


def loss_blackscholes(x, y, x_to_train_f, d, net):
    """
    :param x: boundary / terminal condition points
    :param y: exact solution values at x
    :param x_to_train_f: interior collocation points
    :param d: problem dimension (d-1 spatial + 1 time)
    :param net: neural network
    :return: (residual vector, scalar loss)
    """
    loss_fun = nn.MSELoss()
    loss_BC = loss_fun(net.forward(x), y)

    g = x_to_train_f.clone()
    g.requires_grad = True

    u = net.forward(g)
    u_x_t = autograd.grad(
        u, g,
        torch.ones([x_to_train_f.shape[0], 1]).to(g.device),
        retain_graph=True, create_graph=True,
    )[0]

    u_t = u_x_t[:, [-1]]
    f = u_t
    for i in range(d - 1):
        f = f + rate * g[:, i].view(-1, 1) * u_x_t[:, [i]]

    num = x_to_train_f.shape[0]
    lap = torch.zeros(num, 1).to(g.device)
    for i in range(d - 1):
        vec = torch.zeros_like(u_x_t)
        vec[:, i] = torch.ones(num)
        u_xx_i = autograd.grad(u_x_t, g, vec, create_graph=True)[0]
        lap = lap + 0.5 * sigma ** 2 * (g[:, i].view(-1, 1) ** 2) * u_xx_i[:, [i]]

    f = f + lap - rate * u

    ff = forcing(g)
    loss_PDE = loss_fun(f, ff)
    loss = loss_BC + loss_PDE

    res_PDE = f - ff
    res_BC = net.forward(x) - y
    res = torch.cat((res_PDE, res_BC), dim=0)
    res = torch.flatten(res)
    return res, loss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ", device)

# PDE parameters
rate  = 0.1
sigma = 0.2

d = 100
layers = np.array([d, 50, 50, 50, 1])

niters = 3000
N_u = 400
N_f = 4000
lr = 0.01
print_freq = 100
num_repeats = 1
acceleration_type = "anderson"
relaxation = 0.5
history_depth = 10
store_each_nth = 1
frequency = 5
resample = 500
average = True

def _sample_interior(N, d, device):
    return torch.cat((4 * torch.rand(N, d - 1) - 2, torch.rand(N, 1)), dim=1).to(device)

# ---------------------------------------------------------------------------
# Loop 1: Adam baseline
# ---------------------------------------------------------------------------
start_time = time.time()
print((2 * "%7s    ") % ("step", "Loss"))
err_average = 0.0

record = np.zeros([niters + 1, num_repeats])
for repeat in range(num_repeats):
    torch.manual_seed(repeat)
    x = bound_data(N_u, d).to(device)
    y = data_gen(x).to(device)
    x_to_train_f = _sample_interior(N_f, d, device)
    x_val = _sample_interior(500, d, device)
    y_val = data_gen(x_val).to(device)

    net = MLP(layers)
    net.to(device)
    optim = torch.optim.Adam(net.parameters(), lr=lr)
    record[0, repeat] = loss_blackscholes(x, y, x_to_train_f, d, net)[1].detach()

    for itr in range(1, niters + 1):
        optim.zero_grad()
        loss = loss_blackscholes(x, y, x_to_train_f, d, net)[1]
        loss.backward()
        optim.step()
        record[itr, repeat] = loss.detach()
        if itr % print_freq == 0:
            print(("%06d    " + "%1.4e    ") % (itr, loss))
        if itr % resample == 0:
            x = bound_data(N_u, d).to(device)
            y = data_gen(x).to(device)
            x_to_train_f = _sample_interior(N_f, d, device)

    err = torch.mean(torch.abs(y_val - net(x_val)))
    err_average += err
    print("Validation results, error in absolute value: ", err)

print("average validation error: ", err_average / num_repeats)
print("Training time: %.2f" % (time.time() - start_time))
record_default = record

# ---------------------------------------------------------------------------
# Loop 2: Adam + AADL
# ---------------------------------------------------------------------------
start_time = time.time()
err_average = 0.0
record = np.zeros([niters + 1, num_repeats])
for repeat in range(num_repeats):
    torch.manual_seed(repeat)
    x = bound_data(N_u, d).to(device)
    y = data_gen(x).to(device)
    x_to_train_f = _sample_interior(N_f, d, device)
    x_val = _sample_interior(500, d, device)
    y_val = data_gen(x_val).to(device)

    net = MLP(layers)
    net.to(device)
    optim = torch.optim.Adam(net.parameters(), lr=lr)
    AADL.accelerate(
        optim,
        acceleration_type=acceleration_type,
        relaxation=relaxation,
        history_depth=history_depth,
        store_each_nth=store_each_nth,
        frequency=frequency,
        average=average,
    )
    record[0, repeat] = loss_blackscholes(x, y, x_to_train_f, d, net)[1].detach()

    _last_loss = [None]
    for itr in range(1, niters + 1):
        def closure():
            optim.zero_grad()
            _, loss = loss_blackscholes(x, y, x_to_train_f, d, net)
            loss.backward()
            _last_loss[0] = loss
            return loss
        optim.step(closure)
        loss = _last_loss[0]
        record[itr, repeat] = loss.detach()
        if itr % print_freq == 0:
            print(("%06d    " + "%1.4e    ") % (itr, loss))
        if itr % resample == 0:
            x = bound_data(N_u, d).to(device)
            y = data_gen(x).to(device)
            x_to_train_f = _sample_interior(N_f, d, device)

    err = torch.mean(torch.abs(y_val - net(x_val)))
    err_average += err
    print("Validation results, error in absolute value: ", err)

print("average validation error: ", err_average / num_repeats)
print("Training time: %.2f" % (time.time() - start_time))
record_AADL = record

# ---------------------------------------------------------------------------
# Loop 3: Adam + DD-AADL
# ---------------------------------------------------------------------------
start_time = time.time()
err_average = 0.0
record = np.zeros([niters + 1, num_repeats])
for repeat in range(num_repeats):
    torch.manual_seed(repeat)
    x = bound_data(N_u, d).to(device)
    y = data_gen(x).to(device)
    x_to_train_f = _sample_interior(N_f, d, device)
    x_val = _sample_interior(500, d, device)
    y_val = data_gen(x_val).to(device)

    net = MLP(layers)
    net.to(device)
    optim = torch.optim.Adam(net.parameters(), lr=lr)
    accelerate(optim, relaxation=1.0, store_each_nth=store_each_nth,
               history_depth=history_depth, frequency=frequency)
    record[0, repeat] = loss_blackscholes(x, y, x_to_train_f, d, net)[1].detach()

    _last_loss = [None]
    for itr in range(1, niters + 1):
        def closure():
            optim.zero_grad()
            res, loss = loss_blackscholes(x, y, x_to_train_f, d, net)
            loss.backward()
            _last_loss[0] = loss
            return res, loss
        optim.step(closure)
        loss = _last_loss[0]
        record[itr, repeat] = loss.detach()
        if itr % print_freq == 0:
            print(("%06d    " + "%1.4e    ") % (itr, loss))
        if itr % resample == 0:
            x = bound_data(N_u, d).to(device)
            y = data_gen(x).to(device)
            x_to_train_f = _sample_interior(N_f, d, device)
            clear_hist(optim)

    err = torch.mean(torch.abs(y_val - net(x_val)))
    err_average += err
    print("Validation results, error in absolute value: ", err)

print("average validation error: ", err_average / num_repeats)
print("Training time: %.2f" % (time.time() - start_time))
record_DDAADL = record

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
import matplotlib.pyplot as plt

fig = plt.figure()
for rec, color, label in [
    (record_default, "b", "Adam"),
    (record_AADL,    "g", "Adam + AADL"),
    (record_DDAADL,  "r", "Adam + Data Driven AADL"),
]:
    avg = np.mean(rec, axis=1)
    std = np.std(rec, axis=1)
    plt.plot(range(niters + 1), avg, color=color, linewidth=2, label=label)
    plt.fill_between(range(niters + 1),
                     avg - std * 2 / math.sqrt(num_repeats),
                     avg + std * 2 / math.sqrt(num_repeats),
                     color=color, alpha=0.2)

plt.yscale("log")
plt.ylim([1.0e-8, 1.0e2])
plt.legend()
plt.xlabel("Number of iterations")
plt.ylabel("Loss")
plt.title(f"{d}d Black-Scholes Equation – Gaussian solution")
fig.savefig("HighD_BlackScholes_gaussian.jpg", dpi=500)
