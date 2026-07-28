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
from src.experiment_utils import val_metrics, save_records

import AADL as AADL

# ## Problem Setup
#
# Consider (nonlinear) Helmholtz Equation
#
# $$ u^3 - \Delta u = f $$
#
# Exact solution (Method of Manufactured Solutions):
#
# $$ u = \exp\!\left(-\beta \sum_{i=1}^{d} (x_i - 0.5)^2\right) $$
#
# Domain: [0, 1]^d  (purely spatial, no time)
# The solution has a sharp interior spike at x = (0.5,...,0.5) when beta is large.
# Boundary conditions: u = exp(-beta*(d-1)*0.25 - beta*0.25) at each face — non-zero.
#
# Key derivatives:
#   u_{x_i}     = -2*beta*(x_i - 0.5) * u
#   u_{x_i x_i} = (-2*beta + 4*beta^2*(x_i-0.5)^2) * u
#   Delta u = u * sum_i(-2*beta + 4*beta^2*(x_i-0.5)^2)
#   f = u^3 - Delta u


def data_gen(x, beta):
    # exact solution to Helmholtz equation
    sol = torch.exp(-beta * torch.sum((x - 0.5) ** 2, dim=1).view(-1, 1))
    return sol


def forcing(x, beta):
    # forcing term computed analytically via MMS
    u = data_gen(x, beta)
    lap = u * torch.sum(-2 * beta + 4 * beta ** 2 * (x - 0.5) ** 2, dim=1).view(-1, 1)
    return u ** 3 - lap


def bound_data(n, d):
    # sample points on the boundary of [0, 1]^d
    n0 = math.floor(n / d / 2)
    x = torch.empty(n, d)
    for i in range(d):
        x0 = torch.rand(n0, d)
        x0[:, i] = 0.
        x[i * 2 * n0:i * 2 * n0 + n0, :] = x0
        x0 = torch.rand(n0, d)
        x0[:, i] = 1.
        x[i * 2 * n0 + n0:(i + 1) * 2 * n0, :] = x0
    if n % d != 0:
        n1 = n % d
        idx = torch.randint(0, n - n1, (n1,))
        x[n - n1:, :] = x[idx, :]
    return x


def loss_helmholtz(x, y, x_to_train_f, d, net, beta):
    """
    :param x: boundary condition points
    :param y: exact solution values at x
    :param x_to_train_f: interior collocation points
    :param d: spatial dimension
    :param net: neural network
    :param beta: spike sharpness parameter
    :return: (residual vector, scalar loss)
    """
    loss_fun = nn.MSELoss()
    loss_BC = loss_fun(net.forward(x), y)

    g = x_to_train_f.clone()
    g.requires_grad = True

    u = net.forward(g)
    u_x = autograd.grad(
        u, g,
        torch.ones([x_to_train_f.shape[0], 1]).to(g.device),
        retain_graph=True, create_graph=True,
    )[0]

    num = x_to_train_f.shape[0]
    lap = torch.zeros(num, 1).to(g.device)
    for i in range(d):
        vec = torch.zeros_like(u_x)
        vec[:, i] = torch.ones(num)
        u_xx_i = autograd.grad(u_x, g, vec, create_graph=True)[0]
        lap = lap + u_xx_i[:, [i]]

    f = u ** 3 - lap
    ff = forcing(g, beta)

    loss_PDE = loss_fun(f, ff)
    loss = loss_BC + loss_PDE

    res_PDE = f - ff
    res_BC = net.forward(x) - y
    res = torch.cat((res_PDE, res_BC), dim=0)
    res = torch.flatten(res)
    return res, loss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ", device)

# Parameter controlling the sharpness of the interior spike.
# Large beta => narrow spike => more challenging for the network.
beta = 5.0

d = 100
layers = np.array([d, 50, 50, 50, 1])

niters = 3000
N_u = 400
N_f = 4000
lr = 0.01
print_freq = 100
num_repeats = 5
acceleration_type = "anderson"
relaxation = 0.5
history_depth = 10
store_each_nth = 1
frequency = 5
resample = 500
average = True

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
    y = data_gen(x, beta).to(device)
    x_to_train_f = torch.rand(N_f, d).to(device)
    x_val = torch.rand(500, d).to(device)
    y_val = data_gen(x_val, beta).to(device)

    net = MLP(layers)
    net.to(device)
    optim = torch.optim.Adam(net.parameters(), lr=lr)
    record[0, repeat] = loss_helmholtz(x, y, x_to_train_f, d, net, beta)[1].detach()

    for itr in range(1, niters + 1):
        optim.zero_grad()
        loss = loss_helmholtz(x, y, x_to_train_f, d, net, beta)[1]
        loss.backward()
        optim.step()
        record[itr, repeat] = loss.detach()
        if itr % print_freq == 0:
            print(("%06d    " + "%1.4e    ") % (itr, loss))
        if itr % resample == 0:
            x = bound_data(N_u, d).to(device)
            y = data_gen(x, beta).to(device)
            x_to_train_f = torch.rand(N_f, d).to(device)

    err_abs, err_rel = val_metrics(net, x_val, y_val)
    err_average += err_rel
    print(f"Validation: L1={err_abs:.4e}  rel-L2={err_rel:.4e}")

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
    y = data_gen(x, beta).to(device)
    x_to_train_f = torch.rand(N_f, d).to(device)
    x_val = torch.rand(500, d).to(device)
    y_val = data_gen(x_val, beta).to(device)

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
    record[0, repeat] = loss_helmholtz(x, y, x_to_train_f, d, net, beta)[1].detach()

    _last_loss = [None]
    for itr in range(1, niters + 1):
        def closure():
            optim.zero_grad()
            _, loss = loss_helmholtz(x, y, x_to_train_f, d, net, beta)
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
            y = data_gen(x, beta).to(device)
            x_to_train_f = torch.rand(N_f, d).to(device)

    err_abs, err_rel = val_metrics(net, x_val, y_val)
    err_average += err_rel
    print(f"Validation: L1={err_abs:.4e}  rel-L2={err_rel:.4e}")

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
    y = data_gen(x, beta).to(device)
    x_to_train_f = torch.rand(N_f, d).to(device)
    x_val = torch.rand(500, d).to(device)
    y_val = data_gen(x_val, beta).to(device)

    net = MLP(layers)
    net.to(device)
    optim = torch.optim.Adam(net.parameters(), lr=lr)
    accelerate(optim, relaxation=1.0, store_each_nth=store_each_nth,
               history_depth=history_depth, frequency=frequency)
    record[0, repeat] = loss_helmholtz(x, y, x_to_train_f, d, net, beta)[1].detach()

    _last_loss = [None]
    for itr in range(1, niters + 1):
        def closure():
            optim.zero_grad()
            res, loss = loss_helmholtz(x, y, x_to_train_f, d, net, beta)
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
            y = data_gen(x, beta).to(device)
            x_to_train_f = torch.rand(N_f, d).to(device)
            clear_hist(optim)

    err_abs, err_rel = val_metrics(net, x_val, y_val)
    err_average += err_rel
    print(f"Validation: L1={err_abs:.4e}  rel-L2={err_rel:.4e}")

print("average validation error: ", err_average / num_repeats)
print("Training time: %.2f" % (time.time() - start_time))
record_DDAADL = record
save_records(__file__, record_default, record_AADL, record_DDAADL)

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
plt.title(f"{d}d Helmholtz Equation (u³−Δu=f) – Interior spike (β={beta})")
fig.savefig("HighD_Helmholtz_interior_spike.jpg", dpi=500)
