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
# Consider the linear Heat (parabolic diffusion) Equation
#
# $$ u_t - \Delta_x u = f $$
#
# Exact solution (Method of Manufactured Solutions):
#
# $$ u = e^{-t} \prod_{i=1}^{d-1} \cos\!\left(\frac{\pi x_i}{2}\right) $$
#
# Domain: [-1, 1]^{d-1} x [0, 1]
# Boundary conditions: u = 0 on all spatial faces (cos(±pi/2) = 0);
#                      initial condition u|_{t=0} = prod cos(pi*x_i/2).
#
# Key derivatives:
#   u_t         = -u
#   u_{x_i x_i} = -(pi/2)^2 * u   =>  Delta_x u = -(d-1)*(pi/2)^2 * u
#
# Forcing:
#   f = u_t - Delta_x u = -u + (d-1)*(pi/2)^2 * u = u * ((d-1)*pi^2/4 - 1)


def data_gen(x):
    # exact solution to the heat equation
    d = x.shape[1]
    xx = x[:, :d - 1]
    sol = torch.exp(-x[:, -1].view(-1, 1)) * torch.prod(torch.cos(math.pi * xx / 2), dim=1).view(-1, 1)
    return sol


def forcing(x):
    # forcing term computed analytically via MMS
    d = x.shape[1]
    u = data_gen(x)
    return u * ((d - 1) * math.pi ** 2 / 4 - 1)


def bound_data(n, d):
    # sample on boundary of [-1, 1]^{d-1} x [0, 1]
    # spatial faces at ±1 (where cosine vanishes) plus t=0 initial condition
    n0 = math.floor(n / d / 2)
    x = torch.empty(n, d)
    for i in range(d - 1):
        x0 = torch.cat((2 * torch.rand(n0, d - 1) - 1, torch.rand(n0, 1)), dim=1)
        x0[:, i] = -1.
        x[i * 2 * n0:i * 2 * n0 + n0, :] = x0
        x0 = torch.cat((2 * torch.rand(n0, d - 1) - 1, torch.rand(n0, 1)), dim=1)
        x0[:, i] = 1.
        x[i * 2 * n0 + n0:(i + 1) * 2 * n0, :] = x0
    n1 = n - 2 * n0 * (d - 1)
    x0 = torch.cat((2 * torch.rand(n1, d - 1) - 1, torch.rand(n1, 1)), dim=1)
    x0[:, -1] = 0.
    x[n - n1:, :] = x0
    return x


def loss_heat(x, y, x_to_train_f, d, net):
    """
    :param x: boundary / initial condition points
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

    num = x_to_train_f.shape[0]
    lap = torch.zeros(num, 1).to(g.device)
    for i in range(d - 1):
        vec = torch.zeros_like(u_x_t)
        vec[:, i] = torch.ones(num)
        u_xx_i = autograd.grad(u_x_t, g, vec, create_graph=True)[0]
        lap = lap + u_xx_i[:, [i]]

    f = u_t - lap
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
    y = data_gen(x).to(device)
    x_to_train_f = torch.cat((2 * torch.rand(N_f, d - 1) - 1, torch.rand(N_f, 1)), dim=1).to(device)
    x_val = torch.cat((2 * torch.rand(500, d - 1) - 1, torch.rand(500, 1)), dim=1).to(device)
    y_val = data_gen(x_val).to(device)

    net = MLP(layers)
    net.to(device)
    optim = torch.optim.Adam(net.parameters(), lr=lr)
    record[0, repeat] = loss_heat(x, y, x_to_train_f, d, net)[1].detach()

    for itr in range(1, niters + 1):
        optim.zero_grad()
        loss = loss_heat(x, y, x_to_train_f, d, net)[1]
        loss.backward()
        optim.step()
        record[itr, repeat] = loss.detach()
        if itr % print_freq == 0:
            print(("%06d    " + "%1.4e    ") % (itr, loss))
        if itr % resample == 0:
            x = bound_data(N_u, d).to(device)
            y = data_gen(x).to(device)
            x_to_train_f = torch.cat((2 * torch.rand(N_f, d - 1) - 1, torch.rand(N_f, 1)), dim=1).to(device)

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
    y = data_gen(x).to(device)
    x_to_train_f = torch.cat((2 * torch.rand(N_f, d - 1) - 1, torch.rand(N_f, 1)), dim=1).to(device)
    x_val = torch.cat((2 * torch.rand(500, d - 1) - 1, torch.rand(500, 1)), dim=1).to(device)
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
    record[0, repeat] = loss_heat(x, y, x_to_train_f, d, net)[1].detach()

    _last_loss = [None]
    for itr in range(1, niters + 1):
        def closure():
            with torch.enable_grad():
                optim.zero_grad()
                _, loss = loss_heat(x, y, x_to_train_f, d, net)
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
            x_to_train_f = torch.cat((2 * torch.rand(N_f, d - 1) - 1, torch.rand(N_f, 1)), dim=1).to(device)

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
    y = data_gen(x).to(device)
    x_to_train_f = torch.cat((2 * torch.rand(N_f, d - 1) - 1, torch.rand(N_f, 1)), dim=1).to(device)
    x_val = torch.cat((2 * torch.rand(500, d - 1) - 1, torch.rand(500, 1)), dim=1).to(device)
    y_val = data_gen(x_val).to(device)

    net = MLP(layers)
    net.to(device)
    optim = torch.optim.Adam(net.parameters(), lr=lr)
    accelerate(optim, relaxation=1.0, store_each_nth=store_each_nth,
               history_depth=history_depth, frequency=frequency)
    record[0, repeat] = loss_heat(x, y, x_to_train_f, d, net)[1].detach()

    _last_loss = [None]
    for itr in range(1, niters + 1):
        def closure():
            optim.zero_grad()
            res, loss = loss_heat(x, y, x_to_train_f, d, net)
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
            x_to_train_f = torch.cat((2 * torch.rand(N_f, d - 1) - 1, torch.rand(N_f, 1)), dim=1).to(device)
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
plt.title(f"{d}d Heat Equation (u_t − Δu = f) – Cosine solution")
fig.savefig("HighD_Heat_cosine.jpg", dpi=500)
