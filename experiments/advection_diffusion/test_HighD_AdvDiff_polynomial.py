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
# Consider the Advection-Diffusion Equation
#
# $$ u_t + c \cdot \nabla_x u - \varepsilon \Delta_x u = f $$
#
# where c is a constant advection speed (same in all spatial directions).
# The Peclet number Pe = c / eps controls the dominance of advection vs diffusion.
# High Pe (e.g. c=1, eps=0.01) creates boundary layers and stresses PINNs.
#
# Exact solution (Method of Manufactured Solutions):
#
# $$ u = e^{-t} \prod_{i=1}^{d-1} x_i(1 - x_i) $$
#
# Domain: [0, 1]^{d-1} x [0, 1]
# Boundary conditions: u = 0 on all spatial faces (x_i=0 or x_i=1).
#
# Key derivatives (using product-of-others to avoid division-by-zero):
#   Let P_i = x_i*(1-x_i),  P_i' = 1-2*x_i,  P_i'' = -2
#   u_{x_i} = exp(-t) * P_i' * prod_{j!=i} P_j
#   u_{x_i x_i} = exp(-t) * (-2) * prod_{j!=i} P_j
#   u_t = -u
#
# Forcing:
#   f = -u  +  exp(-t) * c * sum_i [(1-2*x_i) * prod_{j!=i} P_j]
#           +  exp(-t) * 2*eps * sum_i [prod_{j!=i} P_j]


def data_gen(x):
    # exact solution to advection-diffusion equation
    d = x.shape[1]
    xx = x[:, :d - 1]
    pp = xx * (1 - xx)  # x_i*(1-x_i)
    sol = torch.exp(-x[:, -1].view(-1, 1)) * torch.prod(pp, dim=1).view(-1, 1)
    return sol


def forcing(x, c, eps):
    # forcing term computed analytically via MMS
    d = x.shape[1]
    n = x.shape[0]
    xx = x[:, :d - 1]
    pp = xx * (1 - xx)
    exp_t = torch.exp(-x[:, -1].view(-1, 1))

    u = exp_t * torch.prod(pp, dim=1).view(-1, 1)
    ut = -u

    conv = torch.zeros(n, 1, device=x.device)   # c * sum_i u_{x_i}
    diff = torch.zeros(n, 1, device=x.device)   # eps * sum_i u_{x_ix_i}
    for i in range(d - 1):
        mask = [j for j in range(d - 1) if j != i]
        prod_others = (torch.prod(pp[:, mask], dim=1).view(-1, 1)
                       if mask else torch.ones(n, 1, device=x.device))
        prod_others = prod_others * exp_t

        dprime_i = (1 - 2 * xx[:, [i]])          # P_i' = 1 - 2*x_i
        conv = conv + c * prod_others * dprime_i  # c * u_{x_i}
        diff = diff + eps * prod_others * (-2)    # eps * u_{x_ix_i}

    # PDE: u_t + c*grad_u - eps*lap_u = f  =>  diff already = eps * lap_u
    return ut + conv - diff


def bound_data(n, d):
    # sample on boundary of [0, 1]^{d-1} x [0, 1]
    n0 = math.floor(n / d / 2)
    x = torch.empty(n, d)
    for i in range(d - 1):
        x0 = torch.cat((torch.rand(n0, d - 1), torch.rand(n0, 1)), dim=1)
        x0[:, i] = 0.
        x[i * 2 * n0:i * 2 * n0 + n0, :] = x0
        x0 = torch.cat((torch.rand(n0, d - 1), torch.rand(n0, 1)), dim=1)
        x0[:, i] = 1.
        x[i * 2 * n0 + n0:(i + 1) * 2 * n0, :] = x0
    n1 = n - 2 * n0 * (d - 1)
    x0 = torch.cat((torch.rand(n1, d - 1), torch.rand(n1, 1)), dim=1)
    x0[:, -1] = 0.
    x[n - n1:, :] = x0
    return x


def loss_advdiff(x, y, x_to_train_f, d, net, c, eps):
    """
    :param x: boundary / initial condition points
    :param y: exact solution values at x
    :param x_to_train_f: interior collocation points
    :param d: problem dimension (d-1 spatial + 1 time)
    :param net: neural network
    :param c: advection speed (scalar, same in all spatial directions)
    :param eps: diffusion coefficient
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

    # advection: c * sum_i u_{x_i}
    f = u_t
    for i in range(d - 1):
        f = f + c * u_x_t[:, [i]]

    # diffusion: -eps * Laplacian
    num = x_to_train_f.shape[0]
    lap = torch.zeros(num, 1).to(g.device)
    for i in range(d - 1):
        vec = torch.zeros_like(u_x_t)
        vec[:, i] = torch.ones(num)
        u_xx_i = autograd.grad(u_x_t, g, vec, create_graph=True)[0]
        lap = lap + u_xx_i[:, [i]]

    f = f - eps * lap

    ff = forcing(g, c, eps)
    loss_PDE = loss_fun(f, ff)
    loss = loss_BC + loss_PDE

    res_PDE = f - ff
    res_BC = net.forward(x) - y
    res = torch.cat((res_PDE, res_BC), dim=0)
    res = torch.flatten(res)
    return res, loss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ", device)

# PDE parameters — increase Pe = c/eps for a more advection-dominated problem
c   = 1.0   # advection speed
eps = 1.0   # diffusion coefficient  (Pe = c/eps = 1; try c=1, eps=0.01 for Pe=100)

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
    x_to_train_f = torch.cat((torch.rand(N_f, d - 1), torch.rand(N_f, 1)), dim=1).to(device)
    x_val = torch.cat((torch.rand(500, d - 1), torch.rand(500, 1)), dim=1).to(device)
    y_val = data_gen(x_val).to(device)

    net = MLP(layers)
    net.to(device)
    optim = torch.optim.Adam(net.parameters(), lr=lr)
    record[0, repeat] = loss_advdiff(x, y, x_to_train_f, d, net, c, eps)[1].detach()

    for itr in range(1, niters + 1):
        optim.zero_grad()
        loss = loss_advdiff(x, y, x_to_train_f, d, net, c, eps)[1]
        loss.backward()
        optim.step()
        record[itr, repeat] = loss.detach()
        if itr % print_freq == 0:
            print(("%06d    " + "%1.4e    ") % (itr, loss))
        if itr % resample == 0:
            x = bound_data(N_u, d).to(device)
            y = data_gen(x).to(device)
            x_to_train_f = torch.cat((torch.rand(N_f, d - 1), torch.rand(N_f, 1)), dim=1).to(device)

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
    x_to_train_f = torch.cat((torch.rand(N_f, d - 1), torch.rand(N_f, 1)), dim=1).to(device)
    x_val = torch.cat((torch.rand(500, d - 1), torch.rand(500, 1)), dim=1).to(device)
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
    record[0, repeat] = loss_advdiff(x, y, x_to_train_f, d, net, c, eps)[1].detach()

    _last_loss = [None]
    for itr in range(1, niters + 1):
        def closure():
            optim.zero_grad()
            _, loss = loss_advdiff(x, y, x_to_train_f, d, net, c, eps)
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
            x_to_train_f = torch.cat((torch.rand(N_f, d - 1), torch.rand(N_f, 1)), dim=1).to(device)

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
    x_to_train_f = torch.cat((torch.rand(N_f, d - 1), torch.rand(N_f, 1)), dim=1).to(device)
    x_val = torch.cat((torch.rand(500, d - 1), torch.rand(500, 1)), dim=1).to(device)
    y_val = data_gen(x_val).to(device)

    net = MLP(layers)
    net.to(device)
    optim = torch.optim.Adam(net.parameters(), lr=lr)
    accelerate(optim, relaxation=1.0, store_each_nth=store_each_nth,
               history_depth=history_depth, frequency=frequency)
    record[0, repeat] = loss_advdiff(x, y, x_to_train_f, d, net, c, eps)[1].detach()

    _last_loss = [None]
    for itr in range(1, niters + 1):
        def closure():
            optim.zero_grad()
            res, loss = loss_advdiff(x, y, x_to_train_f, d, net, c, eps)
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
            x_to_train_f = torch.cat((torch.rand(N_f, d - 1), torch.rand(N_f, 1)), dim=1).to(device)
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

Pe = c / eps
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
plt.title(f"{d}d Advection-Diffusion (Pe={Pe:.1f}) – Polynomial solution")
fig.savefig("HighD_AdvDiff_polynomial.jpg", dpi=500)
