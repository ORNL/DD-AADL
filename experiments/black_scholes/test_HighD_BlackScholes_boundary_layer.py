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
# Exact solution (Method of Manufactured Solutions) – boundary layer profile:
#
# $$ u = e^{-t} \prod_{i=1}^{d-1} \phi_i(x_i), \quad
#    \phi_i(x) = x(1-x)\frac{A}{\varepsilon}
#               \!\left(e^{-\beta(x-\varepsilon)^2} + e^{-\beta(x-1+\varepsilon)^2}\right) $$
#
# Domain: [0,1]^{d-1} x [0,1]
# Boundary conditions: u = 0 on all spatial faces (x_i=0 or x_i=1).
# The functions phi_i vanish at the endpoints because x*(1-x)=0 there.
#
# Forcing term: f = u_t + r*sum_i(x_i*u_{x_i}) + sigma^2/2*sum_i(x_i^2*u_{x_ix_i}) - r*u
# The first/second derivatives of phi_i are the same as in test_HighD_Burgers_boundary_layer.py,
# only the PDE operator (advection weights r*x_i, diffusion weights sigma^2/2*x_i^2) differs.


def data_gen(x, A, beta, eps):
    # exact solution to Black-Scholes equation
    d = x.shape[1]
    n = x.shape[0]
    sol = torch.ones(n, 1, device=x.device)
    for i in range(d - 1):
        xx = x[:, i]
        k = xx * (1 - xx) * (A / eps) * (
            torch.exp(-beta * (xx - eps) ** 2) + torch.exp(-beta * (xx - 1 + eps) ** 2)
        )
        sol = sol * k.view(-1, 1)
    sol = sol * torch.exp(-x[:, -1]).view(-1, 1)
    return sol


def forcing(x, A, beta, eps):
    # forcing term computed analytically via MMS
    d = x.shape[1]
    n = x.shape[0]
    u = data_gen(x, A, beta, eps)
    ut = -u

    conv = torch.zeros(n, 1, device=x.device)
    diff = torch.zeros(n, 1, device=x.device)

    for i in range(d - 1):
        xx = x[:, i]

        # product of phi_j for j != i, times exp(-t)  =>  u / phi_i
        part1 = torch.ones(n, 1, device=x.device)
        for j in range(d - 1):
            if j != i:
                xj = x[:, j]
                k = xj * (1 - xj) * (A / eps) * (
                    torch.exp(-beta * (xj - eps) ** 2) + torch.exp(-beta * (xj - 1 + eps) ** 2)
                )
                part1 = part1 * k.view(-1, 1)
        part1 = part1 * torch.exp(-x[:, -1]).view(-1, 1)

        # phi_i'(x_i)
        G = torch.exp(-beta * (xx - eps) ** 2) + torch.exp(-beta * (xx - 1 + eps) ** 2)
        H = ((xx - eps) * torch.exp(-beta * (xx - eps) ** 2)
             + (xx - 1 + eps) * torch.exp(-beta * (xx - 1 + eps) ** 2))
        phi_prime = (1 - 2 * xx) * (A / eps) * G - xx * (1 - xx) * (2 * A * beta / eps) * H
        u_xi = part1 * phi_prime.view(-1, 1)
        conv = conv + rate * xx.view(-1, 1) * u_xi

        # phi_i''(x_i)
        K = ((xx - eps) ** 2 * torch.exp(-beta * (xx - eps) ** 2)
             + (xx - 1 + eps) ** 2 * torch.exp(-beta * (xx - 1 + eps) ** 2))
        phi_dbl = (-2 * (A / eps) * G
                   + ((xx - eps) * torch.exp(-beta * (xx - eps) ** 2)
                      + (xx - 1 + eps) * torch.exp(-beta * (xx - 1 + eps) ** 2))
                   * ((2 * xx - 1) * (4 * A * beta / eps))
                   - xx * (1 - xx) * (2 * A * beta / eps) * G
                   + xx * (1 - xx) * (4 * A * beta ** 2 / eps) * K)
        u_xxi = part1 * phi_dbl.view(-1, 1)
        diff = diff + 0.5 * sigma ** 2 * (xx.view(-1, 1) ** 2) * u_xxi

    f = ut + conv + diff - rate * u
    return f


def bound_data(n, d):
    # sample on boundary of [0,1]^{d-1} x [0,1]
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
    x0[:, -1] = 1.
    x[n - n1:, :] = x0
    return x


def loss_blackscholes(x, y, x_to_train_f, d, net, A, beta, eps):
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

    ff = forcing(g, A, beta, eps)
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

# Boundary-layer parameters
A    = 1.0
beta = 2.0
eps  = 1.0

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
    y = data_gen(x, A, beta, eps).to(device)
    x_to_train_f = torch.cat((torch.rand(N_f, d - 1), torch.rand(N_f, 1)), dim=1).to(device)
    x_val = torch.cat((torch.rand(500, d - 1), torch.rand(500, 1)), dim=1).to(device)
    y_val = data_gen(x_val, A, beta, eps).to(device)

    net = MLP(layers)
    net.to(device)
    optim = torch.optim.Adam(net.parameters(), lr=lr)
    record[0, repeat] = loss_blackscholes(x, y, x_to_train_f, d, net, A, beta, eps)[1].detach()

    for itr in range(1, niters + 1):
        optim.zero_grad()
        loss = loss_blackscholes(x, y, x_to_train_f, d, net, A, beta, eps)[1]
        loss.backward()
        optim.step()
        record[itr, repeat] = loss.detach()
        if itr % print_freq == 0:
            print(("%06d    " + "%1.4e    ") % (itr, loss))
        if itr % resample == 0:
            x = bound_data(N_u, d).to(device)
            y = data_gen(x, A, beta, eps).to(device)
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
    y = data_gen(x, A, beta, eps).to(device)
    x_to_train_f = torch.cat((torch.rand(N_f, d - 1), torch.rand(N_f, 1)), dim=1).to(device)
    x_val = torch.cat((torch.rand(500, d - 1), torch.rand(500, 1)), dim=1).to(device)
    y_val = data_gen(x_val, A, beta, eps).to(device)

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
    record[0, repeat] = loss_blackscholes(x, y, x_to_train_f, d, net, A, beta, eps)[1].detach()

    _last_loss = [None]
    for itr in range(1, niters + 1):
        def closure():
            optim.zero_grad()
            _, loss = loss_blackscholes(x, y, x_to_train_f, d, net, A, beta, eps)
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
            y = data_gen(x, A, beta, eps).to(device)
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
    y = data_gen(x, A, beta, eps).to(device)
    x_to_train_f = torch.cat((torch.rand(N_f, d - 1), torch.rand(N_f, 1)), dim=1).to(device)
    x_val = torch.cat((torch.rand(500, d - 1), torch.rand(500, 1)), dim=1).to(device)
    y_val = data_gen(x_val, A, beta, eps).to(device)

    net = MLP(layers)
    net.to(device)
    optim = torch.optim.Adam(net.parameters(), lr=lr)
    accelerate(optim, relaxation=1.0, store_each_nth=store_each_nth,
               history_depth=history_depth, frequency=frequency)
    record[0, repeat] = loss_blackscholes(x, y, x_to_train_f, d, net, A, beta, eps)[1].detach()

    _last_loss = [None]
    for itr in range(1, niters + 1):
        def closure():
            optim.zero_grad()
            res, loss = loss_blackscholes(x, y, x_to_train_f, d, net, A, beta, eps)
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
            y = data_gen(x, A, beta, eps).to(device)
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
plt.title(f"{d}d Black-Scholes Equation – Boundary layer solution")
fig.savefig("HighD_BlackScholes_boundary_layer.jpg", dpi=500)
