import time
import math
import torch
import torch.autograd as autograd  # computation graph
import torch.nn as nn

from src.NN_models import *
from src.anderson_acceleration import *

from src.utils import count_parameters

import AADL as AADL

# ## Problem Setup
#
# Consider Burgers Equation
#
# $$ u_t + u \cdot \nabla u - \Delta u = f         $$
#
#
# Suppose we define $u$ as a function  composite of polynomials and expenential fucntions, i.e.
#
# $$ u = e^{-t} (x_1 -1)(x_1 +1) \dots (x_n -1)(x_n + 1)$$


def data_gen(x):
    # solution to burgers equation
    d = x.shape[1]
    xx = x[:, : d - 1]
    xx = (xx - 1) * (xx + 1)
    sol = torch.prod(xx, dim=1).view(-1, 1)
    sol = sol * torch.exp(-x[:, -1].view(-1, 1))

    return sol


def forcing(x):
    # forcing term for the burgers equation
    d = x.shape[1]
    u = data_gen(x)
    ut = -u
    f = ut

    for i in range(d - 1):
        ux_i = u / ((x[:, i] - 1) * (x[:, i] + 1)).view(-1, 1)
        ux_i = 2 * x[:, i].view(-1, 1) * ux_i
        f = f + u * ux_i
    # laplacian
    for i in range(d - 1):
        uxx_i = 2 * u / ((x[:, i] - 1) * (x[:, i] + 1)).view(-1, 1)
        f = f - uxx_i

    return f

def bound_data(n, d):
    # n -- number of samples on boundary, may not be precise
    # d -- dimension of problem, last dim time
    # consider a boxed region with each axis from -1 to 1, time should be 0
    n0 = math.floor(n/d/2) # number of samples on each face of boundary
    x = torch.empty(n,d)
    for i in range(d-1):
        x0 = torch.cat(((2 * torch.rand(n0, d-1)) - 1, torch.rand(n0, 1)), dim=1)
        x0[:,i] = -1. ; x[i*2*n0:i*2*n0+n0,:] = x0
        x0 = torch.cat(((2 * torch.rand(n0, d - 1)) - 1, torch.rand(n0, 1)), dim=1)
        x0[:,i] = 1. ;  x[i*2*n0+n0:(i+1)*2*n0,:] = x0
    # for last dim -- time
    n1 = n - 2*n0*(d-1)
    x0 = 2*torch.rand(n1,d) - 1.
    x0[:,-1] = 0. ; x[n-n1:,:] = x0


    return x

# define a test problem
def loss_burgers(x, y, x_to_train_f, d, net):
    """
    :param x: input for boundary condition
    :param y: boundary data
    :param x_to_train_f: input for calculating PDE loss
    :param d: number of dimension
    :param net: network
    :return:  loss
    """

    ### u_t + u*u_x1 + u^2*u_x2 + ... + u^d*u_xd = 0 with boundary and initial condition
    loss_fun = nn.MSELoss()
    loss_BC = loss_fun(net.forward(x), y)

    g = x_to_train_f.clone()
    g.requires_grad = True

    u = net.forward(g)
    # gradient
    u_x_t = autograd.grad(
        u,
        g,
        torch.ones([x_to_train_f.shape[0], 1]).to(g.device),
        retain_graph=True,
        create_graph=True,
    )[0]

    u_t = u_x_t[:, [-1]]
    f = u_t
    for i in range(d - 1):
        # iterate for space states
        f = f + u * u_x_t[:, [i]]

    # laplacian? depending on the problem
    num = x_to_train_f.shape[0]
    lap = torch.zeros(num, 1).to(g.device)
    for i in range(d - 1):
        vec = torch.zeros_like(u_x_t)
        vec[:, i] = torch.ones(num)
        u_xx_i = autograd.grad(u_x_t, g, vec, create_graph=True)[0]
        u_xxi = u_xx_i[:, [i]]

        lap = lap + u_xxi

    f = f - lap

    ## forcing term
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

# Hyperparameters for the neural network
d = 100
layers = np.array([d, 50, 50, 50, 1])

# parameter list
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
frequency = 1
resample = 500
average = False

start_time = time.time()
print((2 * "%7s    ") % ("step", "Loss"))
err_average = 0.0

record = np.zeros([niters + 1, num_repeats])
for repeat in range(num_repeats):
    torch.manual_seed(repeat)
    x = bound_data(N_u,d).to(device)
    y = data_gen(x)
    y = y.to(device)
    x_to_train_f = torch.cat(
        ((2 * torch.rand(N_f, d - 1)) - 1, torch.rand(N_f, 1)), dim=1
    ).to(device)

    x_val = torch.cat(((2 * torch.rand(500, d - 1))-1, torch.rand(500, 1)), dim=1).to(device)
    y_val = data_gen(x_val)

    net = MLP(layers)
    net.to(device)
    optim = torch.optim.Adam(net.parameters(), lr=lr)
    record[0, repeat] = loss_burgers(x, y, x_to_train_f, d, net)[1].detach()

    for itr in range(1, niters + 1):

        optim.zero_grad()
        loss = loss_burgers(x, y, x_to_train_f, d, net)[1]
        loss.backward()
        optim.step()
        record[itr, repeat] = loss.detach()

        if itr % print_freq == 0:
            print(("%06d    " + "%1.4e    ") % (itr, loss))

        # resample
        if itr % resample == 0:
            x = bound_data(N_u,d).to(device)
            y = data_gen(x)
            y = y.to(device)
            x_to_train_f = torch.cat(
                ((2 * torch.rand(N_f, d - 1)) - 1, torch.rand(N_f, 1)), dim=1
            ).to(device)
            # clear_hist(optim)

        """
        # change learning rate
        if itr % 1000 == 0:
            for p in optim.param_groups:
                p['lr'] *= 0.5
        """

    # Validation
    err = torch.mean(torch.abs(y_val - net(x_val)))
    err_average += err
    print("Validation results, error in absolute value: ", err)

print("average validation error: ", err_average / 1)
elapsed = time.time() - start_time
print("Training time: %.2f" % (elapsed))

record_default = record

err_average = 0.0
record = np.zeros([niters + 1, num_repeats])
for repeat in range(num_repeats):
    torch.manual_seed(repeat)
    x = bound_data(N_u,d).to(device)
    y = data_gen(x)
    y = y.to(device)
    x_to_train_f = torch.cat(
        ((2 * torch.rand(N_f, d - 1)) - 1, torch.rand(N_f, 1)), dim=1
    ).to(device)

    x_val = torch.cat(((2 * torch.rand(500, d - 1))-1, torch.rand(500, 1)), dim=1).to(device)
    y_val = data_gen(x_val)

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
    record[0, repeat] = loss_burgers(x, y, x_to_train_f, d, net)[1].detach()

    for itr in range(1, niters + 1):

        def closure():
            optim.zero_grad()
            res, loss = loss_burgers(x, y, x_to_train_f, d, net)
            loss.backward()
            return loss

        optim.step(closure)
        loss = loss_burgers(x, y, x_to_train_f, d, net)[1]
        record[itr, repeat] = loss.detach()

        if itr % print_freq == 0:
            print(("%06d    " + "%1.4e    ") % (itr, loss))

        # resample
        if itr % resample == 0:
            x = bound_data(N_u,d).to(device)
            y = data_gen(x)
            y = y.to(device)
            x_to_train_f = torch.cat(
                ((2 * torch.rand(N_f, d - 1)) - 1, torch.rand(N_f, 1)), dim=1
            ).to(device)
            # clear_hist(optim)

        """
        # change learning rate
        if itr % 1000 == 0:
            for p in optim.param_groups:
                p['lr'] *= 0.5
        """

    # Validation
    err = torch.mean(torch.abs(y_val - net(x_val)))
    err_average += err
    print("Validation results, error in absolute value: ", err)

print("average validation error: ", err_average / 1)
elapsed = time.time() - start_time
print("Training time: %.2f" % (elapsed))

record_AADL = record

err_average = 0.0
record = np.zeros([niters + 1, num_repeats])
for repeat in range(num_repeats):
    torch.manual_seed(repeat)
    x = bound_data(N_u,d).to(device)
    y = data_gen(x)
    y = y.to(device)
    x_to_train_f = torch.cat(
        ((2 * torch.rand(N_f, d - 1)) - 1, torch.rand(N_f, 1)), dim=1
    ).to(device)

    x_val = torch.cat(((2 * torch.rand(500, d - 1))-1, torch.rand(500, 1)), dim=1).to(device)
    y_val = data_gen(x_val)

    net = MLP(layers)
    net.to(device)
    optim = torch.optim.Adam(net.parameters(), lr=lr)
    accelerate(optim, frequency=20)
    record[0, repeat] = loss_burgers(x, y, x_to_train_f, d, net)[1].detach()

    for itr in range(1, niters + 1):

        def closure():
            optim.zero_grad()
            res, loss = loss_burgers(x, y, x_to_train_f, d, net)
            loss.backward()
            return res, loss

        optim.step(closure)
        loss = loss_burgers(x, y, x_to_train_f, d, net)[1]
        record[itr, repeat] = loss.detach()

        if itr % print_freq == 0:
            print(("%06d    " + "%1.4e    ") % (itr, loss))

        # resample
        if itr % 500 == 0:
            x = bound_data(N_u,d).to(device)
            y = data_gen(x)
            y = y.to(device)
            x_to_train_f = torch.cat(
                ((2 * torch.rand(N_f, d - 1)) - 1, torch.rand(N_f, 1)), dim=1
            ).to(device)
            clear_hist(optim)

        """
        # change learning rate
        if itr % 1000 == 0:
            for p in optim.param_groups:
                p['lr'] *= 0.5
        """

    # Validation
    err = torch.mean(torch.abs(y_val - net(x_val)))
    err_average += err
    print("Validation results, error in absolute value: ", err)

print("average validation error: ", err_average / 1)
elapsed = time.time() - start_time
print("Training time: %.2f" % (elapsed))

record_DDAADL = record

import math
import matplotlib.pyplot as plt

avg = np.mean(record_default, axis=1)
std = np.std(record_default, axis=1)

fig = plt.figure()
plt.plot(range(niters + 1), avg, color="b", linewidth=2)
plt.fill_between(
    range(niters + 1),
    avg - std * 2 / math.sqrt(num_repeats),
    avg + std * 2 / math.sqrt(num_repeats),
    color="b",
    alpha=0.2,
)


AADL_avg = np.mean(record_AADL, axis=1)
AADL_std = np.std(record_AADL, axis=1)
plt.plot(range(niters + 1), AADL_avg, color="g", linewidth=2)
plt.fill_between(
    range(niters + 1),
    AADL_avg - AADL_std * 2 / math.sqrt(num_repeats),
    AADL_avg + AADL_std * 2 / math.sqrt(num_repeats),
    color="g",
    alpha=0.2,
)

DDAADL_avg = np.mean(record_DDAADL, axis=1)
DDAADL_std = np.std(record_DDAADL, axis=1)
plt.plot(range(niters + 1), DDAADL_avg, color="r", linewidth=2)
plt.fill_between(
    range(niters + 1),
    DDAADL_avg - DDAADL_std * 2 / math.sqrt(num_repeats),
    DDAADL_avg + DDAADL_std * 2 / math.sqrt(num_repeats),
    color="r",
    alpha=0.2,
)


plt.yscale("log")
plt.ylim([1.0e-8, 1.0e2])
plt.legend(["Adam", "Adam + AADL", "Adam + Data Driven AADL"])
plt.xlabel("Number of iterations")
plt.ylabel("Validation Mean Squared Error")
plt.title(f"{d}d Burgers' Equation")
fig.savefig("HighDBurgers_solution.jpg", dpi=500)
