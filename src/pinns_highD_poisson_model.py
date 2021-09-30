from math import pi
import torch
import torch.autograd as autograd
import torch.nn as nn
from src.NN_models import *


# define a test function
def poisson_gen(x):
    # create data for the problem
    # fix this for a better example
    A = torch.exp(x) + torch.sin(x)
    return torch.sum(A, dim=1).view(-1, 1)


def loss_poisson(x, y, x_to_train_f, d, net):
    '''
    :param x: input for boundary condition
    :param y: boundary data
    :param x_to_train_f: input for calculating PDE loss
    :param d: number of dimension
    :param net: network
    :return:  loss
    '''
    loss_fun = nn.MSELoss()
    loss_BC = loss_fun(net.forward(x), y)

    g = x_to_train_f.clone()
    g.requires_grad = True

    u = net.forward(g)
    # gradient
    u_x = \
    autograd.grad(u, g, torch.ones([x_to_train_f.shape[0], 1]).to(g.device), retain_graph=True, create_graph=True)[0]
    # laplacian
    num = x_to_train_f.shape[0]
    lap = torch.zeros(num, 1).to(g.device)
    for i in range(d):
        vec = torch.zeros_like(u_x)
        vec[:, i] = torch.ones(num)
        u_xx_i = autograd.grad(u_x, g, vec, create_graph=True)[0]
        u_xxi = u_xx_i[:, [i]]

        lap = lap + u_xxi

    f = lap - torch.sum(torch.exp(g), dim=1).view(-1, 1) + torch.sum(torch.sin(g), dim=1).view(-1, 1)

    loss_PDE = loss_fun(f, torch.zeros_like(f))
    loss = loss_BC + loss_PDE

    res_PDE = f
    res_BC = net.forward(x) - y
    res = torch.cat((res_PDE, res_BC), dim=0)
    res = torch.flatten(res)

    return res, loss


if __name__ == "__main__":
    # testing
    d = 5
    layers = np.array([d, 20, 20, 1])
    net = MLP(layers)

    x = torch.randn(10,d)
    y = poisson_gen(x)

    x_to_train_f = torch.randn(20,d)

    print('loss = ', loss_poisson(x, y, x_to_train_f, d, net)[1])
