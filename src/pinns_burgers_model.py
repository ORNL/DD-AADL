from math import pi
import torch
import torch.autograd as autograd
from src.NN_models import *


def loss_burgers(x, y, x_to_train_f, net, nu=0.01 / pi):
    """

    :param x: input for boundary condition
    :param y: boundary data
    :param x_to_train_f: input for calculating PDE loss
    :param nu: comes from the PDE itself
    :param net: network
    :return:  loss
    """

    loss_fun = nn.MSELoss()
    loss_BC = loss_fun(net.forward(x), y)

    g = x_to_train_f.clone()
    g.requires_grad = True

    u = net.forward(g)
    # gradient
    u_x_t = autograd.grad(
        u, g, torch.ones_like(u).to(g.device), retain_graph=True, create_graph=True
    )[
        0
    ]  # TODO: check device
    # Hessian
    num = x_to_train_f.shape[0]
    vec = torch.cat([torch.ones(num, 1), torch.zeros(num, 1)], dim=1).to(g.device)
    u_xx_tt = autograd.grad(u_x_t, g, vec, create_graph=True)[0]

    u_x = u_x_t[:, [0]]
    u_t = u_x_t[:, [1]]
    u_xx = u_xx_tt[:, [0]]

    f = u_t + (net.forward(g)) * (u_x) - (nu) * u_xx
    loss_PDE = loss_fun(f, torch.zeros_like(f))

    loss = loss_BC + loss_PDE

    # calculate residual
    res_PDE = f
    res_BC = net.forward(x) - y
    res = torch.cat((res_PDE, res_BC), dim=0)
    res = torch.flatten(res)

    return loss, res


if __name__ == "__main__":

    # testing
    x = torch.tensor([1.0, 2.0])
    x = x.view(1, -1)
    y = torch.tensor([[10.0]])
    x_to_train_f = torch.rand(10, 2)

    layers = np.array([2, 50, 50, 1])
    net = MLP(layers)

    print("loss = ", loss_burgers(x, y, x_to_train_f, net, nu=0.01 / pi)[0])
