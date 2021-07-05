from math import pi
import torch
import torch.autograd as autograd
from src.NN_models import *

def loss_helmholtz(x, y, x_to_train_f, net):
    '''

    :param x: input for boundary condition
    :param y: boundary data
    :param x_to_train_f: input for calculating PDE loss
    :param nu: comes from the PDE itself
    :param net: network
    :return:  loss
    '''

    loss_fun = nn.MSELoss()
    loss_BC = loss_fun(net.forward(x), y)

    g = x_to_train_f.clone()
    g.requires_grad = True

    u = net.forward(g)
    # gradient
    u_x = autograd.grad(u, g, torch.ones([x_to_train_f.shape[0], 1]).to(g.device), retain_graph=True, create_graph=True)[0]
    # laplacian
    num = x_to_train_f.shape[0]
    vec1 = torch.cat([torch.ones(num, 1), torch.zeros(num, 1)], dim=1).to(g.device)
    vec2 = torch.cat([torch.zeros(num, 1), torch.ones(num, 1)], dim=1).to(g.device)
    u_xx1 = autograd.grad(u_x, g, vec1, create_graph=True)[0]
    u_xx_1 = u_xx1[:, [0]]
    u_xx2 = autograd.grad(u_x, g, vec2, create_graph=True)[0]
    u_xx_2 = u_xx2[:, [1]]

    # assemble
    a_1 = 1
    a_2 = 1
    k = 1

    x_1_f = x_to_train_f[:, [0]]
    x_2_f = x_to_train_f[:, [1]]

    q = (-(a_1 * pi) ** 2 - (a_2 * pi) ** 2 + k ** 2) * torch.sin(a_1 * pi * x_1_f) * torch.sin(
        a_2 * pi * x_2_f)
    f = u_xx_1 + u_xx_2 + k ** 2 * u - q
    loss_PDE = loss_fun(f, torch.zeros_like(f))

    loss = loss_BC + loss_PDE

    # calculate residual
    res_PDE = f
    res_BC = net.forward(x) - y
    res = torch.cat((res_PDE, res_BC), dim=0)
    res = torch.flatten(res)

    return res, loss

if __name__ == "__main__":

    # testing
    x = torch.tensor([1., 2.]); x = x.view(1, -1)
    y = torch.tensor([[10.0]])
    x_to_train_f = torch.rand(10,2)

    layers = np.array([2, 50, 50, 1])
    net = MLP(layers)

    print('loss = ', loss_helmholtz(x, y, x_to_train_f, net)[0])