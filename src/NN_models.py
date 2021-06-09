import torch
import torch.nn as nn
import numpy as np


class MLP(nn.Module):
    def __init__(self, layers):
        super().__init__()
        # input layers as list, array or tensor
        self.activation = nn.Tanh() # activation function
        self.net_len = len(layers)

        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i + 1]) for i in range(self.net_len - 1)])

        # initialization of weights
        for i in range(len(layers) - 1):
            nn.init.xavier_normal_(self.linears[i].weight.data, gain=1.0)
            nn.init.zeros_(self.linears[i].bias.data)


    def forward(self,x):
        if torch.is_tensor(x) != True:
            x = torch.from_numpy(x)    # TODO: check whether this will lead to device errors

        # no scaling of input data
        x = x.float()

        for i in range(self.net_len - 2):
            x = self.activation(self.linears[i](x))

        x = self.linears[-1](x)    # no activation for last layer

        return x


if __name__ == "__main__":

    # test case
    layers = np.array([2,50,50,1])
    net = MLP(layers)
    x = torch.Tensor([[1.0 ,4.0 ],[2.0,5.0],[3.0,6.0],[0.0,0.0]])
    y = net(x)
    print('y = ', y)