import torch
import torch.optim as optim

import numpy as np
import time
from pyDOE import lhs         #Latin Hypercube Sampling
import scipy.io
import argparse


from src.NN_models import *
from src.pinns_helmholtz_model import *
from src.plotter import *

# Default parameters
parser = argparse.ArgumentParser('Helmholtz PINNS')
parser.add_argument('--niters', type=int, default=2000)
parser.add_argument('--lr'    , type=float, default=0.01)
parser.add_argument('--optim' , type=str, default='adam', choices=['adam', 'lbfgs'])

parser.add_argument('--N_u' , type=int  , default=100, help="Total number of data points for u")
parser.add_argument('--N_f' , type=int  , default=4096, help="Total number of collocation points")

parser.add_argument('--lr_freq' , type=int  , default=100, help="how often to decrease lr")
parser.add_argument('--val_freq', type=int, default=50, help="how often to run model on validation set")
parser.add_argument('--print_freq', type=int, default=10, help="how often to print results")

args = parser.parse_args()

# Data Prep
x_1 = np.linspace(-1,1,256)  # 256 points between -1 and 1 [256x1]
x_2 = np.linspace(1,-1,256)  # 256 points between 1 and -1 [256x1]
X, Y = np.meshgrid(x_1,x_2)

# Test data
X_u_test = np.hstack((X.flatten(order='F')[:,None], Y.flatten(order='F')[:,None]))

a_1 = 1
a_2 = 1
usol = np.sin(a_1 * np.pi * X) * np.sin(a_2 * np.pi * Y) #solution chosen for convinience
u_true = torch.from_numpy(usol.flatten('F')[:,None]).float()

# Domain bounds
lb = np.array([-1, -1]) #lower bound
ub = np.array([1, 1])  #upper bound



# training data
def trainingdata(N_u, N_f):
    leftedge_x = np.hstack((X[:, 0][:, None], Y[:, 0][:, None]))
    leftedge_u = usol[:, 0][:, None]

    rightedge_x = np.hstack((X[:, -1][:, None], Y[:, -1][:, None]))
    rightedge_u = usol[:, -1][:, None]

    topedge_x = np.hstack((X[0, :][:, None], Y[0, :][:, None]))
    topedge_u = usol[0, :][:, None]

    bottomedge_x = np.hstack((X[-1, :][:, None], Y[-1, :][:, None]))
    bottomedge_u = usol[-1, :][:, None]

    all_X_u_train = np.vstack([leftedge_x, rightedge_x, bottomedge_x, topedge_x])
    all_u_train = np.vstack([leftedge_u, rightedge_u, bottomedge_u, topedge_u])

    # choose random N_u points for training
    idx = np.random.choice(all_X_u_train.shape[0], N_u, replace=False)

    X_u_train = all_X_u_train[idx[0:N_u], :]  # choose indices from  set 'idx' (x,t)
    u_train = all_u_train[idx[0:N_u], :]  # choose corresponding u

    '''Collocation Points'''

    # Latin Hypercube sampling for collocation points
    # N_f sets of tuples(x,t)
    X_f = lb + (ub - lb) * lhs(2, N_f)
    X_f_train = np.vstack((X_f, X_u_train))  # append training points to collocation points

    return torch.from_numpy(X_f_train).float(), torch.from_numpy(X_u_train).float(), torch.from_numpy(u_train).float()

# Parameter list
N_u = args.N_u
N_f = args.N_f
X_f_train, X_u_train, u_train = trainingdata(N_u,N_f)

layers = np.array([2, 50, 50, 50, 1]) #3 hidden layers
net = MLP(layers)
# net.to(device)

lr = args.lr
if args.optim=='adam':
    optim = optim.Adam(net.parameters(), lr=lr, weight_decay=0)
elif args.optim=='lbfgs':
    optim = torch.optim.LBFGS(net.parameters(), lr=lr, max_iter=20, max_eval=500, line_search_fn="strong_wolfe")
else:
    print('no valid optimizer')
    exit(1)

niters = args.niters
val_freq = args.val_freq
print_freq = args.print_freq
lr_freq = args.lr_freq


if __name__ == '__main__':
    start_time = time.time()   # TODO: implement time per iteration

    print((3 * "%7s    ") % ("step", "Loss", "Type"))

    net.train()
    for itr in range(1, niters + 1):
        if args.optim == 'adam':
            optim.zero_grad()
            _, loss = loss_helmholtz(X_u_train, u_train, X_f_train, net)
            loss.backward()
            optim.step()
        else: # lbfgs
            def closure():
                optim.zero_grad()
                _, loss = loss_helmholtz(X_u_train, u_train, X_f_train, net)
                loss.backward()
                return loss
            optim.step(closure)


        # print
        if itr % print_freq == 0:
            loss = loss_helmholtz(X_u_train, u_train, X_f_train, net)[1]
            print(("%06d    " + "%1.4e    " + "%5s") %(itr, loss, "Train"))

        # validation
        if itr % val_freq == 0 or itr == niters:
            u_pred = net.forward(X_u_test)
            err = torch.linalg.norm((u_true-u_pred),2)/torch.linalg.norm(u_true,2)

            u_pred = u_pred.cpu().detach().numpy()
            u_pred = np.reshape(u_pred, (256, 256), order='F')

            print(("%06d    " + "%1.4e    " + "%5s") % (itr, err, "Eval"))

        # shrink step size
        if itr % lr_freq == 0:
            for p in optim.param_groups:
                p['lr'] *= 0.1

        # TODO: resample

    elapsed = time.time() - start_time
    print('Training time: %.2f' % (elapsed))

    # TODO: add plotting
    solutionplot_helmholtz(u_pred, X_u_train, u_train, usol, x_1, x_2)






