import torch
import torch.optim as optim

import numpy as np
import time
from pyDOE import lhs         #Latin Hypercube Sampling
import scipy.io


from src.NN_models import *
from src.pinns_burgers_model import *
from src.plotter import *




# Data Prep
data = scipy.io.loadmat('Data/burgers_shock_mu_01_pi.mat')
x = data['x']                                   # 256 points between -1 and 1 [256x1]
t = data['t']                                   # 100 time points between 0 and 1 [100x1]
usol = data['usol']                             # solution of 256x100 grid points

X, T = np.meshgrid(x,t)

# test data
X_u_test = np.hstack((X.flatten()[:,None], T.flatten()[:,None])); X_u_test = torch.from_numpy(X_u_test)
# Domain bounds
lb = X_u_test[0]  # [-1. 0.]
ub = X_u_test[-1] # [1.  0.99]
u_true = usol.flatten('F')[:,None]; u = torch.from_numpy(u_true)

def trainingdata(N_u,N_f):

    '''Boundary Conditions'''

    #Initial Condition -1 =< x =<1 and t = 0
    leftedge_x = np.hstack((X[0,:][:,None], T[0,:][:,None])) #L1
    leftedge_u = usol[:,0][:,None]

    #Boundary Condition x = -1 and 0 =< t =<1
    bottomedge_x = np.hstack((X[:,0][:,None], T[:,0][:,None])) #L2
    bottomedge_u = usol[-1,:][:,None]

    #Boundary Condition x = 1 and 0 =< t =<1
    topedge_x = np.hstack((X[:,-1][:,None], T[:,0][:,None])) #L3
    topedge_u = usol[0,:][:,None]

    all_X_u_train = np.vstack([leftedge_x, bottomedge_x, topedge_x]) # X_u_train [456,2] (456 = 256(L1)+100(L2)+100(L3))
    all_u_train = np.vstack([leftedge_u, bottomedge_u, topedge_u])   #corresponding u [456x1]

    #choose random N_u points for training
    idx = np.random.choice(all_X_u_train.shape[0], N_u, replace=False)

    X_u_train = all_X_u_train[idx, :] #choose indices from  set 'idx' (x,t)
    u_train = all_u_train[idx,:]      #choose corresponding u

    '''Collocation Points'''

    # Latin Hypercube sampling for collocation points
    # N_f sets of tuples(x,t)
    X_f_train = lb + (ub-lb)*lhs(2,N_f)
    X_f_train = np.vstack((X_f_train, X_u_train)) # append training points to collocation points

    return torch.from_numpy(X_f_train).float(), torch.from_numpy(X_u_train).float(), torch.from_numpy(u_train).float()


# Parameter list
N_u = 100 #Total number of data points for 'u'
N_f = 10000 #Total number of collocation points
X_f_train, X_u_train, u_train = trainingdata(N_u,N_f)

layers = np.array([2,20,20,20,20,20,20,20,20,1]) #8 hidden layers
net = MLP(layers)
# net.to(device)

lr = 0.01
optim = optim.Adam(net.parameters(), lr=lr, weight_decay=0)
niters = 4000

val_freq = 500
print_freq = 50
lr_freq = 3000




if __name__ == '__main__':
    start_time = time.time()   # TODO: implement time per iteration

    print((3 * "%7s    ") % ("step", "Loss", "Type"))

    net.train()
    for itr in range(1, niters + 1):
        optim.zero_grad()
        loss = loss_burgers(X_u_train, u_train, X_f_train, net)
        loss.backward()
        optim.step()


        # print
        if itr % print_freq == 0:
            print(("%06d    " + "%1.4e    " + "%5s") %(itr, loss, "Train"))

        # validation
        if itr % val_freq == 0 or itr == niters:
            net.eval()
            u_pred = net.forward(X_u_test)
            err = torch.linalg.norm((u-u_pred),2)/torch.linalg.norm(u,2)

            u_pred = u_pred.cpu().detach().numpy()
            u_pred = np.reshape(u_pred, (256, 100), order='F')

            print(("%06d    " + "%1.4e    " + "%5s") % (itr, err, "Eval"))

            net.train()

        # shrink step size
        if itr % lr_freq == 0:
            for p in optim.param_groups:
                p['lr'] *= 0.1

        # TODO: resample

    elapsed = time.time() - start_time
    print('Training time: %.2f' % (elapsed))

    solutionplot(u_pred, X_u_train, u_train, X, T, x, t, usol)







