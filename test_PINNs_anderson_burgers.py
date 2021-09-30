#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('unzip /content/DataDrivenAcceleration-XJ.zip -d /content/DataDrivenAcceleration-XJ')
get_ipython().system('pip install pydoe')

get_ipython().run_line_magic('cd', '/content/DataDrivenAcceleration-XJ')


# In[2]:


import torch
import torch.optim as optim

import numpy as np
import time
from pyDOE import lhs         #Latin Hypercube Sampling
import scipy.io
import argparse


from src.NN_models import *
from src.pinns_burgers_model import *
from src.plotter import *
from src.anderson_acceleration import *
import AADL as AADL


# ## Data Prep

# In[3]:


# Data Prep
data = scipy.io.loadmat('/Users/7ml/Documents/NSF-MSGI/XingjianLi_work/DataDrivenAcceleration/Data/burgers_shock_mu_01_pi.mat')
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


# ## Training

# In[4]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device: ', device)


N_u = 400
N_f = 4096
# X_f_train, X_u_train, u_train = trainingdata(N_u,N_f)
# X_f_train=X_f_train.to(device); X_u_train=X_u_train.to(device); u_train=u_train.to(device)


niters = 2000
lr = 0.01
print_freq = 50

layers = np.array([2,20,20,20,20,20,20,20,20,1]) #8 hidden layers
# net = MLP(layers); net.to(device)

# optim = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=0)
# accelerate(optim, frequency = 5)

start_time = time.time() 
print((2 * "%7s    ") % ("step", "Loss"))

# test data
X_u_test = X_u_test.to(device); u = u.to(device)

record = np.zeros((niters+1,20))
validation_err = 0.
for repeat in range(20):

    X_f_train, X_u_train, u_train = trainingdata(N_u,N_f)
    X_f_train=X_f_train.to(device); X_u_train=X_u_train.to(device); u_train=u_train.to(device)
    net = MLP(layers); net.to(device)
    idx_loss = loss_burgers(X_u_train, u_train, X_f_train, net)[0]
    record[0,repeat] = idx_loss.detach()

    optim = torch.optim.Adam(net.parameters(), lr=lr)
    # accelerate(optim, frequency = 5)
    #AADL.accelerate(optim, relaxation=1.0, store_each_nth=10, frequency=5)

    for itr in range(1, niters + 1):
        def closure():
            optim.zero_grad()
            loss, res = loss_burgers(X_u_train, u_train, X_f_train, net)
            loss.backward()
            return loss
        optim.step(closure)
        idx_loss = loss_burgers(X_u_train, u_train, X_f_train, net)[0]
        
        record[itr,repeat] = idx_loss.detach()

        # print
        if itr % print_freq == 0:
            print(("%06d    " + "%1.4e    ") %(itr, idx_loss))
        
        
        # change learning rate
        if itr % (niters/4) == 0:
            for p in optim.param_groups:
                p['lr'] *= 0.1
        

        # resample
        if itr % 500 == 0:
            X_f_train, X_u_train, u_train = trainingdata(N_u,N_f)
            X_f_train=X_f_train.to(device); X_u_train=X_u_train.to(device); u_train=u_train.to(device)
            # clear_hist(optim)

    # validation
    u_pred = net.forward(X_u_test)
    err = torch.linalg.norm((u-u_pred),2)/torch.linalg.norm(u,2)
    u_pred = u_pred.cpu().detach().numpy()
    u_pred = np.reshape(u_pred, (256, 100), order='F')
    print('Validation results, error in absolute value: ', err)
    validation_err += err


elapsed = time.time() - start_time
print('Training time: %.2f' % (elapsed))


# In[ ]:


record_AADL = record


# In[ ]:


# save the data
from scipy.io import savemat
# record_sgd = record
record_sgd_anderson = record

mdic = {"record_sgd_anderson": record_sgd_anderson, "label": "Burgers SGD Anderson"}
savemat("Burgers SGD Anderson 20 itr.mat", mdic)

print('validation error: ', validation_err/20)


# In[ ]:


import matplotlib.pyplot as plt

plt.plot(range(niters+1),record_sgd[:,0], color='b', linewidth=2)
plt.plot(range(niters+1),record_sgd_anderson[:,0], color='r', linewidth=2)
plt.yscale('log')


# In[7]:


anderson_avg = np.mean(record_anderson,axis=1)
anderson_std = np.std(record_anderson,axis=1)
avg = np.mean(record,axis=1)
std = np.std(record,axis=1)

fig = plt.figure()
plt.plot(range(niters+1),avg, color='b', linewidth=2)
plt.fill_between(range(niters+1), avg-std,avg+std,color='b',alpha=.2)
plt.plot(range(niters+1),anderson_avg, color='r', linewidth=2)
plt.fill_between(range(niters+1), anderson_avg-anderson_std,anderson_avg+anderson_std,color='r',alpha=.2)

AADL_avg = np.mean(record_AADL,axis=1)
AADL_std = np.std(record_AADL,axis=1)
plt.plot(range(niters+1),AADL_avg, color='g', linewidth=2)
plt.fill_between(range(niters+1), AADL_avg-AADL_std,AADL_avg+AADL_std,color='g',alpha=.2)


plt.yscale('log')
plt.ylim([0.001, 2])
plt.legend(['default', 'data_driven_acceleration', 'AADL'])
plt.xlabel('number of iteration')
plt.ylabel('loss')
plt.title('2D Burgers Equation')
fig.savefig('file.png', dpi=500)


# In[ ]:


X_u_train = X_u_train.cpu(); X_u_train = X_u_train.detach().numpy()
u_train = u_train.cpu(); u_train   = u_train.detach().numpy()

 
 
 
solutionplot_burgers(u_pred, X_u_train, u_train, X, T, x, t, usol) # TODO: check input

