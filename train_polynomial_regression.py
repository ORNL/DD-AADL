# test problem
from __future__ import print_function
import torch

from src.anderson_acceleration import *

POLY_DEGREE = 4
W_target = torch.randn(POLY_DEGREE, 1) * 5
b_target = torch.randn(1) * 5

# Helper Functions
def make_features(x):
    """Builds features i.e. a matrix with columns [x, x^2, x^3, x^4]."""
    x = x.unsqueeze(1)
    return torch.cat([x ** i for i in range(1, POLY_DEGREE+1)], 1)

def f(x):
    """Approximated function."""
    return x.mm(W_target) + b_target.item()


def poly_desc(W, b):
    """Creates a string description of a polynomial."""
    result = 'y = '
    for i, w in enumerate(W):
        result += '{:+.2f} x^{} '.format(w, i + 1)
    result += '{:+.2f}'.format(b[0])
    return result


def get_batch(batch_size=32):
    """Builds a batch i.e. (x, f(x)) pair."""
    random = torch.randn(batch_size)
    x = make_features(random)
    y = f(x)
    return x, y

# Define model
fc = torch.nn.Linear(W_target.size(0), 1)
max_iters = 2000
lr = 0.01
optimizer_anderson = torch.optim.SGD(fc.parameters(), lr=lr, weight_decay=0)
accelerate(optimizer_anderson, frequency = 5)
resample_frequency = 400

batch_x, batch_y = get_batch(batch_size=64)
for idx in range(max_iters):
    # training

    def closure():
        optimizer_anderson.zero_grad()
        res = (fc(batch_x) - batch_y).reshape(-1)
        output = F.smooth_l1_loss(fc(batch_x), batch_y)
        loss = output.item()
        output.backward()
        return res, loss

    optimizer_anderson.step(closure)

    if idx % resample_frequency == 0:
        batch_x, batch_y = get_batch(batch_size=64)
        clear_hist(optimizer_anderson)

    # Stop criterion
    res, loss = closure()
    if loss < 1e-3:
        break

print('Loss: {:.6f} after {} batches'.format(loss, idx))
print('==> Learned function:\t' + poly_desc(fc.weight.view(-1), fc.bias))
print('==> Actual function:\t' + poly_desc(W_target.view(-1), b_target))