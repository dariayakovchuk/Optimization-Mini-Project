import torch
import numpy as np
from inflation import BBI

def rosenbrock(x):
    """
    f(x) = sum_{i=1:D-1} 100*(x(i+1) - x(i)^2)^2 + (1-x(i))^2 
    The true minimum of the function is 0 at x = (1 1 ... 1)

    Returns:
        f : function value
        df : partial derivatives wrt x_i
    """
    D = len(x)
    f = torch.sum( 100 * torch.pow((x[1:D] - torch.pow(x[:D-1], 2)),2) + torch.pow(torch.ones(D-1) - x[:D-1], 2))
    
    df = torch.zeros(D)
    df[:D-1] = -400 * x[:D-1] * (x[1:D] - torch.pow(x[:D-1],2)) - 2 * (torch.ones(D-1) - x[:D-1])
    df[1:D] += 200 * (x[1:D] - torch.pow(x[:D-1],2))

    return f, df

def sgd_optimizer(x0, func, iterations = 1000, lr=0.0001, momentum=.95,):
    """
    this function is taken from https://github.com/gbdl/BBI
    """
    xslist = []
    xs = torch.tensor(x0)
    xs.requires_grad=True
    optimizer = torch.optim.SGD([xs], lr=lr, momentum = momentum)
    xs_best = torch.tensor(x0)
    minloss = np.inf
    trajectory = [xs.detach().numpy()]
    for i in range(iterations):
        optimizer.zero_grad()
        loss_fn, grad = func(xs)
        pos = xs - lr*grad
        trajectory.append(pos.detach().numpy())
        if  loss_fn.item() < minloss: 
            minloss = loss_fn.item()
            xs_best = xs.detach().clone()
        loss_fn.backward()
        optimizer.step()
        xslist.append(xs.tolist())
    return xs_best, xslist, trajectory


def adam_optimizer(x0, func, iterations = 1000, lr=0.0001,):
    xslist = []
    xs = torch.tensor(x0)
    xs.requires_grad=True
    optimizer = torch.optim.Adam([xs], lr=lr)
    xs_best = torch.tensor(x0)
    minloss = np.inf
    trajectory = [xs.detach().numpy()]
    for i in range(iterations):
        optimizer.zero_grad()
        loss_fn, grad = func(xs)
        pos = xs - lr*grad
        trajectory.append(pos.detach().numpy())
        if  loss_fn.item() < minloss: 
            minloss = loss_fn.item()
            xs_best = xs.detach().clone()
        loss_fn.backward()
        optimizer.step()
        xslist.append(xs.tolist())
    return xs_best, xslist, trajectory


def BBI_optimizer(x0, func, iterations = 1000, lr=0.0001, threshold0=50, threshold=1000000, v0=1e-10, deltaEn =.0, n_fixed_bounces = 1, consEn = True):
    """
    this function is taken from https://github.com/gbdl/BBI
    """
    xslist = []
    xs = torch.tensor(x0)
    xs.requires_grad=True
    optimizer = BBI([xs], lr=lr, threshold0 = int(threshold0), threshold = int(threshold), v0 = v0, deltaEn = deltaEn, consEn = consEn, n_fixed_bounces = n_fixed_bounces)
    xs_best = torch.tensor(x0)
    minloss = np.inf
    trajectory = [xs.detach().numpy()]
    for i in range(iterations):
        optimizer.zero_grad()
        loss_fn, grad = func(xs)
        pos = xs - lr*grad
        trajectory.append(pos.detach().numpy())
        if  loss_fn.item() < minloss: 
            minloss = loss_fn.item()
            xs_best = xs.detach().clone()
        loss_fn.backward()
        def closure():
                    return loss_fn
        optimizer.step(closure)
        xslist.append(xs.tolist())
    return xs_best, xslist, trajectory
