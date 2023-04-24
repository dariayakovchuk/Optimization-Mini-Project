import torch

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