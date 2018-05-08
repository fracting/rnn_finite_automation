import torch

def semantics_loss_fn(x, dim):
    epsilon = 1e-07
    x = torch.clamp(x, epsilon, 1 - epsilon)
    one_minus_x = 1 - x
    reduced_log_product = torch.sum(torch.log(one_minus_x), dim)
    loss = - torch.log(torch.sum(x / one_minus_x, dim)) - reduced_log_product
    return loss
