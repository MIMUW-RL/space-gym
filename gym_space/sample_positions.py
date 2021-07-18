import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
import itertools


class BodiesConfiguration(nn.Module):
    def __init__(self, initial_pos: np.ndarray, max_pos: float):
        super().__init__()
        assert initial_pos.shape[1] == 2
        self.max_pos = max_pos
        initial_pos = torch.from_numpy(initial_pos)
        initial_pos /= max_pos
        initial_pos = torch.atanh(initial_pos)
        assert torch.isfinite(initial_pos).all()
        # each from (-inf, +inf)
        self.unconstrained_pos = nn.parameter.Parameter(initial_pos)

    def forward(self):
        return torch.tanh(self.unconstrained_pos) * self.max_pos


def real_soft_min(x: torch.tensor, alpha: float):
    exp_alpha_x = torch.exp(alpha * x)
    x_exp_alpha_x = x * exp_alpha_x
    return torch.sum(x_exp_alpha_x) / torch.sum(exp_alpha_x)


def dist_loss(pos: torch.Tensor, softmin_alpha: float):
    pairs = tuple(itertools.combinations(range(pos.shape[0]), 2))
    pairwise_dist = torch.empty(len(pairs))
    for pair_nr, (i, j) in enumerate(pairs):
        pairwise_dist[pair_nr] = torch.linalg.norm(pos[i] - pos[j])
    gain = real_soft_min(pairwise_dist, softmin_alpha)
    return -gain

def maximize_dist(initial_pos: np.ndarray, max_pos: float, tol: float = 1e-4, softmin_alpha: float = -50.0):
    bodies_conf = BodiesConfiguration(initial_pos, max_pos)
    opt = Adam(bodies_conf.parameters())
    loss = float('inf')
    prev_loss = -loss
    while abs(loss - prev_loss) > tol:
        opt.zero_grad()
        pos = bodies_conf.forward()
        prev_loss = loss
        loss = dist_loss(pos, softmin_alpha)
        print(loss)
        loss.backward()
        opt.step()
    return bodies_conf.forward().detach().numpy()

