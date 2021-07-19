import torch
import torch.nn as nn
from torch.nn.functional import softplus, relu
from torch.optim import Adam
import numpy as np
import itertools


class BodiesConfiguration(nn.Module):
    def __init__(self, initial_pos: torch.Tensor, max_pos: float):
        super().__init__()
        assert initial_pos.shape[1] == 2
        self.max_pos = max_pos
        initial_pos /= max_pos
        initial_pos = torch.atanh(initial_pos)
        assert torch.isfinite(initial_pos).all()
        # each from (-inf, +inf)
        self.unconstrained_pos = nn.parameter.Parameter(initial_pos)

    def forward(self):
        return torch.tanh(self.unconstrained_pos) * self.max_pos


def real_soft_max(x: torch.tensor, alpha: float = 1.0):
    exp_alpha_x = torch.exp(alpha * x)
    x_exp_alpha_x = x * exp_alpha_x
    return torch.sum(x_exp_alpha_x) / torch.sum(exp_alpha_x)


def pairwise_dist(pos: torch.Tensor, radii: torch.Tensor) -> torch.Tensor:
    assert radii.shape[0] == pos.shape[0]
    pairs = tuple(itertools.combinations(range(pos.shape[0]), 2))
    pairwise_dist_ = torch.empty(len(pairs))
    for pair_nr, (i, j) in enumerate(pairs):
        pairwise_dist_[pair_nr] = (
            torch.linalg.norm(pos[i] - pos[j]) - radii[i] - radii[j]
        )
    return pairwise_dist_


def clearance_loss(pairwise_dist_: torch.Tensor):
    return torch.mean(softplus(-pairwise_dist_, beta=5.0))


def divergence_loss(pos: torch.Tensor, initial_pos: torch.Tensor):
    pos_diff = pos - initial_pos
    return torch.mean(torch.linalg.norm(pos_diff, dim=1))


def ship_goal_dist_loss(pos: torch.Tensor):
    return 1 / (torch.linalg.norm(pos[0] - pos[1]) + 0.1)


def dist_loss(
    clearance_loss_: torch.Tensor,
    divergence_loss_: torch.Tensor,
    ship_goal_dist_loss_: torch.Tensor,
    divergence_loss_scale: float = 0.1,
    ship_goal_dist_loss_scale: float = 0.1
):
    return clearance_loss_ + divergence_loss_scale * divergence_loss_ + ship_goal_dist_loss_scale * ship_goal_dist_loss_


def make_radii_tensor(radii: list[float], n: int):
    radii0 = torch.tensor(radii)
    radii1 = torch.full((n - len(radii),), radii[-1])
    return torch.cat([radii0, radii1])


def maximize_dist(
    initial_pos: np.ndarray,
    max_pos: float,
    radii: list[float],
    tol: float = 1e-5,
    max_steps: int = 5_000,
):
    radii_tensor = make_radii_tensor(radii, initial_pos.shape[0])
    initial_pos = torch.from_numpy(initial_pos)
    bodies_conf = BodiesConfiguration(initial_pos, max_pos)
    opt = Adam(bodies_conf.parameters())
    loss = np.inf
    for step_nr in range(max_steps):
        opt.zero_grad()
        pos = bodies_conf.forward()
        pairwise_dist_ = pairwise_dist(pos, radii_tensor)
        clearance_loss_ = clearance_loss(pairwise_dist_)
        divergence_loss_ = divergence_loss(pos, initial_pos)
        ship_goal_dist_loss_ = ship_goal_dist_loss(pos)
        prev_loss = loss
        loss = dist_loss(clearance_loss_, divergence_loss_, ship_goal_dist_loss_)
        if abs(loss - prev_loss) < tol and torch.all(pairwise_dist_ >= 0.0):
            break
        if step_nr % 100 == 0:
            print(loss.item())
        loss.backward()
        opt.step()
    print(step_nr)
    return bodies_conf.forward().detach().numpy()
