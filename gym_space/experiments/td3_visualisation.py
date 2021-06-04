import os
import torch
import numpy as np
import neptune.new as neptune
from spinup.algos.pytorch.td3.core import MLPActorCritic, MLPQFunction, MLPActor


def load_ac(run: neptune.Run, epoch: int) -> MLPActorCritic:
    tmp_model_dest = '/tmp/model.pt'
    run[f'model/model{epoch}.pt'].download(destination=tmp_model_dest)
    ac = torch.load(tmp_model_dest)
    os.unlink(tmp_model_dest)
    return ac


def load_obs_buf(run: neptune.Run) -> np.array:
    tmp_obs_buf_dest = '/tmp/obs_buf.npy'
    run[f'model/obs_buf'].download(destination=tmp_obs_buf_dest)
    obs_buf = np.load(tmp_obs_buf_dest)
    os.unlink(tmp_obs_buf_dest)
    return obs_buf


def get_observations_space(obs_buf: np.array, pos_density: int = 50, vel_density: int = 50) -> torch.Tensor:
    min_pos, min_vel = np.min(obs_buf, axis=0)
    max_pos, max_vel = np.max(obs_buf, axis=0)
    pos_linspace = torch.linspace(min_pos, max_pos, pos_density)
    vel_linspace = torch.linspace(min_vel, max_vel, vel_density)
    grid_pos, grid_vel = torch.meshgrid(pos_linspace, vel_linspace)
    return torch.stack([grid_pos, grid_vel], dim=2)


def q_on_const_action(q: MLPQFunction, obs_space: torch.Tensor, action: float):
    # [0, 1] -> [-1, 1]
    raw_action = 2 * action - 1
    action_tensor = torch.tensor(raw_action).expand(*obs_space.shape[:2], 1)
    return q(obs_space, action_tensor)


def q_on_policy(q: MLPQFunction, pi: MLPActor, obs_space: torch.Tensor):
    action_tensor = pi(obs_space)
    return q(obs_space, action_tensor)


def policy(pi: MLPActor, obs_space: torch.Tensor) -> torch.Tensor:
    # [-1, 1] -> [0, 1]
    return (pi(obs_space) + 1) / 2