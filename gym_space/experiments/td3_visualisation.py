import os
import torch
import numpy as np
import neptune.new as neptune
from spinup.algos.pytorch.td3.core import MLPActorCritic, MLPQFunction, MLPActor

import matplotlib.pyplot as plt


class TD3ExperimentResults:
    def __init__(
        self,
        experiment_nr: int,
        pos_density: int = 50,
        vel_density: int = 50,
        do_load_obs_buf: bool = True,
        from_test_run: bool = False
    ):
        project_suffix = "-test" if from_test_run else ""
        run_suffix = "T" if from_test_run else ""
        neptune_run = neptune.init(
            f"kajetan.janiak/hover1d-td3{project_suffix}", run=f"H1TD3{run_suffix}-{experiment_nr}", mode="read-only"
        )
        self.planned_epochs = int(neptune_run["model/hyperparams/epochs"].fetch())
        self.completed_epochs = int(neptune_run["Epoch"].fetch_last())
        self.finished = self.planned_epochs == self.completed_epochs
        self.model_save_freq = int(neptune_run["model/hyperparams/save_freq"].fetch())
        self.acs = dict()
        self._load_acs(neptune_run)
        self.obs_buf = None
        if self.finished and do_load_obs_buf:
            self.obs_buf = load_obs_buf(neptune_run)
            print("obs buf loaded")
        self.planet_radius = neptune_run["env/params/planet_radius"].fetch()
        self.reward_max_height = neptune_run["env/params/reward_max_height"].fetch()
        self.obs_space = None
        self.pos_lin_density = self.pos_log_density = self.max_pos_linear = None
        self.generate_obs_space(pos_density, vel_density)
        print("obs space generated")
        neptune_run.stop()

    def _load_acs(self, neptune_run: neptune.Run):
        if self.completed_epochs < 20:
            epochs_nrs = range(0, self.completed_epochs, self.model_save_freq)
        else:
            initial_freq = self.model_save_freq
            initial_epochs = self.completed_epochs // 10
            epochs_nrs = list(range(0, initial_epochs, initial_freq))
            freq = max(self.model_save_freq, self.completed_epochs // 20)
            epochs_nrs += list(range(initial_epochs, self.completed_epochs, freq))

        n_of_epochs = len(epochs_nrs)
        print(f"loading {n_of_epochs} models from neptune...")
        for i, epoch in enumerate(epochs_nrs, start=1):
            self.acs[epoch] = load_ac(neptune_run, epoch)
            if (i % (n_of_epochs // 5)) == 0:
                print(f"{i} models loaded")
        print("done")

    def generate_obs_space(self, pos_density: int = 50, vel_density: int = 50):
        max_pos_linear = self.planet_radius + 1.5 * self.reward_max_height
        (
            self.obs_space,
            self.pos_lin_density,
            self.pos_log_density,
            self.max_pos_linear,
        ) = get_observations_space(
            self.obs_buf, max_pos_linear, pos_density, vel_density
        )

    def plot_values_over_obs_space(self, ax, values: np.array, cmap="Greys", v_min_max = None):
        min_pos, max_pos = self.obs_space[[0, -1], 0, 0]
        min_vel, max_vel = self.obs_space[0, [0, -1], 1]
        imshow_kwargs = dict(
            origin="lower",
            # -1, 1 doesn't matter, we override it below
            extent=(min_vel, max_vel, -0.5, self.obs_space.shape[0] - 0.5),
            aspect="auto",
            cmap=cmap
        )
        if v_min_max is not None:
            v_min, v_max = v_min_max
            imshow_kwargs["vmin"] = v_min
            imshow_kwargs["vmax"] = v_max
        with torch.no_grad():
            image = ax.imshow(
                values,
                **imshow_kwargs
            )
            lin_ytick_labels = np.arange(
                0, self.max_pos_linear - self.planet_radius, dtype=int
            )
            lin_yticks_max = (
                self.pos_lin_density
                * (lin_ytick_labels[-1] - lin_ytick_labels[0])
                / (self.max_pos_linear - self.planet_radius)
            )
            lin_yticks = np.linspace(0, lin_yticks_max, len(lin_ytick_labels)) - 0.5
            # log_ytick_labels = np.array([max_pos - self.planet_radius], dtype=int)
            # log_yticks = np.array([self.obs_space.shape[0] - 0.5])
            ytick_labels = np.concatenate([lin_ytick_labels, ])
            yticks = np.concatenate([lin_yticks, ])
            ax.set_yticks(yticks)
            ax.set_yticklabels(ytick_labels)
        # plt.colorbar()
        ax.set_xlabel("velocity")
        ax.set_ylabel("position")
        return image

    def plot_q1_on_const_action(self, ax, action: float, epoch: int):
        return self.plot_q_on_const_action(ax, 1, action, epoch)

    def plot_q2_on_const_action(self, ax, action: float, epoch: int):
        return self.plot_q_on_const_action(ax, 2, action, epoch)

    def plot_q_on_const_action(self, ax, q_nr: int, action: float, epoch: int):
        assert epoch in self.acs, f"Model from epoch {epoch} is not available"
        q_vals = q_on_const_action(
            getattr(self.acs[epoch], f"q{q_nr}"), self.obs_space, action
        )
        return self.plot_values_over_obs_space(ax, q_vals.detach().numpy())

    def plot_q1_full_minus_no_thrust(self, ax, epoch: int):
        return self.plot_q_full_minus_no_thrust(ax, 1, epoch)

    def plot_q2_full_minus_no_thrust(self, ax, epoch: int):
        return self.plot_q_full_minus_no_thrust(ax, 2, epoch)

    def plot_q_full_minus_no_thrust(self, ax, q_nr: int, epoch: int):
        assert epoch in self.acs, f"Model from epoch {epoch} is not available"
        q = getattr(self.acs[epoch], f"q{q_nr}")
        q_full_thrust_vals = q_on_const_action(
            q, self.obs_space, action=1
        )
        q_no_thrust_vals = q_on_const_action(
            q, self.obs_space, action=0
        )
        q_full_minus_no_thrust_vals = q_full_thrust_vals - q_no_thrust_vals
        return self.plot_values_over_obs_space(ax, q_full_minus_no_thrust_vals.detach().numpy())

    def plot_q1_on_policy(self, ax, epoch: int):
        return self.plot_q_on_policy(ax, 1, epoch)

    def plot_q2_on_policy(self, ax, epoch: int):
        return self.plot_q_on_policy(ax, 2, epoch)

    def plot_q_on_policy(self, ax, q_nr: int, epoch: int):
        assert epoch in self.acs, f"Model from epoch {epoch} is not available"
        q_vals = q_on_policy(
            getattr(self.acs[epoch], f"q{q_nr}"), self.acs[epoch].pi, self.obs_space
        )
        return self.plot_values_over_obs_space(ax, q_vals.detach().numpy())

    def plot_policy(self, ax, epoch: int):
        assert epoch in self.acs, f"Model from epoch {epoch} is not available"
        policy_vals = policy(self.acs[epoch].pi, self.obs_space)
        return self.plot_values_over_obs_space(ax, policy_vals.detach().numpy(), cmap="inferno", v_min_max=(0, 1))


def load_ac(run: neptune.Run, epoch: int) -> MLPActorCritic:
    tmp_model_dest = "/tmp/model.pt"
    run[f"model/model{epoch}.pt"].download(destination=tmp_model_dest)
    ac = torch.load(tmp_model_dest)
    os.unlink(tmp_model_dest)
    return ac


def load_obs_buf(run: neptune.Run) -> np.array:
    tmp_obs_buf_dest = "/tmp/obs_buf.npy"
    print("loading observations from replay buffer")
    run[f"model/obs_buf"].download(destination=tmp_obs_buf_dest)
    obs_buf = np.load(tmp_obs_buf_dest)
    os.unlink(tmp_obs_buf_dest)
    return obs_buf


def get_observations_space(
    obs_buf: np.array,
    max_pos_linear: float,
    pos_density: int = 50,
    vel_density: int = 50,
) -> torch.Tensor:
    if obs_buf is None:
        min_pos, min_vel = 10.0, -0.015
        max_pos, max_vel = 14.0, 0.13
    else:
        min_pos, min_vel = np.min(obs_buf, axis=0)
        max_pos, max_vel = np.max(obs_buf, axis=0)
    max_pos_linear = min(max_pos_linear, max_pos)
    pos_log_density = pos_density // 3
    pos_lin_density = pos_density - pos_log_density
    pos_linspace = torch.linspace(min_pos, max_pos_linear, pos_lin_density)
    # pos_logspace = torch.logspace(
    #     np.log10(max_pos_linear), np.log10(max_pos), pos_log_density + 1
    # )[1:]
    pos_space = torch.cat([pos_linspace, ])
    vel_space = torch.linspace(min_vel, max_vel, vel_density)
    grid_pos, grid_vel = torch.meshgrid(pos_space, vel_space)
    return (
        torch.stack([grid_pos, grid_vel], dim=2),
        pos_lin_density,
        pos_log_density,
        max_pos_linear,
    )


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
