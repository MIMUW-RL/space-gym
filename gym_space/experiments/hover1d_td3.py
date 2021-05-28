#!/usr/bin/env python3
from spinup.algos.pytorch.td3.td3 import td3
from gym_space.envs.hover_1d import Hover1DContinuousEnv
from spinup.utils.run_utils import setup_logger_kwargs
import neptune.new as neptune
import multiprocessing
import torch
# inefficient parallelization on AMD CPUs
torch.set_num_threads(1)
print()

from gym_space.experiments.utils import make_experiment_hash


def run_experiment(conf: dict):
    test_run_str = '-test' if args.test_run else ''
    run = neptune.init(project=f"kajetan.janiak/{EXPERIMENT_NAME}{test_run_str}")
    max_episode_steps = 300
    env_params = dict(
        planet_radius=10.0,
        planet_mass=5e7,
        ship_mass=0.1,
        ship_engine_force=conf['ship_engine_force'],
        step_size=conf["step_size"],
        max_episode_steps=max_episode_steps,
        reward_max_height=3.0,
        reward_partitions=1,
    )
    num_layers, layer_size = conf["net_shape"]
    model_hyperparams = dict(
        ac_kwargs=dict(hidden_sizes=[layer_size] * num_layers),
        seed=conf["seed"],
        steps_per_epoch=4000,
        epochs=conf["epochs"],
        replay_size=conf["replay_size"],
        gamma=0.99,
        polyak=0.995,
        pi_lr=1e-3,
        q_lr=1e-3,
        batch_size=100,
        start_steps=10000,
        update_after=1000,
        update_every=50,
        act_noise=conf["action_noise"],
        target_noise=conf["target_noise"],
        noise_clip=0.5,
        policy_delay=2,
        num_test_episodes=10,
        max_ep_len=max_episode_steps,
        save_freq=5
    )
    experiment_hash = make_experiment_hash(model_hyperparams, env_params)
    logger_kwargs = setup_logger_kwargs(f"{EXPERIMENT_NAME}-{experiment_hash}", conf["seed"])
    logger_kwargs["neptune_run"] = run
    model_hyperparams["logger_kwargs"] = logger_kwargs
    run["env/params"] = env_params
    run["model/hyperparams"] = model_hyperparams
    run["experiment_hash"] = experiment_hash
    td3(lambda: Hover1DContinuousEnv(**env_params), **model_hyperparams)
    run.stop()

EXPERIMENT_NAME = "hover1d-td3"


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    cpu_count = multiprocessing.cpu_count()
    parser.add_argument('-c', '--cores', type=int, default=int(0.8 * cpu_count))
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--test_run', action='store_true')
    args = parser.parse_args()
    cores = min(args.cores, cpu_count)
    print(f"{cores=}")

    NET_SHAPES = [(2, 6)]
    STEP_SIZES = [15]
    EPOCHS = 100
    ACTION_NOISES = [0.1]
    REPLAY_SIZES = [100_000, 200_000, 400_000]
    SHIP_ENGINE_FORCES = [3e-6, 4.5e-6, 6e-6]
    TARGET_NOISES = [0.2]
    SEEDS = tuple(range(10))

    configs = []
    for seed in SEEDS:
        for net_shape in NET_SHAPES:
            for step_size in STEP_SIZES:
                for action_noise in ACTION_NOISES:
                    for target_noise in TARGET_NOISES:
                        for replay_size in REPLAY_SIZES:
                            for ship_engine_force in SHIP_ENGINE_FORCES:
                                configs.append(
                                    dict(
                                        net_shape=net_shape,
                                        step_size=step_size,
                                        action_noise=action_noise,
                                        target_noise=target_noise,
                                        seed=seed,
                                        epochs=EPOCHS,
                                        replay_size=replay_size,
                                        ship_engine_force=ship_engine_force
                                    )
                                )
    print(f"{len(configs)=}")
    if not args.dry_run:
        with multiprocessing.Pool(cores) as pool:
            pool.map(run_experiment, configs)