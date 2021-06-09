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
    test_run_str = "-test" if args.test_run else ""
    run = neptune.init(project=f"kajetan.janiak/{EXPERIMENT_NAME}{test_run_str}")
    env_params = dict(
        planet_radius=10.0,
        planet_mass=5e7,
        ship_mass=0.1,
        ship_engine_force=6e-6,
        step_size=conf["step_size"],
        max_episode_steps=MAX_EPISODE_STEPS,
        reward_max_height=3.0,
        reward_partitions=1,
    )
    num_layers, layer_size = conf["net_shape"]
    model_hyperparams = dict(
        ac_kwargs=dict(hidden_sizes=[layer_size] * num_layers),
        seed=conf["seed"],
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=EPOCHS,
        replay_size=REPLAY_SIZE,
        gamma=0.99,
        polyak=0.995,
        pi_lr=1e-3,
        q_lr=1e-3,
        batch_size=100,
        start_steps=conf["start_steps"],
        update_after=conf["update_after"],
        update_every=50,
        act_noise=conf["action_noise"],
        target_noise=conf["target_noise"],
        noise_clip=0.5,
        policy_delay=conf["policy_delay"],
        num_test_episodes=10,
        max_ep_len=MAX_EPISODE_STEPS,
        save_freq=SAVE_FREQ,
    )
    experiment_hash = make_experiment_hash(model_hyperparams, env_params)
    logger_kwargs = setup_logger_kwargs(
        f"{EXPERIMENT_NAME}-{experiment_hash}", conf["seed"]
    )
    logger_kwargs["neptune_run"] = run
    model_hyperparams["logger_kwargs"] = logger_kwargs
    run["env/params"] = env_params
    run["model/hyperparams"] = model_hyperparams
    run["experiment_hash"] = experiment_hash
    td3(lambda: Hover1DContinuousEnv(**env_params), **model_hyperparams)
    run.stop()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    cpu_count = multiprocessing.cpu_count()
    parser.add_argument("-c", "--cores", type=int, default=int(0.8 * cpu_count))
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--test_run", action="store_true")
    args = parser.parse_args()
    cores = min(args.cores, cpu_count)
    print(f"{cores=}")

    EPOCHS = 250
    STEPS_PER_EPOCH = 4_000
    REPLAY_SIZE = STEPS_PER_EPOCH * EPOCHS
    SAVE_FREQ = 1
    MAX_EPISODE_STEPS = 300
    EXPERIMENT_NAME = "hover1d-td3"

    NET_SHAPES = [(2, 6), (2, 10), (2, 30), (2, 50), (2, 64)]
    STEP_SIZES = [15]
    ACTION_NOISES = [0.1]
    TARGET_NOISES = [0.2]
    START_STEPS = [30_000]
    UPDATE_AFTERS = [1_000]
    POLICY_DELAYS = [2]
    SEEDS = tuple(range(10))

    configs = []
    for seed in SEEDS:
        for net_shape in NET_SHAPES:
            for step_size in STEP_SIZES:
                for action_noise in ACTION_NOISES:
                    for target_noise in TARGET_NOISES:
                        for start_steps in START_STEPS:
                            for update_after in UPDATE_AFTERS:
                                for policy_delay in POLICY_DELAYS:
                                    configs.append(
                                        dict(
                                            net_shape=net_shape,
                                            step_size=step_size,
                                            action_noise=action_noise,
                                            target_noise=target_noise,
                                            seed=seed,
                                            start_steps=start_steps,
                                            update_after=update_after,
                                            policy_delay=policy_delay,
                                        )
                                    )

    print(f"{len(configs)=}")
    if not args.dry_run:
        with multiprocessing.Pool(cores) as pool:
            pool.map(run_experiment, configs)
