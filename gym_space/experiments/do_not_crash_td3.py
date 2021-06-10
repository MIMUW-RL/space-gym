#!/usr/bin/env python3
from spinup.algos.pytorch.td3.td3 import td3
from gym_space.envs.do_not_crash import DoNotCrashContinuousEnv
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
        max_ep_len=DoNotCrashContinuousEnv.max_episode_steps,
        save_freq=SAVE_FREQ,
    )
    experiment_hash = make_experiment_hash(model_hyperparams)
    logger_kwargs = setup_logger_kwargs(
        f"{EXPERIMENT_NAME}-{experiment_hash}", conf["seed"]
    )
    logger_kwargs["neptune_run"] = run
    model_hyperparams["logger_kwargs"] = logger_kwargs
    run["model/hyperparams"] = model_hyperparams
    run["experiment_hash"] = experiment_hash
    td3(lambda: DoNotCrashContinuousEnv(), **model_hyperparams)
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
    EXPERIMENT_NAME = "donotcrash-td3"

    NET_SHAPES = [(2, 64), (2, 100), (2, 128)]
    ACTION_NOISES = [0.1]
    TARGET_NOISES = [0.2]
    START_STEPS = [10_000]
    UPDATE_AFTERS = [1_000]
    POLICY_DELAYS = [2]
    SEEDS = tuple(range(10))

    configs = []
    for seed in SEEDS:
        for net_shape in NET_SHAPES:
            for action_noise in ACTION_NOISES:
                for target_noise in TARGET_NOISES:
                    for start_steps in START_STEPS:
                        for update_after in UPDATE_AFTERS:
                            for policy_delay in POLICY_DELAYS:
                                configs.append(
                                    dict(
                                        net_shape=net_shape,
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
        if cores > 1:
            with multiprocessing.Pool(cores) as pool:
                pool.map(run_experiment, configs)
        else:
            for config in configs:
                run_experiment(config)
