#!/usr/bin/env python3
from spinup.algos.pytorch.ddpg.ddpg import ddpg
from gym_space.envs.hover_1d import Hover1DContinuousEnv
from spinup.utils.run_utils import setup_logger_kwargs
import neptune.new as neptune
import hashlib
import json
import multiprocessing
print()


def dict_hash(dictionary: dict) -> str:
    """MD5 hash of a dictionary."""
    dhash = hashlib.md5()
    # We need to sort arguments so {'a': 1, 'b': 2} is
    # the same as {'b': 2, 'a': 1}
    encoded = json.dumps(dictionary, sort_keys=True).encode()
    dhash.update(encoded)
    return dhash.hexdigest()


def make_experiment_hash(model_hyperparams, env_params):
    model_hyperparams = model_hyperparams.copy()
    if "logger_kwargs" in model_hyperparams:
        del model_hyperparams["logger_kwargs"]
    del model_hyperparams["seed"]
    del model_hyperparams["save_freq"]
    model_ac_kwargs = model_hyperparams["ac_kwargs"]
    del model_hyperparams["ac_kwargs"]
    model_hyperparams.update({f"AC_KWARGS_{k}": v for k, v in model_ac_kwargs.items()})
    env_params = {f"ENV_{k}": v for k, v in env_params.items()}
    experiment_dict = dict(**model_hyperparams, **env_params)
    return dict_hash(experiment_dict)


def run_experiment(conf: dict):
    run = neptune.init(project="kajetan.janiak/hover1d-ddpg")
    env_params = dict(
        planet_radius=10.0,
        planet_mass=5e7,
        ship_mass=0.1,
        ship_engine_force=7e-6,
        step_size=conf["step_size"],
        max_episode_steps=300,
        reward_max_height=3.0,
        reward_partitions=1,
    )
    num_layers, layer_size = conf["net_shape"]
    model_hyperparams = dict(
        ac_kwargs=dict(hidden_sizes=[layer_size] * num_layers),
        seed=conf["seed"],
        steps_per_epoch=4000,
        epochs=conf["epochs"],
        replay_size=int(1e6),
        gamma=0.99,
        polyak=0.995,
        pi_lr=1e-3,
        q_lr=1e-3,
        batch_size=100,
        start_steps=10000,
        update_after=1000,
        update_every=50,
        act_noise=conf["action_noise"],
        num_test_episodes=10,
        max_ep_len=1000,
        save_freq=5,
    )
    experiment_hash = make_experiment_hash(model_hyperparams, env_params)
    logger_kwargs = setup_logger_kwargs(f"hover1d-ddpg-{experiment_hash}", conf["seed"])
    logger_kwargs["neptune_run"] = run
    model_hyperparams["logger_kwargs"] = logger_kwargs
    run["env/params"] = env_params
    run["model/hyperparams"] = model_hyperparams
    run["experiment_hash"] = experiment_hash
    ddpg(lambda: Hover1DContinuousEnv(**env_params), **model_hyperparams)
    run.stop()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    cpu_count = multiprocessing.cpu_count()
    parser.add_argument('-c', '--cores', type=int, default=int(0.8 * cpu_count))
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args()
    cores = min(args.cores, cpu_count)
    print(f"{cores=}")

    NET_SHAPES = [(1, 3), (1, 6), (2, 2), (2, 4), (2, 6)]
    STEP_SIZES = [5, 10, 15, 20]
    EPOCHS = 100
    ACTION_NOISES = [0.05, 0.1, 0.2]
    SEEDS = tuple(range(10))

    configs = []
    for net_shape in NET_SHAPES:
        for step_size in STEP_SIZES:
            for action_noise in ACTION_NOISES:
                for seed in SEEDS:
                    configs.append(
                        dict(
                            net_shape=net_shape,
                            step_size=step_size,
                            action_noise=action_noise,
                            seed=seed,
                            epochs=EPOCHS,
                        )
                    )
    print(f"{len(configs)=}")
    if not args.dry_run:
        with multiprocessing.Pool(cores) as pool:
            pool.map(run_experiment, configs)