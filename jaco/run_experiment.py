#!/usr/bin/env python
import multiprocessing as mp
import argparse
import copy
import yaml
import gym
import time
import neptune
from rltoolkit import EvalsWrapper, EvalsWrapperACM
from itertools import product
from gym_space.envs.orbit import OrbitEnv

ALGORITHMS = ['ddpg', 'sac', 'td3']


parser = argparse.ArgumentParser(description='Gym-Space experiment configuration')
parser.add_argument('algorithm', choices=ALGORITHMS)
parser.add_argument('env' )
parser.add_argument('-c', '--config', type=str, required=True)
parser.add_argument('--spp', nargs='?', const=True, default=False)
parser.add_argument('--n_runs', type=int, default=1)
parser.add_argument('--n_cores', type=int, default=1)
parser.add_argument('--tune', nargs='*', help='Config parameters to tune, '
                    'which have list of possible values to check')
parser.add_argument('--neptune_proj', type = str)                    
parser.add_argument('--neptune_token', type=str)
parser.add_argument('--log_dir', type=str, default='.')



def train(kwargs, args):

    if args.neptune_proj:
        import neptune
        neptune.init(args.neptune_proj, api_token=args.neptune_token)
        neptune.create_experiment(params=kwargs)

    try:
        if args.spp:
            kwargs['acm_fn'] = lambda in_dim, o_dim, lim, discr: BasicAcM(
                in_dim, o_dim, discr)
            EvalsWrapperACM(**kwargs).perform_evaluations()
        else:           
            EvalsWrapper(**kwargs).perform_evaluations()
        if args.neptune_proj:
            neptune.stop()
    except Exception as e:
        if args.neptune_proj:
            neptune.stop(str(e))
        raise e



def get_algorithm(name, spp):
    if spp:
        name = '{}-{}'.format('spp', name)
    if name == 'ddpg':
        from rltoolkit import DDPG
        return DDPG
    if name == 'spp-ddpg':
        from rltoolkit import DDPG_AcM
        return DDPG_AcM
    if name == 'sac':
        from rltoolkit import SAC
        return SAC
    if name == 'spp-sac':
        from rltoolkit import SAC_AcM
        return SAC_AcM
    if name == 'ppo':
        from rltoolkit import PPO
        return PPO
    if name == 'spp-ppo':
        from rltoolkit import PPO_AcM
        return PPO_AcM
    if name == 'td3':
        from rltoolkit import TD3
        return TD3
    if name == 'spp-td3':
        from rltoolkit.acm.off_policy.td3_acm import TD3_AcM
        return TD3_AcM
    raise AttributeError()


if __name__ == '__main__':
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    env = OrbitEnv()

    gym.envs.register(
        id='DoNotCrash-v0',
        entry_point='gym_space.envs.do_not_crash:DoNotCrashContinuousEnv',
        max_episode_steps=300,
    )

    gym.envs.register(
        id='Orbit-v0',
        entry_point='gym_space.envs.orbit:OrbitEnv',
        max_episode_steps=300,
    )

    gym.envs.register(
        id='GoalContinuous2-v0',
        entry_point='gym_space.envs.goal:GoalContinuousEnv',
        kwargs = {'n_planets' : 2},
    )

    gym.envs.register(
        id='Kepler-v0',
        entry_point='gym_space.envs.kepler:KeplerContinuousEnv',
    )
    

    config['env_name']  = args.env + '-v0'

    algorithm = get_algorithm(args.algorithm, args.spp)
    kwargs_list = []
    acm_args = None
    if args.spp:
        from rltoolkit.acm.models.basic_acm import BasicAcM
        env = gym.make(config['env_name'])
        acm_args = (
            2 * env.observation_space.shape[0], env.action_space.shape[0], False)
    args_to_tune = [] if args.tune is None else args.tune
    configs = []
    for i, h_params in enumerate(product(*[product([param], config[param]) for param in args_to_tune])):
        c = copy.deepcopy(config)
        for param, value in h_params:
            c[param] = value
        for run in range(args.n_runs):
            c['Algo'] = algorithm
            c['evals'] = 1
            c['tensorboard_dir'] = args.log_dir + "/{}{}-{}-{}-{}".format(
                'spp-' if args.spp else '', args.algorithm, c['env_name'], time.time(), i)
            c['log_dir'] = c['tensorboard_dir'] + '/logdir/'
            configs.append(c)
    if len(configs) == 1:
        train(configs[0], args)
    else:
        with mp.Pool(args.n_cores) as p:
            p.starmap(train, product(configs, [args]))