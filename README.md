# Space-Gym
Set of RL environments with locomotion tasks in space


# Installation
`pip install -e .`, then see example in `keyboard_agent.py`

# Environments

### GoalEnv
Navigate spaceship to achieve subsequent goal positions while avoiding
the planets and leaving the world (window) boundaries.

Parameters:

 - `n_planets` - number of planets to avoid
 - `survival_reward_scale` - fraction of reward for staying alive (not crashing)
 - `goal_vel_reward_scale` - fraction of reward for velocity toward current goal
 - `safety_reward_scale` - fraction of reward for not flying fast toward close obstacles
 - `goal_sparse_reward` - reward for achieving a goal
 - `renderer_kwargs` - additional parameters for renderer, see below

# Implementation

### Environments

There are six non-abstract environment classes.
Three with discrete and three with continuous action spaces,
defined in `envs/do_not_crash.py`, `envs/kepler.py`, `envs/goal.py`.  

All of them inherit from abstract base class `SpaceshipEnv`
defined in `envs/spaceship_env.py`.
This class takes care of physics, actions, collisions etc.
Child classes have to instantiate base class with selected
parameters values and implement `_reset` and `_reward` methods.

### Physical dynamic

All code responsible for physics simulation is in `dynamic_model.py`.
Please refer to the docstrings in that file. 

### Rendering

Class `Renderer` in `rendering.py` is responsible for visualization
of the environment. The class won't be instantiated and nothing will be drawn
unless `render()` method of an environment is called.  

Parameters `num_prev_pos_vis` and `prev_pos_color_decay` allow you to control
how the tail trailing the ship position looks and how long it is.

### GoalEnv initial position sampling

In order to make initial position sampling for large number of planets efficient,
we implemented an algorithm based on hexagonal tiling of a plane.
Related code is in `hexagonal_tiling.py`. To make sense of it, please refer to `notebooks/hexagonal_tiling.ipynb`.


# Hyperparameters optimization

Create virtual env

```shell
python3.8 -m venv gym_space_venv
source gym_space_venv/bin/activate
````

Clone and install gym-space and rl-baselines3-zoo
```shell
git clone git@github.com:MIMUW-RL/rl-baselines3-zoo.git
cd rl-baselines3-zoo
git checkout gym-space
pip install -r requirements.txt
cd ..
git clone git@github.com:MIMUW-RL/gym-space.git
pip install -e gym-space
```

### On rl machine:
In screen session:
```shell
for i in {1..3}; do CUDA_VISIBLE_DEVICES=<gpu_nr> python train.py --algo sac --env gym_space:GoalContinuous-v0 --env-kwargs k1:v1 k2:v2 ... -optimize -n <n_timesteps> --n-trials <n_trials> --study-name <study_name> --storage postgresql://hyperopt_example:hyperopt_example@localhost/hyperopt_example & done
```
where `gpu_nr` is 0 or 1.

### On Entropy cluster
Create `job<x>.sh` file containing
```shell
#!/bin/bash
#
#SBATCH --job-name=<study_name><x>
#SBATCH --output=/home/<username>/<study_name><x>.out
#SBATCH --error=/home/<username>/<study_name><x>.err
#SBATCH --partition=common
#SBATCH --qos=<qos>
#SBATCH --gres=gpu:1


source /home/<username>/gym_space_venv/bin/activate
cd /home/<username>/rl-baselines3-zoo
for i in {1..3}
do
   python -u train.py --algo sac --env gym_space:GoalContinuous-v0 -optimize -n <n_timesteps> --n-trials <n_trials> --study-name <study_name> --storage postgresql://hyperopt_example:hyperopt_example@rl/hyperopt_example &
done
wait
deactivate
```
My `qos` is `4gpu7d`, which means up to 4 GPUs and max job time of one week.
Thus for me `x` is 0-3.

Submit each job on slurm
```shell
sbatch job<x>.sh
```
