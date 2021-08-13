# gym-space
RL environments with locomotion tasks in space


# Installation
`pip install -e .`, then see example in `keyboard_agent.py`

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