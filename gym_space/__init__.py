from gym.envs.registration import register

register(
    id='Rocket1D-v0',
    entry_point='gym_space.envs:Rocket1D',
)

register(
    id='Rocket2D2DoF-v0',
    entry_point='gym_space.envs:Rocket2D2DoF',
)

register(
    id='Rocket2D3DoF-v0',
    entry_point='gym_space.envs:Rocket2D3DoF',
)