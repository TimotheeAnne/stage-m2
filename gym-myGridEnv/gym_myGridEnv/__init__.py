from gym.envs.registration import register

register(
    id='myGridEnv-v0',
    entry_point='gym_myGridEnv.envs:MyGridEnv',
)

register(
    id='myGridEnvModel-v0',
    entry_point='gym_myGridEnv.envs:MyGridEnvModel',
)
