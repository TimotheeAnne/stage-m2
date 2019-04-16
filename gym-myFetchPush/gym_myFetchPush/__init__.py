from gym.envs.registration import register

register(
    id='myFetchPush-v0',
    entry_point='gym_myFetchPush.envs:MyFetchPush',
)

register(
    id='myMultiTaskFetchArmNLP-v0',
    entry_point='gym_myFetchPush.envs:MyMultiTaskFetchArmNLP',
)

register(
    id='myMultiTaskFetchArmNLP-v1',
    entry_point='gym_myFetchPush.envs:MyMultiTaskFetchArmNLP_v1',
)

