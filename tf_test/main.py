from model import *
from evaluation import Evaluator
import gym
import gym_flowers 
from tqdm import tqdm
import datetime
import os 
import numpy as np

OBS_DIM = 18
ACS_DIM = 4
OUTPUT_DIM = 22
EPOCH = 20
N_EXPLORATIONS = 0
N_INIT_SAMPLES = 100
REG = 0.000
training_data = None
eval_data = None
B = 1

training_data = "/home/tim/Documents/stage-m2/tf_test/data/ArmToolsToy-v1_40000_train.pk"
eval_data = "/home/tim/Documents/stage-m2/tf_test/data/ArmToolsToy-v1_100_eval.pk"

timestamp = datetime.datetime.now()
logdir = './log/'+str(timestamp)
os.makedirs(logdir)

config = "" 
config += "EPOCH: "+str(EPOCH) +"\n"
config += "N_EXPLORATIONS: "+str(N_EXPLORATIONS)+"\n"
config += "N_INIT_SAMPLES: "+str(N_INIT_SAMPLES)+"\n"
config += "REG: "+str(REG)+"\n"
config += "B : "+str(B )+"\n"
config += "training_data: "+str(training_data)+"\n"
config += "eval_data : "+str(eval_data )+"\n"

with open(logdir+"/config.txt", 'w') as f:
    f.write(config)

DE = Ensemble(OBS_DIM, ACS_DIM, OUTPUT_DIM, reg=REG, B=B, init_samples = N_INIT_SAMPLES, logdir=logdir)
env = gym.make('ArmToolsToys-v1')

Observations = []
Actions = []

""" First training of the model """
if training_data is None:
    for _ in range(N_INIT_SAMPLES):
        observation = [env.reset()]
        actions = DE.select_actions( observation[0], 1, GRBF=False)
        
        """ Perform the action sequence """
        for t in range(50):
            obs, _, _, _ = env.step(actions[t])
            observation.append( obs)

        """ Add the collected data to the replay buffer """
        DE.add_episode( observation,  actions)
        Observations.append(observation)
        Actions.append(actions)
else:
    with open(training_data, 'br') as f:
        [observations, actions] = pickle.load(f)
        DE.add_episodes( observations, actions)
        Observations = observations.copy()
        Actions = actions.copy()

if not eval_data is None:
    with open(eval_data, 'br') as f:
        [observations, actions] = pickle.load(f)
        DE.add_validation( observations, actions)
        
DE.replay_buffer.pretty_print()

DE.train(EPOCH, verbose=True, validation=True, sampling='sample')

# ~ for iteration in tqdm(range(N_EXPLORATIONS)):
    
    # ~ """ Select an action sequence to perform """
    # ~ observation = [env.reset()]
    # ~ actions = DE.select_actions( observation[0], 100)
    
    # ~ """ Perform the action sequence """
    # ~ for t in range(50):
        # ~ env.render()
        # ~ obs, _, _, _ = env.step(actions[t])
        # ~ observation.append( obs)

    # ~ Observations.append( observation)
    # ~ Actions.append(actions)

    # ~ """ Add the collected data to the replay buffer """
    # ~ DE.add_episode( observation,  actions)
    
    # ~ """ Train the ensemble """ 
    # ~ DE.train()


evaluator = Evaluator( [Observations, Actions], eval_data, logdir, OBS_DIM )

evaluator.eval(DE)
