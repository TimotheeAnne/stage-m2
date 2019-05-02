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
EPOCH = 100
N_ITERATIONS = 1000
eval_data = "/home/tim/Documents/stage-m2/tf_test/data/ArmToolsToy_onlypertinentfromrandom_eval.pk"


DE = Ensemble(OBS_DIM, ACS_DIM, OUTPUT_DIM, reg=0)

timestamp = datetime.datetime.now()
logdir = './log/'+str(timestamp)
os.makedirs(logdir)


env = gym.make('ArmToolsToys-v1')

Observations = []
Actions = []

for iteration in tqdm(range(N_ITERATIONS)):
    # ~ print("Iteration ", iteration)
    
    """ Select an action sequence to perform """
    observation = [env.reset()]
    actions = DE.select_actions( observation[0], 100)
    
    """ Perform the action sequence """
    for t in range(50):
        env.render()
        obs, _, _, _ = env.step(actions[t])
        observation.append( obs)

    Observations.append( observation)
    Actions.append(actions)

    """ Add the collected data to the replay buffer """
    DE.add_data( observation,  actions)
    
    """ Train the ensemble """ 
    DE.train()
    
DE.predict_trajectory(Observations, Actions)
evaluator = Evaluator( [Observations, Actions], eval_data, logdir, OBS_DIM )

evaluator.eval(DE)
