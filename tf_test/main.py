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
EPOCH = 50
STEP = 5
N_EXPLORATIONS = 10
N_POPULATION = 500
N_INIT_SAMPLES = 400
REG = 0.000
training_data = None
eval_data = None
episodic_exploration = True
B = 1
objects = [0,1,2,3,4]

training_data = "/home/tim/Documents/stage-m2/tf_test/data/ArmToolsToy-v1_4000_train.pk"
eval_data = "/home/tim/Documents/stage-m2/tf_test/data/ArmToolsToy_1000pertinent.pk"

timestamp = datetime.datetime.now()
logdir = './log/'+str(timestamp)
os.makedirs(logdir)

config = "" 
config += "EPOCH: "+str(EPOCH) +"\n"
config += "N_EXPLORATIONS: "+str(N_EXPLORATIONS)+"\n"
config += "N_POPULATION: "+str(N_POPULATION)+"\n"
config += "N_INIT_SAMPLES: "+str(N_INIT_SAMPLES)+"\n"
config += "STEP: "+str(STEP)+"\n"
config += "REG: "+str(REG)+"\n"
config += "B : "+str(B )+"\n"
config += "training_data: "+str(training_data)+"\n"
config += "eval_data: "+str(eval_data )+"\n"
config += "objects: "+ str(objects) + "\n"
config += "episodic_exploration: "+ str(episodic_exploration) + "\n"

with open(logdir+"/config.txt", 'w') as f:
    f.write(config)

DE = Ensemble(OBS_DIM, ACS_DIM, OUTPUT_DIM, reg=REG, B=B, init_samples = N_INIT_SAMPLES, logdir=logdir, objects=objects)
evaluator = Evaluator( None, eval_data, logdir, OBS_DIM )
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
        DE.add_episodes(observations, actions, N_INIT_SAMPLES)
        Observations = observations.copy()
        Actions = actions.copy()

if not eval_data is None:
    with open(eval_data, 'br') as f:
        [observations, actions] = pickle.load(f)
        DE.add_validation( observations, actions)
        
DE.replay_buffer.pretty_print()

for i in range(EPOCH//STEP):
    DE.train(STEP, verbose=True, validation=True, sampling='choice')
    evaluator.eval(DE)

# ~ if episodic_exploration:
    # ~ """ Episodic Exploration """
    # ~ for iteration in tqdm(range(N_EXPLORATIONS)):
        
        # ~ """ Select an action sequence to perform """
        # ~ Tmax = 50
        # ~ obs = env.reset()
        # ~ actions, uncertainty, pred = DE.select_actions(obs, N_POPULATION, Tmax=Tmax, GRBF=False)
        # ~ observations = [[obs] for _ in range(N_POPULATION)]
        
        # ~ """ Perform the action sequence """
        # ~ for j in range(N_POPULATION):
            # ~ obs = env.reset()
            # ~ for t in range(Tmax):
                #env.render()
                # ~ obs, _, _, _ = env.step(actions[j][t])
                # ~ observations[j] = np.concatenate( (observations[j], [obs]))

        # ~ DE.save_exploration(actions, uncertainty, observations, pred)
        
        #Observations.append(observations)
        #Actions.append(actions)

        # ~ """ Add the collected data to the replay buffer """
        # ~ DE.add_episodes(observations,  actions)
        
        # ~ DE.replay_buffer.pretty_print()
        # ~ """ Train the ensemble """ 
        # ~ for i in range(EPOCH//STEP):
            # ~ DE.train(STEP, verbose=True, validation=False, sampling='choice')
# ~ else:
    # ~ """ Step based exploration """
    # ~ for iteration in tqdm(range(N_EXPLORATIONS)):
        
        # ~ """ Select an action sequence to perform """

        # ~ obs = env.reset()
        # ~ observations = [obs]
        # ~ actions = []
        # ~ for t in range(50):
            # ~ action, uncertainty, pred = DE.select_actions(obs, N_POPULATION, Tmax=Tmax, GRBF=False)
            # ~ actions.append(action)
            
        
            # ~ """ Perform the action  """
            #env.render()
            # ~ obs, _, _, _ = env.step(action)
            # ~ observations.append(obs)

        # ~ DE.save_exploration(action, uncertainty, observations, pred)
        
        #Observations.append(observations)
        #Actions.append(actions)

        # ~ """ Add the collected data to the replay buffer """
        # ~ DE.add_episode(observations,  actions)
        
        # ~ DE.replay_buffer.pretty_print()
        # ~ """ Train the ensemble """ 
        # ~ for i in range(EPOCH//STEP):
            # ~ DE.train(STEP, verbose=True, validation=False, sampling='choice')


# ~ DE.plot_training()


# ~ evaluator = Evaluator( [Observations, Actions], eval_data, logdir, OBS_DIM )


