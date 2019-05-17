from model import *
from evaluation import Evaluator
import gym
import gym_flowers 
from tqdm import tqdm
import datetime
import os 
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ''

OBS_DIM = 18
ACS_DIM = 4
OUTPUT_DIM = 22
EPOCH = 5
STEP = 5
N_EXPLORATIONS = None
N_POPULATION = None 
N_SAMPLES = 10000
N_ITERATIONS = 1
REG = 0.000
training_data = None
eval_data = None
episodic_exploration = False
B = 5
objects = [0,1,2,3,4]
GRBF = False

# ~ training_data = "/home/tim/Documents/stage-m2/tf_test/data/ArmToolsToy-v1_4000_train.pk"
eval_data = "/home/tim/Documents/stage-m2/tf_test/data/ArmToolsToy_1000pertinent.pk"
training_data = "/home/tim/Documents/stage-m2/armtolstoy/arm_run_saves/memory_based0/"

timestamp = datetime.datetime.now()
logdir = './log/'+str(timestamp)
os.makedirs(logdir)

config = "" 
config += "EPOCH: "+str(EPOCH) +"\n"
config += "N_EXPLORATIONS: "+str(N_EXPLORATIONS)+"\n"
config += "N_POPULATION: "+str(N_POPULATION)+"\n"
config += "N_SAMPLES: "+str(N_SAMPLES)+"\n"
config += "N_ITERATIONS: "+str(N_ITERATIONS)+"\n"
config += "STEP: "+str(STEP)+"\n"
config += "REG: "+str(REG)+"\n"
config += "B : "+str(B )+"\n"
config += "training_data: "+str(training_data)+"\n"
config += "eval_data: "+str(eval_data )+"\n"
config += "objects: "+ str(objects) + "\n"
config += "episodic_exploration: "+ str(episodic_exploration) + "\n"
config += "GRBF: "+ str(GRBF) + "\n"

with open(logdir+"/config.txt", 'w') as f:
    f.write(config)

DE = Ensemble(OBS_DIM, ACS_DIM, OUTPUT_DIM, reg=REG, B=B, logdir=logdir, objects=objects)
evaluator = Evaluator( None, eval_data, logdir, OBS_DIM )
env = gym.make('ArmToolsToys-v1')

Observations = []

for iteration in tqdm(range(N_ITERATIONS)):
    """ Generate episodes """
    if training_data is None:
        for _ in range(N_SAMPLES):
            observation = [env.reset()]
            actions = DE.select_actions( observation[0], 1, GRBF=GRBF, exploration=False)[0]
            """ Perform the action sequence """
            
            for t in range(50):
                # ~ env.render()
                obs, _, _, _ = env.step(actions[t])
                observation.append( obs.copy())
                
            """ Add the collected data to the replay buffer """
            DE.add_episode( observation,  actions)
            Observations.append(observation[-1][:18])

    elif 'armtolstoy' in training_data:
        with open(training_data + 'imgep_' + str(iteration) +'.pk', 'br') as f:
            [observations, actions] = pickle.load(f)
            DE.add_episodes(observations, actions)
            Observations = np.array(observations)[:,-1,:18]
    else:
        with open(training_data, 'br') as f:
            [observations, actions] = pickle.load(f)
            DE.add_episodes(observations, actions)
            Observations = observations.copy()


    if not eval_data is None:
        with open(eval_data, 'br') as f:
            [observations, actions] = pickle.load(f)
            DE.add_validation( observations, actions)
            
    DE.replay_buffer.pretty_print()

    """ Training the network """
    for i in range(EPOCH//STEP):
        DE.train(STEP, verbose=True, validation=True, sampling='choice')

    DE.plot_training()

    """ Evaluate """
    with open(logdir+"/final_observations_"+str(iteration)+".pk", 'bw') as f:
        pickle.dump(np.array(Observations), f)
    evaluator.eval(DE)

DE.save_ensemble()
