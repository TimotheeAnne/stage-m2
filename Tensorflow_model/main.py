from model import *
from evaluation import Evaluator
import gym
import gym_flowers 
from tqdm import tqdm
import datetime
import os 
import numpy as np
import argparse

# args
parser = argparse.ArgumentParser()
parser.add_argument('-gpu', type=str, default=False)
parser.add_argument('-type', type=str, default="Random")
args = parser.parse_args()


if not int(args.gpu) :
    os.environ["CUDA_VISIBLE_DEVICES"] = ''

OBS_DIM = 18
ACS_DIM = 4
OUTPUT_DIM = 22
EPOCH = 50
STEP = 10
N_EXPLORATIONS = None
N_POPULATION = None 
N_SAMPLES = 500
N_ITERATIONS = 10
REG = 0.000
training_data = None
eval_data = None
episodic_exploration = False
B = 20
objects = [0,1,2,3,4]
GRBF = False



if args.type == "Random":
    print("Random")
    GRBF = False

elif args.type == "GRBF":
    print("GRBF")
    GRBF = True

else :
    print("IMGEP")
    training_data = "/home/tim/Documents/stage-m2/ArmToolsToys_IMGEP/arm_run_saves/"+args.type+"/"


eval_data = "/home/tim/Documents/stage-m2/Tensorflow_model/data/ArmToolsToyR_eval.pk"


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

Observations = [[] for _ in range(DE.B)]

for iteration in tqdm(range(N_ITERATIONS)):
    """ Generate episodes """
    if training_data is None:
        for b in range(DE.B):
            for _ in range(N_SAMPLES):
                observation = [env.reset()]
                actions, _, _ = DE.select_actions( observation[0], 1, 1, GRBF=GRBF, exploration=False)
                actions = actions[0]
                """ Perform the action sequence """
                
                for t in range(50):
                    obs, _, _, _ = env.step(actions[t])
                    observation.append( obs.copy())
                    
                """ Add the collected data to the replay buffer """
                DE.add_episode(observation,  actions, b)
                Observations[b].append(observation[-1][:18])
            # ~ DE.replay_buffers[b].pretty_print()

    elif 'armtolstoy' in training_data:
        with open(training_data + 'imgep_' + str(iteration) +'.pk', 'br') as f:
            [observations, actions] = pickle.load(f)
            observations, actions = np.array(observations), np.array(actions)
            indexes = np.array(range(len(observations)))
            np.random.shuffle(indexes)
            split_indexes = np.split(indexes, DE.B)
            for b in range(DE.B):
                DE.add_episodes(observations[split_indexes[b]], actions[split_indexes[b]], b)
                # ~ DE.replay_buffers[b].pretty_print()
                Observations[b] = observations[split_indexes[b]][:,-1,:18]
            
            
    """ Training the network """
    for i in range(EPOCH//STEP):
        DE.train(STEP, verbose=False, validation=True, sampling='sample')

    DE.plot_training()

    """ Evaluate """
    for b in range(DE.B):
        with open(logdir+"/final_observations_r"+str(b)+"_"+str(iteration)+".pk", 'bw') as f:
            pickle.dump(np.array(Observations[b]), f)
    evaluator.eval(DE)

DE.save_ensemble()
