import tensorflow as tf
import numpy as np
import pickle 
import datetime 
import matplotlib.pyplot as plt 
import os 
from tqdm import tqdm

FIGSIZE = (15,9)

training_data =  "./data/transition_multi25_0noise_train.pk"
eval_data = "./data/transition_multi25_0noise_eval.pk"

# ~ eval_data ="./data/transition_multiTask_1_eval.pk"
# ~ training_data = "./data/transition_multiTask_1_train.pk"

timestamp = datetime.datetime.now()
logdir = './log/'+str(timestamp)
os.makedirs(logdir)


class Normalization:
    def __init__(self):
        self.inputs_mean = None
        self.inputs_std = None
        self.outputs_mean = None
        self.outputs_std = None
        
    def init(self,inputs,outputs):
        self.inputs_mean = np.mean(inputs,axis=0)
        self.inputs_std = np.std(inputs,axis=0)
        self.outputs_mean = np.mean(outputs,axis=0)
        self.outputs_std = np.std(outputs,axis=0)
        
    def save(self):
        with open(os.path.join(logdir, "norm.pk"), "bw") as f:
            data = [self.inputs_mean,self.inputs_std,self.outputs_mean, self.outputs_std]
            pickle.dump(data,f)
        
    def normalize_inputs(self,x):
        return (x - self.inputs_mean) / self.inputs_std

    def normalize_outputs(self,x):
        return (x - self.outputs_mean) / self.outputs_std
        
    def denormalize_outputs(self,y):
        return (y * self.outputs_std) + self.outputs_mean
        
    def pretty_print(self):
        print( "in mean", self.inputs_mean)
        print( "in std", self.inputs_std)
        print( "out mean", self.outputs_mean)
        print( "out std", self.outputs_std)

REG = 0.0005
INPUT_DIM = 29
EPOCH = 50

norm = Normalization()



def plot_MSE( data):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    plt.plot( data['val_mean_squared_error'], label="validation")
    plt.plot( data['mean_squared_error'], label="training")
    plt.legend()
    plt.yscale('log')
    plt.xlabel('epochs')
    plt.ylabel('MSE')
    with open(os.path.join(logdir, "MSE.png"), "bw") as f:
        fig.savefig(f)
    plt.close(fig)


def compute_samples(data,norm):
    f = open(data , 'rb')
    [true_traj,Acs] = pickle.load(f)
    f.close()
    Inputs = []
    Targets = []
    for j in range(len(true_traj)):
        for t in range(50):
            inputs = np.concatenate((true_traj[j][t],Acs[j][t]))
            targets = true_traj[j][t+1]- true_traj[j][t]
            Inputs.append(inputs)
            Targets.append(targets)
    if norm.inputs_mean is None:
        norm.init(Inputs,Targets)
    return (norm.normalize_inputs(np.array(Inputs)),norm.normalize_outputs(np.array(Targets)))


def evaluation(model, norm, data="eval"):
    print("*** Prediction evaluation: "+data)
    if data == "eval":
        f = open(eval_data , 'rb')
    else:
        f = open(training_data , 'rb')
    [true_traj,Acs] = pickle.load(f)
    f.close()

    pred = []
    error_traj = []
    print("Trajectory")
    for j in tqdm(range(len(true_traj))):
        obs = [true_traj[j][0]]
        for t in range(50):
            inputs = norm.normalize_inputs(np.concatenate((obs[-1],Acs[j][t])))
            inputs = np.array([inputs])
            obs.append(norm.denormalize_outputs(model.predict(inputs)[0])+obs[-1])
        pred.append(obs)
        error_traj.append( np.linalg.norm(obs-np.array(true_traj[j])))
    true_traj = np.array(true_traj)
    pred_traj = np.array(pred)
    
    print("Transition")
    pred = []
    error_trans = []
    for j in tqdm(range(len(true_traj))):
        obs = [true_traj[j][0]]
        for t in range(50):
            inputs = np.array([norm.normalize_inputs(np.concatenate((true_traj[j][t],Acs[j][t])))])
            obs.append(norm.denormalize_outputs(model.predict(inputs)[0])+true_traj[j][t])
        pred.append(obs)
        error_trans.append( np.linalg.norm(obs-np.array(true_traj[j])))
    pred_trans = np.array(pred)
    

    filename = logdir+"/"+data+".pk"
    f = open(filename,'bw')
    pickle.dump((pred_traj,pred_trans,true_traj,error_traj,error_trans),f)
    f.close()


(x_train, y_train) = compute_samples( training_data, norm)
(x_eval, y_eval) = compute_samples( eval_data, norm)


model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation=tf.nn.relu, input_shape=[INPUT_DIM], 
                bias_initializer = tf.constant_initializer(value=0.),
                kernel_initializer = tf.contrib.layers.xavier_initializer(),
                kernel_regularizer = tf.keras.regularizers.l2(l=REG)),
    tf.keras.layers.Dense(256, activation=tf.nn.relu,
                bias_initializer = tf.constant_initializer(value=0.),
                kernel_initializer = tf.contrib.layers.xavier_initializer(),
                kernel_regularizer = tf.keras.regularizers.l2(l=REG)),
    tf.keras.layers.Dense(256, activation=tf.nn.relu,
                bias_initializer = tf.constant_initializer(value=0.),
                kernel_initializer = tf.contrib.layers.xavier_initializer(),
                kernel_regularizer = tf.keras.regularizers.l2(l=REG)),
    tf.keras.layers.Dense(25 , activation=None),
])


model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mean_squared_error']
              )

history = model.fit(x_train, y_train,
                    epochs=EPOCH,
                    validation_data=(x_eval, y_eval),
                    shuffle=True
                    )

""" Saving """
model.save_weights(logdir+'/model.h5')
with open(os.path.join(logdir, "config.txt"), "w") as f:
    def write(x):
        f.write(x+"\n")
    model.summary(print_fn=write)
norm.save()

""" Evaluation """
data = history.history
plot_MSE(data)
evaluation(model,norm, "eval")
evaluation(model,norm, "training")
