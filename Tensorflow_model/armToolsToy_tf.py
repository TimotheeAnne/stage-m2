import tensorflow as tf
import numpy as np
import pickle 
import datetime 
import matplotlib.pyplot as plt 
import os 
from tqdm import tqdm

FIGSIZE = (15,9)

# ~ training_data =  "/home/tim/Documents/stage-m2/tf_test/data/ArmToolsToy_random_train.pk"
training_data = "/home/tim/Documents/stage-m2/tf_test/data/ArmToolsToy_onlypertinentfromrandom_train.pk"
# ~ training_data = "/home/tim/Documents/stage-m2/tf_test/data/ArmToolsToy_pertinentfromrandom_noise01_train.pk"

eval_data = "/home/tim/Documents/stage-m2/tf_test/data/ArmToolsToy_onlypertinentfromrandom_eval.pk"
# ~ eval_data =  "/home/tim/Documents/stage-m2/tf_test/data/ArmToolsToy_random_eval.pk"

timestamp = datetime.datetime.now()
logdir = './log/'+str(timestamp)
os.makedirs(logdir)

REG = 0.0000

OBS_DIM = 18
ACS_DIM = 4
INPUT_DIM = OBS_DIM + ACS_DIM
OUTPUT_DIM = 22
EPOCH = 1


class Normalization:
    def __init__(self):
        self.inputs_mean = 0
        self.inputs_std = 1
        self.outputs_mean = 0
        self.outputs_std = 1
        self.samples = 0
        
    def init(self,inputs,outputs):
        if self.samples == 0:
            self.inputs_mean = np.mean(inputs,axis=0)
            self.inputs_std = np.std(inputs,axis=0)
            self.outputs_mean = np.mean(outputs,axis=0)
            self.outputs_std = np.std(outputs,axis=0)
            # ~ self.samples = len(inputs)
        else:
            n_old_samples = self.samples
            n_new_samples = len(inputs)
            self.samples = n_old_samples + n_new_samples
            alpha = n_old_samples/self.samples
            self.inputs_mean = alpha * self.inputs_mean + (1-alpha) * np.mean(inputs,axis=0)
            self.inputs_std = alpha * self.inputs_std + (1-alpha) * np.std(inputs,axis=0)
            self.outputs_mean = alpha * self.outputs_mean + (1-alpha) * np.mean(outputs,axis=0)
            self.outputs_std = alpha * self.outputs_std + (1-alpha) * np.std(outputs,axis=0)
        for i in range(len(self.inputs_std)):
            if self.inputs_std[i] == 0.:
                self.inputs_std[i] = 1
        for i in range(len(self.outputs_std)):
            if self.outputs_std[i] == 0.:
                self.outputs_std[i] = 1
        
    def load(self, model_dir):
        with open(os.path.join(model_dir, "norm.pk"), "br") as f:
            [self.inputs_mean,self.inputs_std,self.outputs_mean, self.outputs_std] = pickle.load(f)
        
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

def load_data( data):
    f = open(data , 'rb')
    data = pickle.load(f)
    f.close()
    
    if type(data) is dict:
        true_traj,Acs = data['o'], data['u']
    else:
        [true_traj,Acs] = data
    return np.array(true_traj), Acs

def compute_samples(data,norm):
    true_traj, Acs = load_data(data)
    
    Inputs = []
    Targets = []
    moved_objects = 0
    for j in range(len(true_traj)):
        for t in range(50):
            inputs = np.concatenate((true_traj[j][t][:OBS_DIM],Acs[j][t]))
            targets = true_traj[j][t+1][:OBS_DIM] - true_traj[j][t][:OBS_DIM]
            bool_targets = [1 if abs(targets[i])>0 else -1 for i in [6,10,14,16]]
            Inputs.append(inputs)
            Targets.append(np.concatenate((targets,bool_targets)))
            if 1 in bool_targets :
                moved_objects += 1
    print( 'moved_objects transitions', moved_objects, moved_objects/(50*j))
    # ~ norm.init(Inputs,Targets)
    # ~ norm.pretty_print()
    samples = list(zip( Inputs,Targets))
    np.random.shuffle( samples)
    Inputs, Targets = zip( *samples)
    
    return (norm.normalize_inputs(np.array(Inputs)),norm.normalize_outputs(np.array(Targets)))


def evaluation(model, norm, data_type="eval", epoch='final'):
    print("*** Prediction evaluation: "+data_type)
    if data_type == "eval":
        f = open(eval_data , 'rb')
    else:
        f = open(training_data , 'rb')
    
    data =  pickle.load(f)
    f.close()

    if type(data) is dict:
        true_traj, Acs = data['o'], data['u']
    else:
        [true_traj,Acs] = data

    true_traj = np.array(true_traj)
    pred = []
    error_traj = []
    confusion_matrix = np.zeros((2,2))
    print("Trajectory")
    for j in tqdm(range(len(true_traj))):
        obs = [true_traj[j][0][:OBS_DIM]]
        for t in range(50):
            inputs = norm.normalize_inputs(np.concatenate((obs[-1][:OBS_DIM],Acs[j][t])))
            inputs = np.array([inputs])
            output = norm.denormalize_outputs(model.predict(inputs)[0])
            obs_pred = output[:OBS_DIM]+obs[-1]
            obs_pred[5] = 1 if obs_pred[5]>0 else -1
            for (i,b) in [(6,18),(7,18),(8,18),(9,18),(10,19),(11,19),(12,19),(13,19),(14,20),(15,20),(16,21),(17,21)]:
                obs_pred[i] = obs_pred[i] if output[b] > 0 else obs[-1][i]
            obs.append(obs_pred)
        pred.append(obs)
        error_traj.append( np.linalg.norm(np.array(obs)[:,:OBS_DIM]-np.array(true_traj[j,:,:OBS_DIM])))
        ''' confusion matrice on sticks reward '''
        for (x,y) in [ (8,9), (12,13)]:
            t_d_x = true_traj[j,-1,x]-true_traj[j,0,x]
            t_d_y = true_traj[j,-1,y]-true_traj[j,0,y]
            d_x = pred[j][-1][x]-pred[j][0][x]
            d_y = pred[j][-1][y]-pred[j][0][y]
            
            t_l,l = t_d_x > 0.1, d_x > 0.1
            t_r,r = t_d_x < -0.1, d_x < -0.1
            t_f,f = t_d_y > 0.1, d_y > 0.1
            t_c,c = t_d_y < -0.1, d_y < -0.1
            
            for (t,m) in [(t_l,l),(t_r,r),(t_f,f),(t_c,c)]:
                confusion_matrix[int(t)][int(m)] += 1
    print( 'confusion matrix', confusion_matrix)
    pred_traj = np.array(pred)
    
    print("Transition")
    pred = []
    error_trans = []
    for j in tqdm(range(len(true_traj))):
        obs = [true_traj[j][0][:OBS_DIM]]
        for t in range(50):
            inputs = np.array([norm.normalize_inputs(np.concatenate((true_traj[j][t][:OBS_DIM],Acs[j][t])))])
            obs.append(norm.denormalize_outputs(model.predict(inputs)[0])[:OBS_DIM]+true_traj[j][t][:OBS_DIM])
        pred.append(obs)
        error_trans.append( np.linalg.norm(np.array(obs)[:,:OBS_DIM]-np.array(true_traj[j,:,:OBS_DIM])))
    pred_trans = np.array(pred)
    

    filename = logdir+"/"+data_type+str(epoch)+".pk"
    f = open(filename,'bw')
    pickle.dump((pred_traj,pred_trans,true_traj,error_traj,error_trans),f)
    f.close()
    
    filename = logdir+"/confusion_matrix"+str(epoch)+".pk"
    f = open(filename,'bw')
    pickle.dump(confusion_matrix,f)
    f.close()


# ~ (x_train, y_train) = compute_samples( training_data, norm)
# ~ (x_eval, y_eval) = compute_samples( eval_data, norm)


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
    tf.keras.layers.Dense(OUTPUT_DIM , activation=None),
])


model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mean_squared_error']
              )

# ~ model_dir = "/home/tim/Documents/stage-m2/gym-myFetchPush/log/tf25/"
# ~ model.load_weights(model_dir+'model.h5')
# ~ norm.load(model_dir)

# ~ history = model.fit(x_train, y_train,
                        # ~ epochs=EPOCH,
                        # ~ #validation_split=0.1,
                        # ~ validation_data = (x_eval,y_eval),
                        # ~ shuffle=True
                        # ~ )

# ~ for epoch in range(EPOCH):
    # ~ history = model.fit(x_train, y_train,
                        # ~ epochs=5,
                        # ~ #validation_split=0.1,
                        # ~ validation_data = (x_eval,y_eval),
                        # ~ shuffle=True
                        # ~ )
    # ~ evaluation(model,norm, "eval", epoch=epoch)
    
""" Saving """
model.save_weights(logdir+'/model.h5')
with open(os.path.join(logdir, "config.txt"), "w") as f:
    def write(x):
        f.write(x+"\n")
    model.summary(print_fn=write)


# ~ """ Evaluation """
# ~ data = history.history
# ~ plot_MSE(data)
# ~ evaluation(model,norm, "eval")
# ~ evaluation(model,norm, "training")
