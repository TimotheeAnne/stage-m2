import tensorflow as tf
import numpy as np
import pickle 
import datetime 
import matplotlib.pyplot as plt 
import os 
from tqdm import tqdm

FIGSIZE = (15,9)

# ~ training_data =  "./data/transition_multi25_0noise_train.pk"
# ~ eval_data = "./data/transition_multi25_0noise_eval.pk"

# ~ training_data = "./data/transition_multiTask_1_train.pk"
# ~ eval_data ="./data/transition_multiTask_1_eval.pk"

# ~ training_data = "/home/tim/Documents/stage-m2/multi-task-her-rl/src/data/MultiTaskFetchArmNLP1-v0/16700/train_episodes32.pk"

# ~ training_data = "/home/tim/Documents/stage-m2/tf_test/data/transition_multi25_firstrandom_train.pk"
# ~ eval_data = "/home/tim/Documents/stage-m2/tf_test/data/transition_multi25_firstrandom_eval.pk"

training_data = "/home/tim/Documents/stage-m2/tf_test/data/multi25_50percent_train.pk"
eval_data = "/home/tim/Documents/stage-m2/tf_test/data/multi25_50percent_eval.pk"
training_transition = "/home/tim/Documents/stage-m2/tf_test/data/multi25_50percent_train_transition.pk"


timestamp = datetime.datetime.now()
logdir = './log/'+str(timestamp)
os.makedirs(logdir)

REG = 0.0005
INPUT_DIM = 29
EPOCH = 20


class Normalization:
    def __init__(self):
        self.inputs_mean = 0
        self.inputs_std = 1
        self.outputs_mean = 0
        self.outputs_std = 1
        
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

    def load(self, model_dir):
        with open(os.path.join(model_dir, "norm.pk"), "br") as f:
            [self.inputs_mean,self.inputs_std,self.outputs_mean, self.outputs_std] = pickle.load(f)

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


def compute_samples(data,norm):
    f = open(data , 'rb')
    data = pickle.load(f)
    f.close()

    Inputs = []
    Targets = []
        
    if type(data) is dict:
        true_traj,Acs = data['o'], data['u']
    elif type(data) is tuple:
        (Inputs,Targets) = data
        norm.init(Inputs,Targets)
        return (norm.normalize_inputs(np.array(Inputs)),norm.normalize_outputs(np.array(Targets)))
    else:
        [true_traj,Acs] = data
        
    moving_cube = 0
    for j in range(len(true_traj)):
        for t in range(50):
            inputs = np.concatenate((true_traj[j][t][:25],Acs[j][t]))
            targets = true_traj[j][t+1][:25] - true_traj[j][t][:25]
            Inputs.append(inputs)
            Targets.append(targets)
            if np.linalg.norm(targets[3:6]) > 0.001:
                moving_cube += 1
    print( 'moving cube transtions', moving_cube, moving_cube/(50*j))
    norm.init(Inputs,Targets)
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
        if data_type == "eval":
            data = data[0][0]
            true_traj, Acs = data['o'][1:], data['u'][1:]
        else:
            [true_traj,Acs] = data

    true_traj = np.array(true_traj)
    pred = []
    error_traj = []
    confusion_matrix = np.zeros((2,2))
    print("Trajectory")
    for j in tqdm(range(len(true_traj))):
        obs = [true_traj[j][0][:25]]
        for t in range(50):
            inputs = norm.normalize_inputs(np.concatenate((obs[-1],Acs[j][t])))
            inputs = np.array([inputs])
            # ~ if j == 0 and t<3:
                # ~ print('input', inputs)
                # ~ print(j,t, model.predict(inputs)[0])
                # ~ print('pred', norm.denormalize_outputs(model.predict(inputs)[0])+obs[-1])
            obs.append(norm.denormalize_outputs(model.predict(inputs)[0])+obs[-1])
        pred.append(obs)
        error_traj.append( np.linalg.norm(obs-np.array(true_traj[j,:,:25])))
        ''' confusion matrice on cube reward '''
        t_d_x = true_traj[j,-1,3]-true_traj[j,0,3]
        t_d_y = true_traj[j,-1,4]-true_traj[j,0,4]
        d_x = pred[j][-1][3]-pred[j][0][3]
        d_y = pred[j][-1][4]-pred[j][0][4]
        
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
    # ~ for j in tqdm(range(len(true_traj))):
        # ~ obs = [true_traj[j][0][:25]]
        # ~ for t in range(50):
            # ~ inputs = np.array([norm.normalize_inputs(np.concatenate((true_traj[j][t][:25],Acs[j][t])))])
            # ~ obs.append(norm.denormalize_outputs(model.predict(inputs)[0])+true_traj[j][t][:25])
        # ~ pred.append(obs)
        # ~ error_trans.append( np.linalg.norm(obs-np.array(true_traj[j,:,:25])))
    pred_trans = np.array(pred)
    

    filename = logdir+"/"+data_type+str(epoch)+".pk"
    f = open(filename,'bw')
    pickle.dump((pred_traj,pred_trans,true_traj,error_traj,error_trans),f)
    f.close()
    
    filename = logdir+"/confusion_matrix"+str(epoch)+".pk"
    f = open(filename,'bw')
    pickle.dump(confusion_matrix,f)
    f.close()


(x_train, y_train) = compute_samples( training_transition, norm)
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

print(model.trainable_variables)
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
    
# ~ """ Saving """
# ~ model.save_weights(logdir+'/model.h5')
# ~ with open(os.path.join(logdir, "config.txt"), "w") as f:
    # ~ def write(x):
        # ~ f.write(x+"\n")
    # ~ model.summary(print_fn=write)
# ~ norm.save()

# ~ """ Evaluation """
# ~ data = history.history
# ~ plot_MSE(data)
# ~ evaluation(model,norm, "eval")
# ~ evaluation(model,norm, "training")
