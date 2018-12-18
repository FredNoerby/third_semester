import cv2
import datetime
import itertools
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from robot_control.project_robot import ProjectRobot
from reinforcement_learning.project_environment import ProjectEnvironment

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from keras import regularizers

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.YlOrBr):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = 0.699
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True banana position')
    plt.xlabel('Predicted banana position')
    plt.tight_layout()


"""
##################################
##################################
###-----SIMULATED VERSION------###
##################################
##################################

##########################
### DQN IMPLEMENTATION ###

ENV_NAME = "L4_N40_28_11_2018"

env = ProjectEnvironment(simulated=True, print_log=True, occlusions=False, manual_occlusion=True)

nb_actions = env.action_space.n
print("Number of actions = {}".format(nb_actions))

# Lambda
lambd = 0.01

# Next, we build a very simple model.
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(40, kernel_regularizer=regularizers.l2(lambd)))
model.add(Activation('relu'))
model.add(Dense(40, kernel_regularizer=regularizers.l2(lambd)))
model.add(Activation('relu'))
model.add(Dense(40, kernel_regularizer=regularizers.l2(lambd)))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=5000, window_length=1)
policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

#### Loading model ####
dqn.load_weights('reinforcement_learning/dqn_{}_wghts_100000.h5f'.format(ENV_NAME))


#### FOR TRAINING MODEL ####
#training_episodes = 100000
#dqn.fit(env, nb_steps=training_episodes, verbose=2)
#dqn.save_weights('reinforcement_learning/dqn_{}_wghts_100000.h5f'.format(ENV_NAME), overwrite=True)


######################
episodes = 1000

dqn.test(env, nb_episodes=episodes)
total_reward = 0
total_wins = 0
total_steps = 0
y_true = []
y_pred = []
w_occ = 0
l_occ = 0
w_steps = 0
l_steps = 0

for entry in env.history:
    total_steps += len(entry['actions'])
    total_reward += np.sum(entry['rewards'])
    if entry['won']:
        total_wins += 1
        print('won, occlusions:', entry['occlusions'], 'reward', np.sum(entry['rewards']))
        w_occ += entry['occlusions']
        w_steps += len(entry['actions'])
    else:
        print('Lost, occlusions:', entry['occlusions'], 'reward', np.sum(entry['rewards']))
        l_occ += entry['occlusions']
        l_steps += len(entry['actions'])
    y_true.append(entry['banana_pose'])
    y_pred.append(entry['banana_pred'])

print('Average steps taken:', (total_steps / episodes))
print('Average reward:', (total_reward / episodes))
print('Accuracy:', (total_wins / episodes))
# print(y_true)
# print(y_pred)
print('F1 score weighted:', f1_score(y_true, y_pred, average='weighted'))

print('Average occlusions per win:', (w_occ / total_wins))
print('Average steps per win:', (w_steps / total_wins))

print('Average occlusions per loss:', (l_occ / (episodes - total_wins)))
print('Average steps per loss:', (l_steps / (episodes - total_wins)))


# Compute confusion matrix
cnf_matrix = confusion_matrix(y_true, y_pred)
np.set_printoptions(precision=2)

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8], normalize=True,
                      title='Normalized confusion matrix')

plt.show()

env.close()



"""
###################################
###################################
###-----REAL ROBOT VERSION------###
###################################
###################################

rob = ProjectRobot(acc=1, vel=1)



##########################
### DQN IMPLEMENTATION ###

ENV_NAME = "L4_N40_28_11_2018"

# Get the environment and extract the number of actions.
#env = ProjectEnvironment(print_log=False)
env = ProjectEnvironment(banana_pose=7, occlusions=True, manual_occlusion=False, robot_pose_start=7, simulated=False, ProjectRobot=rob, print_log=True, video_cap=0, incremental_robot_positions=True, frozen_graph_path='/home/frederik/AAU_CPH/models/research/object_detection/rcnn_training_folder/exported_graphs/model_october15/frozen_inference_graph.pb')
# np.random.seed(123)
# env.seed(123)
nb_actions = env.action_space.n
print("Number of actions = {}".format(nb_actions))

# Next, we build a very simple model.
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(40))
model.add(Activation('relu'))
model.add(Dense(40))
model.add(Activation('relu'))
model.add(Dense(40))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=5000, window_length=1)
policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy)

dqn.compile(Adam(lr=1e-3), metrics=['mae'])

dqn.load_weights('reinforcement_learning/dqn_{}_wghts_100000.h5f'.format(ENV_NAME))

##########################
episodes = 96

dqn.test(env, nb_episodes=episodes)
total_reward = 0
total_wins = 0
total_steps = 0
y_true = []
y_pred = []


for entry in env.history:
    total_steps += len(entry['actions'])
    total_reward += np.sum(entry['rewards'])
    if entry['won']:
        total_wins += 1
    y_true.append(entry['banana_pose'])
    y_pred.append(entry['banana_pred'])

print('Average steps taken:', (total_steps / episodes))
print('Average reward:', (total_reward / episodes))
print('Accuracy:', (total_wins / episodes))
# print(y_true)
# print(y_pred)
print('F1 score weighted:', f1_score(y_true, y_pred, average='weighted'))



# Compute confusion matrix
cnf_matrix = confusion_matrix(y_true, y_pred)
np.set_printoptions(precision=2)

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8], normalize=True,
                      title='Normalized confusion matrix')

plt.show()

env.close()
