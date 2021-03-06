# OpenGym CartPole - v0 with A3C on GPU
# -----------------------------------
#
# https://github.com/jaara/AI-blog/blob/master/CartPole-A3C.py
# A3C implementation with GPU optimizer threads.
#
# Made as part of blog series Let's make an A3C, available at
# https://jaromiru.com/2017/02/16/lets-make-an-a3c-theory/
#
# author: Jaromir Janisch, 2017

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

import time, random, threading
import EnvironmentB

from keras.models import *
from keras.layers import *
from keras import backend as K

# -- constants
ENV = EnvironmentB.EnvironmentB()

RUN_TIME = 220  # 30 seconds for cartpole
THREADS = 2  # (n Agents) 8 Due to 4 physical CPU cores in author PC
OPTIMIZERS = 2  # 2
THREAD_DELAY = 0.001  # 0.001

GAMMA = 0.99  # 0.99

N_STEP_RETURN = 8  # 8
GAMMA_N = GAMMA ** N_STEP_RETURN

EPS_START = 1.0  # 0.40
EPS_STOP = .4  # 0.15
EPS_STEPS = 375000  # 75000

MIN_BATCH = 24  # 32  (24 seems like a good candidate)
LEARNING_RATE = 5e-7  # 5e-5 (5e-7 seems like a good candidate)

LOSS_V = .5  # v loss coefficient 0.5
LOSS_ENTROPY = .1  # entropy coefficient 0.01

# Self made values for improving and testing script:
# ------------------
saver_toggle = 0
loader_toggle = 0

score_counter = 0
scores = []
acc_scores = 0
eps_current = 0
eps_list = []
# ------------------


# ---------
class Brain:
    train_queue = [[], [], [], [], []]  # s, a, r, s', s' terminal mask
    lock_queue = threading.Lock()

    def __init__(self):



        self.session = tf.Session()
        K.set_session(self.session)
        K.manual_variable_initialization(True)


        self.model = self._build_model()

        if loader_toggle:
            self.model.load_weights('model_saved.h5f')
            print('Weights loaded')


        self.graph = self._build_graph(self.model)

        self.session.run(tf.global_variables_initializer())
        print(self.model.get_weights())
        self.default_graph = tf.get_default_graph()


        self.default_graph.finalize()  # comment out to allow for saving of model and weights

    def _build_model(self):

        l_input = Input(batch_shape=(None, NUM_STATE))
        l_dense = Dense(16, activation='relu')(l_input)

        out_actions = Dense(NUM_ACTIONS, activation='softmax')(l_dense)
        out_value = Dense(1, activation='linear')(l_dense)

        if loader_toggle:
            model = load_model('model_saved.h5f')
            print('Saved model has been loaded')
        else:
            model = Model(inputs=[l_input], outputs=[out_actions, out_value])

        model._make_predict_function()  # have to initialize before threading

        return model

    def _build_graph(self, model):
        s_t = tf.placeholder(tf.float32, shape=(None, NUM_STATE))
        a_t = tf.placeholder(tf.float32, shape=(None, NUM_ACTIONS))
        r_t = tf.placeholder(tf.float32, shape=(None, 1))  # not immediate, but discounted n step reward

        p, v = model(s_t)

        log_prob = tf.log(tf.reduce_sum(p * a_t, axis=1, keep_dims=True) + 1e-10)
        advantage = r_t - v

        loss_policy = - log_prob * tf.stop_gradient(advantage)  # maximize policy
        loss_value = LOSS_V * tf.square(advantage)  # minimize value error
        entropy = LOSS_ENTROPY * tf.reduce_sum(p * tf.log(p + 1e-10), axis=1,
                                               keep_dims=True)  # maximize entropy (regularization)

        loss_total = tf.reduce_mean(loss_policy + loss_value + entropy)

        # optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
        optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
        # optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, decay=.99)
        minimize = optimizer.minimize(loss_total)

        #self.saver = tf.train.Saver([s_t, a_t, r_t, minimize])

        return s_t, a_t, r_t, minimize

    def optimize(self):
        if len(self.train_queue[0]) < MIN_BATCH:
            time.sleep(0)  # yield
            return

        with self.lock_queue:
            if len(self.train_queue[0]) < MIN_BATCH:  # more thread could have passed without lock
                return  # we can't yield inside lock

            s, a, r, s_, s_mask = self.train_queue
            self.train_queue = [[], [], [], [], []]

        s = np.vstack(s)
        a = np.vstack(a)
        r = np.vstack(r)
        s_ = np.vstack(s_)
        s_mask = np.vstack(s_mask)

        if len(s) > 5 * MIN_BATCH: print("Optimizer alert! Minimizing batch of %d" % len(s))

        v = self.predict_v(s_)
        r = r + GAMMA_N * v * s_mask  # set v to 0 where s_ is terminal state

        s_t, a_t, r_t, minimize = self.graph
        self.session.run(minimize, feed_dict={s_t: s, a_t: a, r_t: r})

    def train_push(self, s, a, r, s_):
        with self.lock_queue:
            self.train_queue[0].append(s)
            self.train_queue[1].append(a)
            self.train_queue[2].append(r)

            if s_ is None:
                self.train_queue[3].append(NONE_STATE)
                self.train_queue[4].append(0.)
            else:
                self.train_queue[3].append(s_)
                self.train_queue[4].append(1.)

    def predict(self, s):
        with self.default_graph.as_default():
            p, v = self.model.predict(s)
            return p, v

    def predict_p(self, s):
        with self.default_graph.as_default():
            p, v = self.model.predict(s)
            return p

    def predict_v(self, s):
        with self.default_graph.as_default():
            p, v = self.model.predict(s)
            return v

    def saving(self):
        if saver_toggle:
            self.model.save_weights('test_file.h5f')

            tf.keras.models.save_model(
                self.model,
                'model_saved.h5f',
                overwrite=True,
                include_optimizer=True
            )

            # self.model.save('model_saved.h5f', overwrite=True, include_optimizer=True)
            # self.saver.save(tf.Session(), 'my-model', global_step=999)

            print("Model Saved...")


# ---------
frames = 0


class Agent:
    def __init__(self, eps_start, eps_end, eps_steps):
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_steps = eps_steps

        self.memory = []  # used for n_step return
        self.R = 0.

    def getEpsilon(self):
        if (frames >= self.eps_steps):
            return self.eps_end
        else:
            return self.eps_start + frames * (self.eps_end - self.eps_start) / self.eps_steps  # linearly interpolate

    def act(self, s):
        eps = self.getEpsilon()
        global eps_current
        eps_current = eps
        global frames
        # print("Epsilon for thread {} is {}".format(threading.get_ident(),eps))
        frames = frames + 1

        if random.random() < eps:
            return random.randint(0, NUM_ACTIONS - 1)

        else:
            s = np.array([s])
            p = brain.predict_p(s)[0]

            # a = np.argmax(p)
            a = np.random.choice(NUM_ACTIONS, p=p)

            return a

    def train(self, s, a, r, s_):
        def get_sample(memory, n):
            s, a, _, _ = memory[0]
            _, _, _, s_ = memory[n - 1]

            return s, a, self.R, s_

        a_cats = np.zeros(NUM_ACTIONS)  # turn action into one-hot representation
        a_cats[a] = 1

        self.memory.append((s, a_cats, r, s_))

        self.R = (self.R + r * GAMMA_N) / GAMMA

        if s_ is None:
            while len(self.memory) > 0:
                n = len(self.memory)
                s, a, r, s_ = get_sample(self.memory, n)
                brain.train_push(s, a, r, s_)

                self.R = (self.R - self.memory[0][2]) / GAMMA
                self.memory.pop(0)

            self.R = 0

        if len(self.memory) >= N_STEP_RETURN:
            s, a, r, s_ = get_sample(self.memory, N_STEP_RETURN)
            brain.train_push(s, a, r, s_)

            self.R = self.R - self.memory[0][2]
            self.memory.pop(0)


# possible edge case - if an episode ends in <N steps, the computation is incorrect

# ---------
class Environment(threading.Thread):
    stop_signal = False

    def __init__(self, render=False, eps_start=EPS_START, eps_end=EPS_STOP, eps_steps=EPS_STEPS):
        threading.Thread.__init__(self)

        self.render = render
        self.env = ENV
        self.agent = Agent(eps_start, eps_end, eps_steps)

    def runEpisode(self):
        s = self.env.reset()

        R = 0
        while True:

            time.sleep(THREAD_DELAY)  # yield

            # if self.render: self.env.render()

            a = self.agent.act(s)
            s_, r, done, info = self.env.step(a)

            if done:  # terminal state
                s_ = None

            self.agent.train(s, a, r, s_)

            s = s_
            R += r

            if done or self.stop_signal:
                global score_counter
                global acc_scores
                acc_scores += R
                score_counter += 1
                if score_counter % 200 == 0:
                    eps_list.append(eps_current)
                    scores.append(acc_scores)

                break

        print("Time/Tot.Time/Nthreads/Total R: {:.1f} / {} / {} / {}".format(time.clock(), RUN_TIME, threading.active_count(), R))

    def run(self):
        while not self.stop_signal:
            self.runEpisode()

    def stop(self):
        self.stop_signal = True


Full_stop = 0
n_tests = 11
runs = 0


class EnvironmentRunner():
    stop_signal = False

    def __init__(self, render=False, eps_start=EPS_START, eps_end=EPS_STOP, eps_steps=EPS_STEPS):

        self.render = render
        self.env = ENV
        self.agent = Agent(eps_start, eps_end, eps_steps)

    def counter(self):
        global Full_stop
        if Full_stop < n_tests:
            Full_stop += 1
            print("Full_stop count = {}".format(Full_stop))

    def runEpisode(self):
        EnvironmentRunner.counter(self)
        step_counter = 0

        if Full_stop < n_tests:
            s = self.env.reset()

            R = 0
            while not self.stop_signal or (Full_stop >= n_tests):

                time.sleep(THREAD_DELAY)  # yield

                # if self.render: self.env.render()

                a = self.agent.act(s)
                s_, r, done, info = self.env.step(a)

                if done:  # terminal state
                    s_ = None

                self.agent.train(s, a, r, s_)

                s = s_
                R += r

                step_counter += 1
                if step_counter > 20:
                    print("__Forced DONE__" * 5)
                    done = True

                if done or self.stop_signal or (Full_stop >= n_tests):
                    break


            print("Runner - n threads / Total R: {} / {}".format(threading.active_count(), R))

    def run(self):
        while not self.stop_signal or (Full_stop >= n_tests):
            global runs
            runs += 1
            if runs < 20:
                print("Runs: {}".format(runs))
            self.runEpisode()
            if Full_stop >= n_tests - 1:
                return

    def stop(self):
        self.stop_signal = True


# ---------
class Optimizer(threading.Thread):
    stop_signal = False

    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        while not self.stop_signal:
            brain.optimize()

    def stop(self):
        self.stop_signal = True


# -- main
env_test = EnvironmentRunner(eps_start=0., eps_end=0)
NUM_STATE = env_test.env.observation_space.shape[0]
NUM_ACTIONS = env_test.env.action_space.n
NONE_STATE = np.zeros(NUM_STATE)
print("NUM_STATE and NUM ACTIONS: {} / {}".format(NUM_STATE, NUM_ACTIONS))


brain = Brain()  # brain is global in A3C

envs = [Environment() for i in range(THREADS)]
opts = [Optimizer() for i in range(OPTIMIZERS)]

for o in opts:
    o.start()

for e in envs:
    e.start()

time.sleep(RUN_TIME)

for i in range(20):
    print("__TEST__" * 8)

for e in envs:
    e.stop()
for e in envs:
    e.join()

for o in opts:
    o.stop()
for o in opts:
    o.join()

print("Training finished")
brain.saving()
print("pre-run")
while True:
    if Full_stop < n_tests:
        env_test.run()
    else:
        break

print("plotting scores and eps...")
print("scores length = {}".format(len(scores)))
# print(scores)

fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
ax1.plot(eps_list, linestyle='--')
ax2.plot(scores)
plt.show()

print("done")
