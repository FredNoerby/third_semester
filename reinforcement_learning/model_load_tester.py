import tensorflow as tf

import time, random, threading
import EnvironmentB

from keras.models import *
from keras.layers import *

model = load_model('model_saved.h5f')
print(model.get_config())
print(" ")
print(model.get_weights())