import matplotlib.pyplot as plt
import json, time

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.python.client import device_lib

print(tf.test.is_built_with_cuda())

print(device_lib.list_local_devices())

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(physical_devices, 'GPU')

print(device_lib.list_local_devices())
