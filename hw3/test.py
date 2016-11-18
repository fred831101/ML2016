import pickle
import numpy as np
import csv, math
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.models import load_model
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import pandas as pd
import sys

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

data= pd.read_pickle(sys.argv[1]+'test.p')
testid = np.array(data['ID'])
Y_input = np.array(data['data'])
Y_input = Y_input.reshape(10000, 3, 32, 32)
Y_input = np.rollaxis(Y_input, 1, 4)
Y_input = Y_input.astype('float32')
Y_input = Y_input/255

model = load_model(sys.argv[2])
Y_out = model.predict(Y_input)
Y_label = np.argmax(Y_out, axis = 1)

f = open(sys.argv[3], 'w')
f.write('ID,class\n')
for i in range(0,10000):
    f.write(repr(i)+','+repr(Y_label[i])+'\n')
f.close()
