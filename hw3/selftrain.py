import pickle
import random
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.normalization import BatchNormalization
from keras.backend.tensorflow_backend import set_session
import sys

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
set_session(tf.Session(config=config))

def shuffle(a, b):
    permutation = np.random.permutation(a.shape[0])
    sa = a[permutation]
    sb = b[permutation]
    return sa, sb

minibatch = 50
epo1 = 120
epo2 = 20
iterrange = 15


all_label = pickle.load(open(sys.argv[1]+'all_label.p','rb'))
X_input = np.array(all_label)
X_input = np.reshape(X_input,newshape=(5000,3,32,32))
X_input = np.rollaxis(X_input, 1, 4)
X_input = X_input.astype('float32')
X_input /= 255

X_label=np.zeros((5000,10),dtype=int)
for i in range(0,10):
    for j in range(i*500,i*500+500):
        X_label[j][i]=1

X_input, X_label = shuffle(X_input, X_label)
X_validdata = X_input[4500:5000, :, :, :]
X_validlabel = X_label[4500:5000, :]
X_input = X_input[0:4500, :, :, :]
X_label = X_label[0:4500, :]

Actinit = PReLU(init='zero', weights=None)
Act = LeakyReLU(alpha=0.3)

model = Sequential()
model.add(Convolution2D(32, 3, 3, border_mode='same',
                        input_shape=(32,32,3)))
#L1
model.add(Actinit)
model.add(Convolution2D(32, 3, 3, border_mode='same'))
model.add(BatchNormalization())
model.add(Act)
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
#L2
model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(Act)
model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(BatchNormalization())
model.add(Act)
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
#L3
model.add(Convolution2D(128, 3, 3, border_mode='same'))
model.add(Act)
model.add(Convolution2D(128, 3, 3, border_mode='same'))
model.add(BatchNormalization())
model.add(Act)
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
#L4
model.add(Convolution2D(256, 3, 3, border_mode='same'))
model.add(Act)
model.add(Convolution2D(256, 3, 3, border_mode='same'))
model.add(BatchNormalization())
model.add(Act)
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
#Dense Layer
model.add(Flatten())
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Act)
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

#unlabel first store
unb = pickle.load(open(sys.argv[1]+'all_unlabel.p','rb'))
X_unlabel = np.array(unb)
X_unlabel = np.reshape(X_unlabel,newshape=(45000,3,32,32))
X_unlabel = np.rollaxis(X_unlabel, 1, 4)
X_unlabel = X_unlabel.astype('float32')
X_unlabel /= 255

# this will do preprocessing and realtime data augmentation
datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False)  # randomly flip images

datagen.fit(X_input)

print('Done Pickling and Modeling')

model.fit_generator(datagen.flow(X_input, X_label,
                    batch_size=minibatch),
                    samples_per_epoch=X_input.shape[0],
                    nb_epoch=epo1,
                    validation_data=(X_validdata,X_validlabel))

for i in range(0, iterrange):
    print(repr(i)+'iter')
    if i > 9:
        epo2 = 10
    X_out = model.predict_proba(X_unlabel, batch_size=minibatch)
    toadd_index = np.where(X_out > 0.995)
    X_toadd = X_unlabel[toadd_index[0],:,:,:]
    X_unlabel = np.delete(X_unlabel, toadd_index[0], axis = 0)
    X_newlab = np.zeros((np.size(toadd_index[0]), 10))
    for k in range(0, np.size(toadd_index[0])):
        X_newlab[k, toadd_index[1][k]] = 1
    X_input = np.concatenate((X_input,X_toadd), axis=0)
    X_label = np.concatenate((X_label,X_newlab), axis=0)
    datagen.fit(X_input)
    model.fit_generator(datagen.flow(X_input, X_label,
                        batch_size=minibatch),
                        samples_per_epoch=X_input.shape[0],
                        nb_epoch=epo2,
                        validation_data=(X_validdata,X_validlabel))

model.save(sys.argv[2])
