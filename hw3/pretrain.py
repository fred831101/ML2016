import pickle
import random
import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Dropout, Flatten, Reshape
from keras.layers import Convolution2D, MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.backend.tensorflow_backend import set_session
from keras.models import Model
from keras.layers import Input
import sys

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
set_session(tf.Session(config=config))

def shuffle(a, b):
    permutation = np.random.permutation(a.shape[0])
    sa = a[permutation]
    sb = b[permutation]
    return sa, sb

minibatch = 100
epo = 75
epo2 = 150

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

unb = pickle.load(open(sys.argv[1]+'all_unlabel.p','rb'))
X_unlabel = np.array(unb)
X_unlabel = np.reshape(X_unlabel,newshape=(45000,3,32,32))
X_unlabel = np.rollaxis(X_unlabel, 1, 4)
X_unlabel = X_unlabel.astype('float32')
X_unlabel /= 255

print(X_unlabel.shape)
X_inputall = np.concatenate((X_input, X_unlabel), axis = 0)


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

#Convolutional layer
input_img = Input(shape=(32, 32, 3))

x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(input_img)
x = MaxPooling2D((2, 2), border_mode='same')(x)
x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(x)
x = MaxPooling2D((2, 2), border_mode='same')(x)
x = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(x)
x = MaxPooling2D((2, 2), border_mode='same')(x)
x = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(x)
encoded = MaxPooling2D((2, 2), border_mode='same')(x)

x = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(x)
x = UpSampling2D((2, 2))(x)
x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(x)
x = UpSampling2D((2, 2))(x)
x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Convolution2D(3, 3, 3, activation='sigmoid', border_mode='same')(x)

encoder = Model(input=input_img, output=encoded)
autoencoder = Model(input=input_img, output=decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=["accuracy"])


#Dense Layer
input_img2 = Input(shape=(256,))
encoded2 = Dense(150, activation='relu')(input_img2)
decoded2 = Dense(256, activation='relu')(encoded2)
encoder2 = Model(input=input_img2, output=encoded2)
autoencoder2 = Model(input=input_img2, output=decoded2)
autoencoder2.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=["accuracy"])

#Softmax Layer
softlayer = Sequential()
softlayer.add(Dense(10, input_dim=150))
softlayer.add(Activation('softmax'))
softlayer.compile(loss='categorical_crossentropy', optimizer= 'adam', metrics=['accuracy'])


#Apply Noise Factor
noise_factor = 0.5
X_withnoise = X_inputall + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_inputall.shape)
X_withnoise = np.clip(X_withnoise, 0., 1.)


#Autoencoder fitting (Pretrain)
datagen.fit(X_withnoise)
autoencoder.fit_generator(datagen.flow(X_withnoise, X_inputall, batch_size=minibatch),
                                        samples_per_epoch=X_withnoise.shape[0],
                                        nb_epoch=epo)
X_inputall = encoder.predict(X_inputall)
X_inputenc = encoder.predict(X_input)


X_inputall = np.reshape(X_inputall, (X_inputall.shape[0],256))
X_withnoise = X_inputall + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_inputall.shape)
X_withnoise = np.clip(X_withnoise, 0., 1.)
autoencoder2.fit(X_withnoise, X_inputall, nb_epoch=epo2, batch_size=minibatch)

X_inputall = encoder2.predict(np.reshape(X_inputenc, (X_inputenc.shape[0],256)))
softlayer.fit(X_inputall,X_label,batch_size=minibatch,nb_epoch=epo)

encoders = []
encoders.append(encoder)
encoders.append(encoder2)
encoders.append(softlayer)

# Fine-tune
model = Sequential()
model.add(encoders[0])
model.add(Flatten())
model.add(encoders[1])
model.add(encoders[2])
model.compile(loss='categorical_crossentropy',
              optimizer= 'adam',
              metrics=['accuracy'])

datagen.fit(X_input)

model.fit_generator(datagen.flow(X_input, X_label,
                    batch_size=minibatch),
                    samples_per_epoch=X_input.shape[0],
                    nb_epoch=epo,validation_data=(X_validdata, X_validlabel))


model.save(sys.argv[2])
