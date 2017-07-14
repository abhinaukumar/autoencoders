# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 21:05:20 2017

@author: nownow
"""

import keras
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Merge, Reshape
from keras.models import Model
import numpy as np

from keras.datasets import mnist

import matplotlib.pyplot as plt

from keras import backend as K

plt.ion()

from keras import metrics

import manifold_script

batch_size = 100
original_dim = 784
latent_dim = 2
intermediate_dim = 256
epochs = 25
epsilon_std = 1.0

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(100,2), mean=0.,
                              stddev=1.)
    return z_mean + K.exp(z_log_var / 2) * epsilon
    
input_img = Input(shape=(1, 28, 28))
class_in = Input(shape=(10,))
    
conv_1 = Conv2D(4, (3,3), activation='relu', padding='same') # 28x28
pool_1 = MaxPooling2D((2, 2), padding='same') # 14x14
conv_2 = Conv2D(8, (3,3), activation='relu', padding='same') # 14x14
pool_2 = MaxPooling2D((2, 2), padding='same') # 7x7
conv_3 = Conv2D(16, (3,3), activation='relu', padding='same') # 7x7
pool_3 = MaxPooling2D((2, 2), padding='same') # 4x4

flat_1 = Flatten()
fc_1 = Dense(64,activation='relu')

fc_2 = Dense(2,activation='linear')
fc_3 = Dense(2,activation='linear')

x = conv_1(input_img)
x = pool_1(x)
x = conv_2(x)
x = pool_2(x)
x = conv_3(x)
x = pool_3(x)
x = flat_1(x)
x = fc_1(x)

z_mean = fc_2(x)
z_log_var = fc_3(x)

z = Merge(mode = sampling, output_shape = (2,))([z_mean, z_log_var]) # 2,

concat_1 = keras.layers.Concatenate()
fc_4 = Dense(64,activation='relu',use_bias=False)
fc_5 = Dense(256,activation='relu')
reshape_1 = Reshape((16,4,4))

conv_8 = Conv2D(16, (3,3), activation='relu', padding='same') # 4x4
up_8 = UpSampling2D((2, 2)) # 8x8
conv_9 = Conv2D(8, (3,3), activation='relu', padding='same') # 8x8
up_9 = UpSampling2D((2, 2)) # 16x16
conv_10 = Conv2D(4, (3,3), activation='relu') # 14x14
up_10 = UpSampling2D((2, 2)) # 28x28
conv_11 = Conv2D(1, (3,3), activation='sigmoid', padding='same') # 28x28

x = keras.layers.Concatenate()([z,class_in])
x = fc_4(x)
x = fc_5(x)
x = reshape_1(x)
x = conv_8(x)
x = up_8(x)
x = conv_9(x)
x = up_9(x)
x = conv_10(x)
x = up_10(x)
decoded = conv_11(x)
    
vae = Model([input_img,class_in], decoded)

def vae_loss(_x, _x_decoded_mean):
    
    x = K.reshape(_x,(100,original_dim))
    x_decoded_mean = K.reshape(_x_decoded_mean,(100,original_dim))
    
    xent_loss = original_dim*(metrics.binary_crossentropy(x, x_decoded_mean))
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    
    return K.mean(xent_loss + kl_loss)
    
vae.compile(optimizer='adadelta', loss=vae_loss)

code_input = Input(shape=(2,))
y = concat_1([code_input,class_in])
y = fc_4(y)
y = fc_5(y)
y = reshape_1(y)
y = conv_8(y)
y = up_8(y)
y = conv_9(y)
y = up_9(y)
y = conv_10(y)
y = up_10(y)
decoded_2 = conv_11(y)
    
decoder = Model([code_input,class_in],decoded_2)

vae.summary()

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 1, 28, 28))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 1, 28, 28))

y_train_cat = keras.utils.np_utils.to_categorical(y_train,num_classes = 10)
y_test_cat = keras.utils.np_utils.to_categorical(y_test, num_classes = 10)
    
y_train_cat = y_train_cat.astype('float32')
y_test_cat = y_test_cat.astype('float32')

weights_file = 'checkpoint_weights.h5'
log_dir = '/tmp/logs'
callbacks =  [keras.callbacks.ModelCheckpoint(weights_file,save_best_only=True,save_weights_only=True), 
    keras.callbacks.TensorBoard(log_dir=log_dir, write_grads=True, write_graph=True,write_images=True,batch_size=batch_size), 
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.0001)]

vae.fit([x_train, y_train_cat],x_train,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=([x_test, y_test_cat], x_test),
        callbacks=callbacks)

for i in range(10):
    manifold_script.show_manifold(decoder,i)