# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 21:05:20 2017

@author: nownow
"""

import keras
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Merge, Reshape
from keras.layers import concatenate
from keras.models import Model,Sequential
import numpy as np
from random import shuffle
from keras.callbacks import TensorBoard

from keras.datasets import mnist

import matplotlib.pyplot as plt
import os

from keras import backend as K
#import tensorflow as tf
import numpy as np
#from PIL import Image
#import cv2

from keras.datasets import mnist
import sklearn

plt.ion()
from scipy.stats import norm
from keras import metrics


batch_size = 100
original_dim = 784
latent_dim = 2
intermediate_dim = 256
epochs = 3
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
#conv_4 = Conv2D(8, (3,3), activation='relu', padding='same') # 4x4
#pool_4 = MaxPooling2D((2, 2), padding='same') # 2x2
#conv_5 = Conv2D(4, (3,3), activation='relu', padding='same') # 2x2
#pool_5 = MaxPooling2D((2, 2), padding='same') # 1x1

#conv_5_1 = Conv2D(2, (3,3), activation='relu', padding='same') # 2x2x2
#pool_5_1 = MaxPooling2D((2, 2), padding='same') #2x1x1
#conv_5_2 = Conv2D(2, (3,3), activation='relu', padding='same') # 2x2x2
#pool_5_2 = MaxPooling2D((2, 2), padding='same') # 2x1x1
#conv_5 = Conv2D(2, (3,3), activation='relu', padding='same') # 2x2x2
#pool_5 = MaxPooling2D((2, 2), padding='same') #2x1
flat_1 = Flatten()
#flat_2 = Flatten()
fc_1 = Dense(64,activation='relu')
#fc_2 = Dense(4,activation='relu')
fc_2 = Dense(2,activation='linear')
fc_3 = Dense(2,activation='linear')

x = conv_1(input_img)
x = pool_1(x)
x = conv_2(x)
x = pool_2(x)
x = conv_3(x)
x = pool_3(x)
#x = conv_4(x)
#x = pool_4(x)
#x = conv_5(x)
#x = pool_5(x)
x = flat_1(x)
x = fc_1(x)
#x = fc_2(x)
#x = keras.layers.Concatenate()([x,class_in])
#_z_mean = conv_5_1(x)
#z_mean = pool_5_1(_z_mean)
#_z_log_var = conv_5_2(x)
#z_log_var = pool_5_2(_z_log_var)

z_mean = fc_2(x)
z_log_var = fc_3(x)

z = Merge(mode = sampling, output_shape = (2,))([z_mean, z_log_var]) # 2,

#encoder_model= Model(x,encoded)

concat_1 = keras.layers.Concatenate()
fc_4 = Dense(64,activation='relu',use_bias=False)
fc_5 = Dense(256,activation='relu')
#fc_7 = Dense(16,activation='relu')
reshape_1 = Reshape((16,4,4))
#conv_6 = Conv2D(4, (3,3), activation='relu', padding='same') # 1x1
#up_6 = UpSampling2D((2, 2)) # 2x2

#conv_7 = Conv2D(8, (3,3), activation='relu', padding='same') # 2x2
#up_7 = UpSampling2D((2, 2)) # 4x4
conv_8 = Conv2D(16, (3,3), activation='relu', padding='same') # 4x4
up_8 = UpSampling2D((2, 2)) # 8x8
conv_9 = Conv2D(8, (3,3), activation='relu', padding='same') # 8x8
up_9 = UpSampling2D((2, 2)) # 16x16
conv_10 = Conv2D(4, (3,3), activation='relu') # 14x14
up_10 = UpSampling2D((2, 2)) # 28x28
conv_11 = Conv2D(1, (3,3), activation='sigmoid', padding='same') # 28x28

#x = conv_6(z)
#x = up_6(x)
x = keras.layers.Concatenate()([z,class_in])
x = fc_4(x)
x = fc_5(x)
x = reshape_1(x)
#x = conv_6(x)
#x = up_6(x)
#x = conv_7(x)
#x = up_7(x)
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
    #return K.mean(kl_loss)
    
vae.load_weights('/home/nownow/Documents/projects/idp_summer_2017/saved_model/conv_vae_2_final_weights_2.h5')

#opt = keras.optimizers.SGD(momentum=0.9)
vae.compile(optimizer='adadelta', loss=vae_loss)

#encoder= Model(input_img,encoded)
code_input = Input(shape=(2,))
y = concat_1([code_input,class_in])
#y = conv_6(code_input)
#y = up_6(y)
y = fc_4(y)
y = fc_5(y)
y = reshape_1(y)
#y = conv_6(y)
#y = up_6(y)
#y = conv_7(y)
#y = up_7(y)
y = conv_8(y)
y = up_8(y)
y = conv_9(y)
y = up_9(y)
y = conv_10(y)
y = up_10(y)
decoded_2 = conv_11(y)
    
decoder = Model([code_input,class_in],decoded_2)
#decoder.compile(optimizer='adadelta', loss='binary_crossentropy')

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

#vae.fit([x_train,y_train_cat], x_train, epochs=5, validation_data=([x_test,y_test_cat],x_test), batch_size=1000)

#vae.load_weights('/home/nownow/Documents/projects/idp_summer_2017/saved_model/conv_vae_2_21_12_Jul_4_best.h5')

a = "0123456789"    
vector = np.random.normal(size = [len(a),latent_dim])
class_input = np.zeros(shape=[len(a),10])
for i in range(len(a)):
    class_input[i][int(a[i])] = 1    

pred1 = decoder.predict([vector,class_input])
pred2 = np.reshape(pred1,(len(a)*28,28))
plt.imsave('init_op.png',pred2, cmap = 'Greys_r')
    
#for i in range(epochs):
#    print("Epoch: ",i)
#    vae.fit([x_train, y_train_cat],x_train,
#        shuffle=True,
#        epochs=1,
#        batch_size=batch_size,
#        validation_data=([x_test, y_test_cat], x_test))
#    a = "0123456789"    
#    vector = np.random.normal(size = [len(a),latent_dim])
#    class_input = np.zeros(shape=[len(a),10])
#    for i in range(len(a)):
#        class_input[i][int(a[i])] = 1    
#    
#    pred1 = decoder.predict([vector,class_input])
#    pred2 = np.reshape(pred1,(len(a)*28,28))
#    plt.imsave('epoch_op.png',pred2, cmap = 'Greys_r')
#vae.load_weights('/home/nownow/Documents/projects/idp_summer_2017/saved_model/conv_vae_2_best_weights.h5')


#callbacks =  [keras.callbacks.ModelCheckpoint('/home/nownow/Documents/projects/idp_summer_2017/saved_model/conv_vae_2_11_43_Jul_11_best.h5',save_best_only=True,save_weights_only=True), 
#    keras.callbacks.TensorBoard(log_dir='/tmp/conv_vae_logs', write_grads=True, write_graph=True,write_images=True,batch_size=batch_size), 
#    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.0001)]
vae.fit([x_train, y_train_cat],x_train,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=([x_test, y_test_cat], x_test))#,callbacks=callbacks)

#vae.save_weights('conv_vae_22_45_Jun_30_final.h5')
n = 15  # figure with 15x15 digits
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))
# linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
# to produce values of the latent variables z, since the prior of the latent space is Gaussian
grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))
for k in range(10):
    print k
    class_vec = np.zeros((10,1))
    class_vec[k] = 1
    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict([z_sample, np.transpose(class_vec)])
            digit = x_decoded[0]
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit              
    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='Greys_r')
    plt.show()

a = "0123456789"    
vector = np.random.normal(size = [len(a),latent_dim])
class_input = np.zeros(shape=[len(a),10])
for i in range(len(a)):
    class_input[i][int(a[i])] = 1    

pred1 = decoder.predict([vector,class_input])
pred2 = np.reshape(pred1,(len(a)*28,28))
plt.imsave('epoch_op.png',pred2, cmap = 'Greys_r')