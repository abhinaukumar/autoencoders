# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 14:28:19 2017

@author: nownow
"""
'''This script demonstrates how to build a variational autoencoder
with Keras and deconvolution layers.
Reference: "Auto-Encoding Variational Bayes" https://arxiv.org/abs/1312.6114
'''
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D, Reshape, Flatten, Merge
from keras.models import Model

import numpy as np
import matplotlib.pyplot as plt

from keras import metrics
from keras.datasets import mnist

from scipy.stats import norm

import keras.backend as K
# input image dimensions
img_rows, img_cols, img_chns = 28, 28, 1
# number of convolutional filters to use
filters = 64
# convolution kernel size
num_conv = 3

batch_size = 100
original_img_size = (img_chns, img_rows, img_cols)

latent_dim = 2
intermediate_dim = 128
epsilon_std = 1.0
epochs = 5

input_img = Input(shape=(1, 28, 28))

x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(input_img)
x = MaxPooling2D((2, 2), border_mode='same')(x)
x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(x)
x = MaxPooling2D((2, 2), border_mode='same')(x)
x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
encoded = MaxPooling2D((2, 2), border_mode='same')(x)

flat = Flatten()(encoded)

z_mean = Dense(latent_dim)(flat)
z_log_var = Dense(latent_dim)(flat)


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim),
                              mean=0., std=epsilon_std)
    return z_mean + K.exp(z_log_var) * epsilon

# note that "output_shape" isn't necessary with the TensorFlow backend
# so you could write `Lambda(sampling)([z_mean, z_log_var])`

merge_1 = Merge(mode = sampling, output_shape=(latent_dim,))
fc_1 = Dense(8*4*4, activation='relu')
reshape_1 = Reshape((8,4,4))
conv_1 = Convolution2D(8, 3, 3, activation='relu', border_mode='same')
up_1 = UpSampling2D((2, 2))
conv_2 = Convolution2D(16, 3, 3, activation='relu', border_mode='same')
up_2 = UpSampling2D((2, 2))
conv_3 = Convolution2D(32, 3, 3, activation='relu')
up_3 = UpSampling2D((2, 2))
conv_4 = Convolution2D(1, 3, 3, activation='sigmoid', border_mode='same')

z = merge_1([z_mean, z_log_var])

y = fc_1(z)
y = reshape_1(y)

y = conv_1(y)
y = up_1(y)
y = conv_2(y)
y = up_2(y)
y = conv_3(y)
y = up_3(y)
decoded = conv_4(y)


# we instantiate these layers separately so as to reuse them later
#decoder_hid = Dense(intermediate_dim, activation='relu')
#decoder_upsample = Dense(filters * 14 * 14, activation='relu')
#
#output_shape = (batch_size, filters, 14, 14)
#
#decoder_reshape = Reshape(output_shape[1:])
#decoder_deconv_1 = Deconv2D(filters,
#                                   num_conv,num_conv,
#                                   border_mode='same',
#                                   subsample=(1,1),
#                                   activation='relu')
#decoder_deconv_2 = Deconv2D(filters, num_conv, num_conv,
#                                   border_mode='same',
#                                   subsample=(1,1),
#                                   activation='relu')
#output_shape = (batch_size, filters, 29, 29)
#
#decoder_deconv_3_upsamp = Deconv2D(filters,
#                                          3, 3,
#                                          subsample=(2, 2),
#                                          border_modeg='valid',
#                                          activation='relu')
#decoder_mean_squash = Deconv2D(img_chns,
#                             2,2,
#                             border_mode='valid',
#                             activation='sigmoid')

#hid_decoded = decoder_hid(z)
#up_decoded = decoder_upsample(hid_decoded)
#reshape_decoded = decoder_reshape(up_decoded)
#deconv_1_decoded = decoder_deconv_1(reshape_decoded)
#deconv_2_decoded = decoder_deconv_2(deconv_1_decoded)
#x_decoded_relu = decoder_deconv_3_upsamp(deconv_2_decoded)
#x_decoded_mean_squash = decoder_mean_squash(x_decoded_relu)


# Custom loss layer
def vae_loss(x, x_decoded_mean_squash):
    x = K.flatten(x)
    x_decoded_mean_squash = K.flatten(x_decoded_mean_squash)
    xent_loss = img_rows * img_cols * metrics.binary_crossentropy(x, x_decoded_mean_squash)
    kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return K.mean(xent_loss + kl_loss)

#    def call(self, inputs):
#        x = inputs[0]
#        x_decoded_mean_squash = inputs[1]
#        loss = self.vae_loss(x, x_decoded_mean_squash)
#        self.add_loss(loss, inputs=inputs)
#        # We don't use this output.
#        return x
#

#y = CustomVariationalLayer()([x, x_decoded_mean_squash])
vae = Model(input_img, decoded)
vae.compile(optimizer='rmsprop', loss=vae_loss)
vae.summary()


(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 1, 28, 28))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 1, 28, 28))  # adapt this if using `channels_first` image data format

print("Read dataset");


vae.fit(x_train, x_train, nb_epoch=150, batch_size=batch_size, shuffle=True, validation_data=(x_test, x_test))

# build a model to project inputs on the latent space
encoder = Model(input_img, z_mean)

# display a 2D plot of the digit classes in the latent space
x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
plt.figure(figsize=(6, 6))
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
plt.colorbar()
plt.show()

# build a digit generator that can sample from the learned distribution
code_input = Input(shape=(latent_dim,))
y_dec = fc_1(code_input)
y_dec = reshape_1(y_dec)
y_dec = conv_1(y_dec)
y_dec = up_1(y_dec)
y_dec = conv_2(y_dec)
y_dec = up_2(y_dec)
y_dec = conv_3(y_dec)
y_dec = up_3(y_dec)
y_dec = conv_4(y_dec)

generator = Model(code_input, y_dec)

# display a 2D manifold of the digits
n = 15  # figure with 15x15 digits
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))
# linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
# to produce values of the latent variables z, since the prior of the latent space is Gaussian
grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi, 0, 0]])
        z_sample = np.tile(z_sample, batch_size).reshape(batch_size, 4)
        x_decoded = generator.predict(z_sample, batch_size=batch_size)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='Greys_r')
plt.show()