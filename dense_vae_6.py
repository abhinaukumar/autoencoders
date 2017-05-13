# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 12:42:00 2017

@author: nownow
"""

import numpy as np
import keras
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Dense, Merge
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.datasets import mnist

batch_size = 100
original_dim = 784
latent_dim = 2
intermediate_dim = 256
epochs = 50
epsilon_std = 1.0


x = Input(shape=(original_dim,))
h = Dense(intermediate_dim, activation='relu')(x)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)
#class_est = Dense(10, activation = 'sigmoid')(h)

#class SamplingLayer(Layer):
#    def __init__(self,**kwargs):
#        self.is_placeholder = True
#        super(SamplingLayer, self).__init__(**kwargs)
#    
#    
#    def call(self, input_tensors, mask):
#        epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.,
#                              std=epsilon_std)
#        return input_tensors[0] + K.exp(input_tenos / 2) * epsilon
        
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(100, 2), mean=0.,
                              std=1.)
    return z_mean + K.exp(z_log_var / 2) * epsilon
#
#def merge_routine(args):
#    code, class_vec = args
#    rep_code = K.tile(code,10)
#    class_vec = np.repeat(class_vec,2,axis=1)
#    #class_vec = K.reshape(class_vec,(20,))
##    out_code = K.zeros((100,20))
##    for i in range(10):
##        out_code[:,2*i] = rep_code[:,2*i]*class_vec[i]
##        out_code[:,2*i+1] = rep_code[:,2*i+1]*class_vec[i]
#    return rep_code*class_vec
# note that "output_shape" isn't necessary with the TensorFlow backend
class_in = Input(shape=(10,))
merge_stuff = Merge(mode = 'concat',concat_axis=1)
#merge_stuff = Merge(mode = 'concat', concat_axis=1)
dist_mean_pred = Dense(2)
dist_log_var_pred = Dense(2)

dist_mean = dist_mean_pred(class_in)
dist_log_var = dist_log_var_pred(class_in)

net_mean = keras.layers.merge([z_mean, dist_mean], mode = 'sum')
net_log_var = keras.layers.merge([z_log_var, dist_log_var], mode = 'sum')
z = Merge(mode = sampling, output_shape = (latent_dim,))([net_mean, net_log_var])
#z = Merge(mode = 'concat', concat_axis=1)([z, class_in])
# we instantiate these layers separately so as to reuse them later
decoder_h = Dense(intermediate_dim, activation='relu')
decoder_mean = Dense(original_dim, activation='sigmoid')
#class_code = Dense(10, activation = 'sigmoid')

#class_in = class_code(class_est)
#z = merge_stuff([z,class_in])
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)


# Custom loss layer
#class CustomVariationalLayer(Layer):
#    def __init__(self, **kwargs):
#        self.is_placeholder = True
#        super(CustomVariationalLayer, self).__init__(**kwargs)

def vae_loss(x, x_decoded_mean):
    xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return K.mean(xent_loss + kl_loss)

#    def call(self, inputs):
#        x = inputs[0]
#        x_decoded_mean = inputs[1]
#        loss = self.vae_loss(x, x_decoded_mean)
#        self.add_loss(loss, inputs=inputs)
#        # We won't actually use the output.
#        return x

#y = CustomVariationalLayer()([x, x_decoded_mean])
vae = Model(input = [x, class_in], output = x_decoded_mean)
vae.compile(optimizer='adadelta', loss=vae_loss)

vae.summary()

# train the VAE on MNIST digits
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

y_train_cat = keras.utils.np_utils.to_categorical(y_train,nb_classes = 10)
y_test_cat = keras.utils.np_utils.to_categorical(y_test, nb_classes = 10)
    
y_train_cat = y_train_cat.astype('float32')
y_test_cat = y_test_cat.astype('float32')

#y_train = y_train.astype('float32')
#y_test = y_test.astype('float32')

vae.fit([x_train, y_train_cat],x_train,
        shuffle=True,
        nb_epoch=100,
        batch_size=batch_size,
        validation_data=([x_test, y_test_cat], x_test))

# build a model to project inputs on the latent space
#encoder = Model(x, z_mean)
#
## display a 2D plot of the digit classes in the latent space
#x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
#plt.figure(figsize=(6, 6))
#plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
#plt.colorbar()
#plt.show()
#
## build a digit generator that can sample from the learned distribution

class_input=  Input(shape = (10,))

_dist_mean = dist_mean_pred(class_input)
_dist_log_var = dist_log_var_pred(class_input)

dist_mean_model = Model(class_input, _dist_mean)
dist_log_var_model = Model(class_input, _dist_log_var)
#code = Merge
#
decoder_input = Input(shape=(latent_dim,))
#_merge_decoded = keras.layers.merge([decoder_input, class_input], mode = 'concat', concat_axis=1)
_h_decoded = decoder_h(decoder_input)
_x_decoded_mean = decoder_mean(_h_decoded)
generator = Model(input = [decoder_input, class_input], output = _x_decoded_mean)
#
## display a 2D manifold of the digits
n = 15  # figure with 15x15 digits
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))
# linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
# to produce values of the latent variables z, since the prior of the latent space is Gaussian
#grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
#grid_y = norm.ppf(np.linspace(0.05, 0.95, n))
for k in range(10):
    class_vec = np.zeros((1,10))
    class_vec[0,k] = 1
    mu = dist_mean_model.predict(class_vec)
    sig = np.exp(dist_log_var_model.predict(class_vec))
    grid_x = norm.ppf(np.linspace(0.05, 0.95, n), loc = mu[0,0], scale = sig[0,0])
    grid_y = norm.ppf(np.linspace(0.05, 0.95, n), loc = mu[0,1], scale = sig[0,1])
    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = np.array([[xi, yi]])
            x_decoded = generator.predict([z_sample, class_vec])
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit              
    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='Greys_r')
    plt.show()

##plt.figure(figsize=(10, 10))
##plt.imshow(figure, cmap='Greys_r')
##plt.show()