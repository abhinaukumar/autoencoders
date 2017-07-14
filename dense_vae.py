'''This script demonstrates how to build a variational autoencoder with Keras.
Reference: "Auto-Encoding Variational Bayes" https://arxiv.org/abs/1312.6114
'''
import numpy as np
import keras
import matplotlib.pyplot as plt
plt.ion()

import manifold_script

from keras.layers import Input, Dense, Merge
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.datasets import mnist

from keras.callbacks import TensorBoard

batch_size = 100
original_dim = 784
latent_dim = 2
intermediate_dim = 256
epochs = 10
epsilon_std = 1.0


x = Input(shape=(original_dim,))
h = Dense(intermediate_dim, activation='relu')(x)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)

     
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(100, 2), mean=0.,
                              stddev=1.)
    return z_mean + K.exp(z_log_var / 2) * epsilon


# note that "output_shape" isn't necessary with the TensorFlow backend
z = Merge(mode = sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
class_in = Input(shape=(10,))

# we instantiate these layers separately so as to reuse them later
merge_stuff = Merge(mode = 'concat',concat_axis=1)

decoder_h = Dense(intermediate_dim, activation='relu')
decoder_mean = Dense(original_dim, activation='sigmoid')

z = merge_stuff([z,class_in])
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)

def vae_loss(x, x_decoded_mean):
    
    xent_loss = original_dim*(metrics.mean_squared_error(x, x_decoded_mean))
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return K.mean(xent_loss + kl_loss)
    
vae = Model(input = [x, class_in], output = x_decoded_mean)
vae.compile(optimizer='adadelta', loss=vae_loss)

vae.summary()

# train the VAE on MNIST digits
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

y_train_cat = keras.utils.np_utils.to_categorical(y_train,num_classes = 10)
y_test_cat = keras.utils.np_utils.to_categorical(y_test, num_classes = 10)
    
y_train_cat = y_train_cat.astype('float32')
y_test_cat = y_test_cat.astype('float32')

# build a model to project inputs on the latent space
encoder = Model(x, z_mean)

# build a digit generator that can sample from the learned distribution
decoder_input = Input(shape=(latent_dim,))
class_input=  Input(shape = (10,))

_merge_decoded = keras.layers.merge([decoder_input, class_input], mode = 'concat', concat_axis=1)
_h_decoded = decoder_h(_merge_decoded)
_x_decoded_mean = decoder_mean(_h_decoded)
generator = Model(input = [decoder_input, class_input], output = _x_decoded_mean)

vae.fit([x_train, y_train_cat],x_train,
        shuffle=True,
        epochs=10,
        batch_size=batch_size,
        validation_data=([x_test, y_test_cat], x_test),callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

for i in range(10):
    manifold_script.show_manifold(generator,i)