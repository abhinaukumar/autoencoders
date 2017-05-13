# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 11:42:58 2017

@author: nownow
"""

import matplotlib as mpl

# This line allows mpl to run with no DISPLAY defined
mpl.use('Agg')

from keras.layers import Dense, Reshape, Flatten, Input, Merge
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.regularizers import l1, l1l2
import keras.backend as K
import pandas as pd
import numpy as np
from keras_adversarial.image_grid_callback import ImageGridCallback

from keras_adversarial import AdversarialModel, fix_names, n_choice
from keras_adversarial import AdversarialOptimizerSimultaneous, normal_latent_sampling
from keras.datasets import mnist
from keras.layers import LeakyReLU, Activation
import os


def model_generator(input_dim, input_shape, hidden_dim=256, reg=lambda: l1(1e-7)):
    z = Input(shape = (input_dim,))    
    decoder_h = Dense(hidden_dim, activation='relu')
    decoder_mean = Dense(input_dim, activation='sigmoid')

    h_decoded = decoder_h(z)
    x_decoded_mean = decoder_mean(h_decoded)
    
    return Model(z,x_decoded_mean, name="generator")

def model_encoder(latent_dim, input_shape, hidden_dim=256):
    x = Input(shape=(input_shape,))
    class_in = Input(shape=(10,))
    h = Dense(hidden_dim, activation='relu')(x)
    z_mean = Dense(latent_dim)(h)
    z_log_var = Dense(latent_dim)(h)
      
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(100, 2), mean=0.,
                              std=1.)
        return z_mean + K.exp(z_log_var / 2) * epsilon

    z = Merge(mode = sampling, output_shape = (latent_dim,))([z_mean, z_log_var])
    merge_stuff = Merge(mode = 'concat',concat_axis=1)
    z_final = merge_stuff([z,class_in])
    
    return Model([x,class_in], z_final, name="encoder")


def model_discriminator(input_dim, output_dim=1, hidden_dim=100):
    z = Input((input_dim,))
    h = z
    h = Dense(hidden_dim, name="discriminator_h1", activation = 'relu')(h)
    h = Dense(hidden_dim/10, name="discriminator_h2", activation = 'relu')(h)
    y = Dense(output_dim, name="discriminator_y", activation="sigmoid")(h)
    return Model(z, y)


def example_aae(path, adversarial_optimizer):
    # z \in R^100
    latent_dim = 2
    # x \in R^{28x28}
    input_shape = 784

    # generator (z -> x)
    generator = model_generator(latent_dim, input_shape)
    # encoder (x ->z)
    encoder = model_encoder(latent_dim, input_shape)
    # autoencoder (x -> x')
    autoencoder = Model(encoder.inputs, generator(encoder(encoder.inputs)))
    # discriminator (z -> y)
    discriminator = model_discriminator(latent_dim)

    # assemple AAE
    x = encoder.inputs[0]
    z = encoder(x)
    xpred = generator(z)
    zreal = normal_latent_sampling((latent_dim,))(x)
    yreal = discriminator(zreal)
    yfake = discriminator(z)
    aae = Model(x, fix_names([xpred, yfake, yreal], ["xpred", "yfake", "yreal"]))

    # print summary of models
    generator.summary()
    encoder.summary()
    discriminator.summary()
    autoencoder.summary()

    # build adversarial model
    generative_params = generator.trainable_weights + encoder.trainable_weights
    model = AdversarialModel(base_model=aae,
                             player_params=[generative_params, discriminator.trainable_weights],
                             player_names=["generator", "discriminator"])
    model.adversarial_compile(adversarial_optimizer=adversarial_optimizer,
                              player_optimizers=[Adam(1e-4, decay=1e-4), Adam(1e-3, decay=1e-4)],
                              loss={"yfake": "binary_crossentropy", "yreal": "binary_crossentropy",
                                    "xpred": "mean_squared_error"},
                              compile_kwargs={"loss_weights": {"yfake": 1e-2, "yreal": 1e-2, "xpred": 1}})

    # load mnist data
    xtrain, xtest = mnist_data()

    # callback for image grid of generated samples
    def generator_sampler():
        zsamples = np.random.normal(size=(10 * 10, latent_dim))
        return generator.predict(zsamples).reshape((10, 10, 28, 28))

    generator_cb = ImageGridCallback(os.path.join(path, "generated-epoch-{:03d}.png"), generator_sampler)

    # callback for image grid of autoencoded samples
    def autoencoder_sampler():
        xsamples = n_choice(xtest, 10)
        xrep = np.repeat(xsamples, 9, axis=0)
        xgen = autoencoder.predict(xrep).reshape((10, 9, 28, 28))
        xsamples = xsamples.reshape((10, 1, 28, 28))
        samples = np.concatenate((xsamples, xgen), axis=1)
        return samples

    autoencoder_cb = ImageGridCallback(os.path.join(path, "autoencoded-epoch-{:03d}.png"), autoencoder_sampler)

    # train network
    # generator, discriminator; pred, yfake, yreal
    n = xtrain.shape[0]
    y = [xtrain, np.ones((n, 1)), np.zeros((n, 1)), xtrain, np.zeros((n, 1)), np.ones((n, 1))]
    ntest = xtest.shape[0]
    ytest = [xtest, np.ones((ntest, 1)), np.zeros((ntest, 1)), xtest, np.zeros((ntest, 1)), np.ones((ntest, 1))]
    history = model.fit(x=xtrain, y=y, validation_data=(xtest, ytest), callbacks=[generator_cb, autoencoder_cb],
                        nb_epoch=100, batch_size=32)

    # save history
    df = pd.DataFrame(history.history)
    df.to_csv(os.path.join(path, "history.csv"))

    # save model
    encoder.save(os.path.join(path, "encoder.h5"))
    generator.save(os.path.join(path, "generator.h5"))
    discriminator.save(os.path.join(path, "discriminator.h5"))


def main():
    example_aae("output/aae", AdversarialOptimizerSimultaneous())


if __name__ == "__main__":
    main()