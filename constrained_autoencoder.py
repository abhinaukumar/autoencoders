# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 11:13:44 2017

@author: nownow
"""

import keras
#import keras.backend as K
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Activation
from keras.models import Model
from keras.datasets import mnist
import numpy as np
#import tensorflow as tf
#import cv2

def load_dataset():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

#    y_train = y_train.astype('float32')
#    y_test = y_test.astype('float32')
    
    y_train_cat = keras.utils.np_utils.to_categorical(y_train,nb_classes = 10)
    y_test_cat = keras.utils.np_utils.to_categorical(y_test, nb_classes = 10)
        
    x_train = np.reshape(x_train, (len(x_train), 1, 28, 28))  # adapt this if using `channels_first` image data format
    x_test = np.reshape(x_test, (len(x_test), 1, 28, 28))  # adapt this if using `channels_first` image data format
    
#    y_combined_train = np.concatenate((y_train_cat, np.reshape(x_train, (len(x_train), 784))), axis = 1)
#    y_combined_test = np.concatenate((y_test_cat,np.reshape(x_test, (len(x_test), 784))), axis = 1)


    return (x_train, y_train_cat), (x_test, y_test_cat)
    
def encoder_model(input_img):
    conv_1 = Conv2D(nb_filter = 32, nb_row = 3, nb_col = 3, activation = 'relu', border_mode = 'same') (input_img)
    max_1 = MaxPooling2D(pool_size = (2, 2), border_mode = 'same') (conv_1)
    conv_2 = Conv2D(nb_filter = 16, nb_row = 3, nb_col = 3, activation = 'relu', border_mode = 'same') (max_1)   
    max_2 = MaxPooling2D(pool_size = (2,2), border_mode = 'same') (conv_2)
    conv_3 = Conv2D(nb_filter = 8, nb_row = 3, nb_col = 3, activation = 'relu', border_mode = 'same') (max_2)
    encoded = MaxPooling2D(pool_size = (2,2), border_mode = 'same') (conv_3)
    
    return encoded
    
def decoder_model(input_code):
    #input_code = Input(shape = (8, 4, 4))
    conv_1 = Conv2D(nb_filter = 8, nb_row = 3, nb_col = 3, activation = 'relu', border_mode = 'same') (input_code)
    up_1 = UpSampling2D(size = (2, 2)) (conv_1)
    conv_2 = Conv2D(nb_filter = 16, nb_row = 3, nb_col = 3, activation = 'relu', border_mode = 'same') (up_1)
    up_2 = UpSampling2D(size = (2,2)) (conv_2)
    conv_3 = Conv2D(nb_filter = 32  , nb_row = 3, nb_col = 3, activation = 'relu') (up_2)
    up_3 = UpSampling2D(size = (2,2)) (conv_3)
    decoded = Conv2D(nb_filter = 1, nb_row = 3, nb_col = 3, activation = 'sigmoid', border_mode = 'same') (up_3)
    
    return decoded

def classifier_model(input_code):
    #input_code = Input(shape = (8, 4, 4))
#    conv_1 = Conv2D(nb_filter = 16, nb_row = 3, nb_col = 3, activation = 'relu', border_mode = 'same') (input_code)
#    max_1 = MaxPooling2D(pool_size = (2,2), border_mode = 'same') (conv_1)
#    conv_2 = Conv2D(nb_filter = 32, nb_row = 3, nb_col = 3, activation = 'relu', border_mode = 'same') (max_1)
#    max_2 = MaxPooling2D(pool_size = (2,2), border_mode = 'same') (conv_2)
    flat_1 = Flatten()(input_code)
    fc_1 = Dense(output_dim = 10, activation = 'sigmoid') (flat_1)
    soft_classified = Activation(activation = 'softmax') (fc_1)
    
    return soft_classified
    
def loss_function(x,x_target):

    class_pred = x[:,:10]
    img_pred = x[:,10:]
    class_true = x_target[:,:10]
    img_true = x_target[:,10:]
    
    class_loss = keras.objectives.categorical_crossentropy(class_true,class_pred)
    rec_loss = keras.objectives.binary_crossentropy(img_true,img_pred)
    
    return class_loss + rec_loss

lmbda = 0.5

print("Loading dataset..")
(x_train,y_train), (x_test,y_test) = load_dataset()

print("Creating model..")
input_img = Input(shape = (1, 28, 28))
#input_class = Input(shape = (1,10))
encoder = encoder_model(input_img)
decoder = decoder_model(encoder)
classifier = classifier_model(encoder)

#decoder_flat = keras.layers.Reshape((784,))(decoder)
#model_output = keras.layers.merge([classifier, decoder_flat], mode='concat', concat_axis=1)

model = Model(input = [input_img]  , output = [classifier, decoder])

print("Compiling model..")
model.compile(optimizer = 'adadelta', loss = ['categorical_crossentropy', 'binary_crossentropy'], loss_weights = [lmbda, 1])

print("Model summary")
model.summary()

print("Training model..")
model.fit(x_train, [y_train, x_train], nb_epoch = 50, batch_size = 32, shuffle = True, validation_data = (x_test,[y_test, x_test]), verbose = 2)
#constrained_autoencoder = Model(input_img, [decoder,classifier])
#constrained_autoencoder.compile(optimizer = 'adadelta', loss = loss_function)
#
#encoder = Model(input_img, _encoder)
#decoder = Model(input_img, e
##_decoder = decoder_model()
#model = Sequential()
#model.add(encoder)
#model.add(decoder)
#decoder = Model(encoder,_decoder)
##_classifier = classifier_model()
##classifier = Model(encoder, _classifier)
##constr_autoencoder = Model(input_img, [classifier,autoencoder])