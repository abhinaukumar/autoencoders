# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 13:44:25 2017

@author: nownow
"""
import numpy as np
import matplotlib.pyplot as plt
plt.ion()
from scipy.stats import norm

def show_manifold(generator):
    n = 15  # figure with 15x15 digits
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
    # to produce values of the latent variables z, since the prior of the latent space is Gaussian
    
    for k in range(10):
        class_vec = np.zeros(shape = [1,10])

        class_vec[0,k] = 1
    
        grid_x = norm.ppf(np.linspace(0.05, 0.95, n), loc = 0 , scale = 1)
        grid_y = norm.ppf(np.linspace(0.05, 0.95, n), loc = 0, scale = 1)
            
        for i, yi in enumerate(grid_x):
            for j, xi in enumerate(grid_y):
                z_sample = np.array([[xi, yi]])

                x_decoded = generator.predict([z_sample,class_vec])
                digit = x_decoded[0].reshape(digit_size, digit_size)
                figure[i * digit_size: (i + 1) * digit_size,
                       j * digit_size: (j + 1) * digit_size] = digit              
        plt.figure(figsize=(10, 10))
        plt.imshow(figure, cmap='Greys_r')
        plt.show()

def show_digits(generator,a):
    latent_dim = generator.input_shape[0][1]
    vector = np.random.normal(size = [len(a),latent_dim])
    class_input = np.zeros(shape=[len(a),10])
    for i in range(len(a)):
        class_input[i][int(a[i])] = 1    
    
    pred1 = generator.predict([vector,class_input])
    pred2 = np.reshape(pred1,(len(a)*28,28))
    plt.imshow(pred2, cmap = plt.cm.gray)
    plt.show()