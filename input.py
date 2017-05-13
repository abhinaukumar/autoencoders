# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 15:28:27 2017

@author: nownow
"""

import numpy as np
import keras
import json
import matplotlib.pyplot as plt
from scipy.stats import norm
a = str(input("Enter number: "))
vector = np.random.normal(size=[len(a),4])
class_input = np.zeros(shape=[len(a),10])
for i in range(len(a)):
    class_input[i][int(a[i])] = 1

with open('/home/nownow/Documents/projects/idp_summer_2017/dense_gen_3_arch.txt') as outfile:
    gen_json = json.load(outfile)
    generator1= keras.models.model_from_json(gen_json)

generator1.load_weights('/home/nownow/Documents/projects/idp_summer_2017/dense_gen_3_weights.h5')

#with open('/home/nownow/Documents/projects/idp_summer_2017/dense_gen_4_arch.txt') as outfile:
#    gen_json = json.load(outfile)
#    generator2= keras.models.model_from_json(gen_json)
#
#generator2.load_weights('/home/nownow/Documents/projects/idp_summer_2017/dense_gen_4_weights.h5')

pred1 = generator1.predict([vector[:,:2],class_input])
#pred2 = generator2.predict([vector,class_input])

pred2 = np.reshape(pred1,(len(a)*28,28))
#plt.subplot(len(a),1)
#for i in range(1,len(a)+1):
#    plt.subplot(1,len(a),i)
#    plt.imshow(np.reshape(pred1[i-1],(28,28)))
#    plt.gray()
#plt.show() 
plt.imshow(pred2)
plt.gray()
plt.show()

choice = input("Do you wish to see the manifold? (y/n): ")
if choice == 'y':
    n = 15  # figure with 15x15 digits
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
    # to produce values of the latent variables z, since the prior of the latent space is Gaussian
    grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
    grid_y = norm.ppf(np.linspace(0.05, 0.95, n))
    for k in range(10):
        class_vec = np.zeros((10,1))
        class_vec[k] = 1
        for i, yi in enumerate(grid_x):
            for j, xi in enumerate(grid_y):
                z_sample = np.array([[xi, yi]])
                x_decoded = generator1.predict([z_sample, np.transpose(class_vec)])
                digit = x_decoded[0].reshape(digit_size, digit_size)
                figure[i * digit_size: (i + 1) * digit_size,
                       j * digit_size: (j + 1) * digit_size] = digit              
        plt.figure(figsize=(10, 10))
        plt.imshow(figure, cmap='Greys_r')
        plt.show()
print("Exiting..")