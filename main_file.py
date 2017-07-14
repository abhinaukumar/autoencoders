# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 15:28:27 2017

@author: nownow
"""

import keras

import manifold_script
a = str(raw_input("Enter number: "))

path_to_file = '/home/nownow/Documents/projects/idp_summer_2017/saved_model/conv_decoder_2_best.json'
generator = keras.models.load_model(path_to_file)
manifold_script.show_digits(generator,a)

choice = raw_input("Do you wish to see the manifold? (y/n): ")
if choice == 'y':
    for i in range(10):
        manifold_script.show_manifold(generator,i)
    
print("Exiting..")
