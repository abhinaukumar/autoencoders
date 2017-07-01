# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 15:28:27 2017

@author: nownow
"""

import keras

import manifold_script
a = str(input("Enter number: "))

path_to_file = ""

generator = keras.models.load_model(path_to_file)
manifold_script.show_digits(generator,a)

choice = raw_input("Do you wish to see the manifold? (y/n): ")
if choice == 'y':
    manifold_script.show_manifold(generator)
    
print("Exiting..")
