# autoencoders

This repo lists my attempts at exploring autoencoders, both dense and convolutional, for my independent project in summer of 2017.

The files in the repo are as follows:

autoencoder.py - Trains a convolutional autoencoder.
denoising_autoencoder.py - Trains a denoising autoencoder.
dense_vae.py - Trains a fully connected conditional variational autoencoder.
conv_vae.py - Trains a convolutional conditional variational autoencoder.

manifold_script.py provides two functions. One to display the 2D manifold learned by the model and another to generate images given an input string. 
main_file.py is the demo script that uses the above functions.

Requirements: Keras, preferably with TensorFlow backend, as the code might need slight changes with Theano.
"Channels first" ordering is followed in all models.
