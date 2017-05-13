# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 13:44:25 2017

@author: nownow
"""

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
    class_vec[k] = 1
    mu = dist_mean_model.predict(class_vec)
    sig = K.exp(dist_log_var_model.predict(class_vec))
    grid_x = norm.ppf(np.linspace(0.05, 0.95, n), loc = mu[0][0], scale = sig[0][0])
    grid_y = norm.ppf(np.linspace(0.05, 0.95, n), loc = mu[0][1], scale = sig[0][1])
    
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
