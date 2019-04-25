import warnings
warnings.filterwarnings('ignore')

import os
import sys
import time
import logging
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.manifold import TSNE

import keras
from keras import backend as K
from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras.utils import plot_model

# Utils
# file_path = os.getcwd()
file_path = os.path.dirname(os.path.relpath(__file__))
utils_path = os.path.abspath(os.path.join(file_path, 'utils_py'))
sys.path.append(utils_path)
import utils_all
import utils

DATADIR = './tidy_data_from_combined'
FILENAME = 'tidy_data.prqt'
OUTDIR = os.path.join(file_path, 'rnaseq_reduce_dim')
os.makedirs(OUTDIR, exist_ok=True)

SEED = 0


def build_autoencoder(input_dim, latent_dim, layers_dim):
    """ Symmetric dense autoencoder.
    Args:
        input_dim (int) :
        latent_dim (int) :
        layers_dim : list of integers where each consecutive value is the dim of the
                     next hidden layer in the encoder

    Returns:
        autoencoder : keras model of autoencoder
        encoder : keras model of encoder
        decoder :keras model of decoder
    """
    assert input_dim >= layers_dim[0], "input_dim must not be smaller than the first encoder layer."
    assert latent_dim <= layers_dim[-1], "latent_dim must not be bigger than the last encoder layer."

    # Define input layers for the encoder and decoder
    encoder_input = Input(shape=(input_dim,), name='encoder_input')
    decoder_input = Input(shape=(latent_dim,), name='decoder_input')

    # Define encoder
    enc = encoder_input
    for i, dim in enumerate(layers_dim):
        # print(f'encoder later {i+1}: {dim}')
        enc = Dense(dim, activation='relu', name='encoder_layer_'+str(i+1))(enc)

    # Latent layer
    z = Dense(latent_dim, activation='relu', name='latent_layer')(enc)

    # Define decoder
    dec = z
    for i, dim in enumerate(layers_dim[::-1]):
        # print(f'decoder later {i+1}: {dim}')
        dec = Dense(dim, activation='relu', name='decoder_layer_'+str(i+1))(dec)

    # Define decoder output
    dec = Dense(input_dim, activation='sigmoid', name='decoder_output')(dec)

    # Autonecoder
    autoencoder = Model(encoder_input, dec)
    # autoencoder.summary()

    # Encoder
    encoder = Model(encoder_input, z)
    # encoder.summary()

    # Decoder
    # https://stackoverflow.com/questions/37758496/python-keras-theano-wrong-dimensions-for-deep-autoencoder
    x = decoder_input
    for i in range(len(layers_dim)+1, 0, -1):
        # print(i)
        x = autoencoder.layers[-i](x)
    decoder = Model(decoder_input, x)
    # decoder.summary()

    return autoencoder, encoder, decoder


# ========================================================================
#       Load data and pre-proc
# ========================================================================
datapath = os.path.join(DATADIR, FILENAME)
print('\nLoad tidy data ... {}'.format(datapath))
data = pd.read_parquet(datapath, engine='auto', columns=None)
train_sources = ['ccle']  # ['ccle', 'gcsi', 'gdsc', 'ctrp']
data = data[data['SOURCE'].isin(train_sources)].reset_index(drop=True)

rnaseq_prefix = 'cell_rna'
xdata = data[[c for c in data.columns if c.split('.')[0] in rnaseq_prefix]].reset_index(drop=True).copy()

# Normalize data
col_names = xdata.columns.tolist()
scaler = StandardScaler()
xdata = scaler.fit_transform(xdata)
xdata = pd.DataFrame(xdata, columns=col_names)

# Split
xtr, xte = train_test_split(xdata, test_size=0.2)
xtr.reset_index(drop=True, inplace=True)
xte.reset_index(drop=True, inplace=True)
print(xtr.shape)
print(xte.shape)

# Build autoencoder
input_dim = xdata.shape[1]  # 784
latent_dim = 64
layers_dim = [512, 256, 128]
autoencoder, encoder, decoder = build_autoencoder(input_dim=input_dim,
                                                  latent_dim=latent_dim, layers_dim=layers_dim)

# Train
autoencoder.compile(optimizer='rmsprop', loss='mean_squared_error')
history = autoencoder.fit(xtr, xtr,
                          epochs=50,
                          batch_size=256,
                          shuffle=True,
                          validation_data=(xte, xte))

utils_all.plot_keras_learning(history, figsize=(10,6),
                              img_name=os.path.join(OUTDIR, 'au_rnaseq_learning_curve.png'))

# Encode and decode some samples
encoded_samples = encoder.predict(xte)
decoded_samples = decoder.predict(encoded_samples)

# https://stackoverflow.com/questions/36886711/keras-runtimeerror-failed-to-import-pydot-after-installing-graphviz-and-pyd
plot_model(model=autoencoder, show_shapes=True, to_file=os.path.join(OUTDIR, 'au_rnaseq_model.png'))



# ================================================================================
#       MNIST example
# ================================================================================
# (x_train, _), (x_test, _) = mnist.load_data()
# x_train = x_train.astype('float32') / 255.
# x_test = x_test.astype('float32') / 255.
# x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
# x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
# print(x_train.shape)
# print(x_test.shape)

# input_dim = 784
# latent_dim = 32
# layers_dim = [256, 128, 64]

# # Define input layers for the encoder and decoder
# encoder_input = Input(shape=(input_dim,), name='encoder_input')
# decoder_input = Input(shape=(latent_dim,), name='decoder_input')

# # Define encoder
# enc = encoder_input
# for i, dim in enumerate(layers_dim):
#     print(f'encoder later {i+1}: {dim}')
#     enc = Dense(dim, activation='relu', name='enc_layer_'+str(i+1))(enc)

# # Latent layer
# z = Dense(latent_dim, activation='relu', name='latent_layer')(enc)

# # Define decoder
# dec = z
# for i, dim in enumerate(layers_dim[::-1]):
#     print(f'decoder later {i+1}: {dim}')
#     dec = Dense(dim, activation='relu', name='dec_layer_'+str(i+1))(dec)

# # Define decoder output
# dec = Dense(input_dim, activation='sigmoid', name='dec_output')(dec)

# # Autonecoder
# autoencoder = Model(encoder_input, dec)
# autoencoder.summary()

# # Encoder
# encoder = Model(encoder_input, z)
# encoder.summary()

# # Decoder
# # https://stackoverflow.com/questions/37758496/python-keras-theano-wrong-dimensions-for-deep-autoencoder
# x = decoder_input
# for i in range(len(layers_dim)+1, 0, -1):
#     print(i)
#     x = autoencoder.layers[-i](x)
# decoder = Model(decoder_input, x)
# decoder.summary()

# autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
# history = autoencoder.fit(x_train, x_train,
#                           epochs=50,
#                           batch_size=256,
#                           shuffle=True,
#                           validation_data=(x_test, x_test))

# plot_model(model=autoencoder, to_file=os.path.join(OUTDIR, 'au_mnist_model.png'))

# utils_all.plot_keras_learning(history, img_name=os.path.join(OUTDIR, 'au_mnist_learning_curve.png'))

# # Encode and decode some digits
# encoded_imgs = encoder.predict(x_test)
# decoded_imgs = decoder.predict(encoded_imgs)

# n = 10  # number of digits to display
# plt.figure(figsize=(20, 4))
# for i in range(n):
#     # display original
#     ax = plt.subplot(2, n, i + 1)
#     plt.imshow(x_test[i].reshape(28, 28))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)

#     # display reconstruction
#     ax = plt.subplot(2, n, i + 1 + n)
#     plt.imshow(decoded_imgs[i].reshape(28, 28))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
# plt.show()
# plt.savefig(os.path.join(OUTDIR, 'au_mnist_recons.png'))
# ================================================================================
