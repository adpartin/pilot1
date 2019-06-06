"""
This script contains various ML models and some utility functions.
"""
import warnings
warnings.filterwarnings('ignore')

import os
from pathlib import Path
from time import time
from collections import OrderedDict

import sklearn
import numpy as np
import pandas as pd

import matplotlib
# matplotlib.use('TkAgg')
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib

try:
    import lightgbm as lgb
except ImportError:
    print('Module not found (lightgbm).')

import tensorflow as tf
import keras
from keras import backend as K
from keras.layers import Input, Dense, Dropout, Activation, BatchNormalization
from keras import optimizers
from keras.optimizers import SGD, Adam, RMSprop, Adadelta
from keras.models import Sequential, Model, model_from_json, model_from_yaml
from keras.utils import np_utils, multi_gpu_model
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard


def r2_krs(y_true, y_pred):
    from keras import backend as K
    SS_res =  K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res/(SS_tot + K.epsilon()))


def nn_model1(input_dim, dr_rate=0.2, opt_name='sgd', logger=None):
    """ ... """
    inputs = Input(shape=(input_dim,))
    a = Dense(1000, activation='relu')(inputs)
    b = Dense(1000, activation='softmax')(inputs)
    x = keras.layers.multiply( [a, b] )
            
    x = Dense(1000, activation='relu')(x)
    x = Dropout(dr_rate)(x)
        
    x = Dense(500, activation='relu')(x)
    x = Dropout(dr_rate)(x)
    
    x = Dense(250, activation='relu')(x)
    x = Dropout(dr_rate)(x)
        
    x = Dense(60, activation='relu')(x)
    x = Dropout(dr_rate)(x)
        
    outputs = Dense(1, activation='relu')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.summary()
        
    if opt_name == 'sgd':
        opt = SGD(lr=1e-4, momentum=0.9)
    elif opt_name == 'adam':
        opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    else:
        opt = SGD(lr=1e-4, momentum=0.9) # for clr

    model.compile(loss='mean_squared_error',
                  optimizer=opt,
                  metrics=['mae', r2_krs])
    return model


def dump_model(self, outdir='.'):
    """ Dump trained model. """        
    self.model.save( str(Path(outdir)/'model.h5') )
        
#         # Serialize model to JSON
#         model_json = self.model.to_json()
#         modelpath = os.path.join(outdir, 'model.' + KERAS_REGRESSOR.model_name + '.json')
#         with open(modelpath, 'w') as mfile:
#             mfile.write(model_json)

#         # serialize weights to HDF5
#         weightpath = os.path.join(outdir, 'weights.' + KERAS_REGRESSOR.model_name + '.h5')
#         self.model.save_weights(weightpath)


