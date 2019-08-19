"""
This script contains various ML models and some utility functions.
"""
import warnings
warnings.filterwarnings('ignore')

import os
import sys
from pathlib import Path
from time import time
from collections import OrderedDict
import math

import sklearn
import numpy as np
import pandas as pd

import matplotlib
# matplotlib.use('TkAgg')
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib

import tensorflow as tf
import keras
from keras import backend as K
from keras.layers import Input, Dense, Dropout, Activation, BatchNormalization, Embedding, Flatten, Lambda, merge
from keras import optimizers
from keras.optimizers import SGD, Adam, RMSprop, Adadelta
from keras.models import Sequential, Model, model_from_json, model_from_yaml
from keras.utils import np_utils, multi_gpu_model
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard

try:
    import lightgbm as lgb
except ImportError:
    print('Module not found (lightgbm).')


def clr_keras_callback(mode=None, base_lr=1e-4, max_lr=1e-3, gamma=0.999994):
    """ Creates keras callback for cyclical learning rate. """
    keras_contrib = '/vol/ml/apartin/projects/keras-contrib/keras_contrib/callbacks'
    sys.path.append(keras_contrib)
    from cyclical_learning_rate import CyclicLR

    if mode == 'trng1':
        clr = CyclicLR(base_lr=base_lr, max_lr=max_lr, mode='triangular')
    elif mode == 'trng2':
        clr = CyclicLR(base_lr=base_lr, max_lr=max_lr, mode='triangular2')
    elif mode == 'exp':
        clr = CyclicLR(base_lr=base_lr, max_lr=max_lr, mode='exp_range', gamma=gamma) # 0.99994; 0.99999994; 0.999994
    return clr


def r2_krs(y_true, y_pred):
    # from keras import backend as K
    SS_res =  K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res/(SS_tot + K.epsilon()))


def get_model(model_name, init_kwargs=None):
    """ Return a model.
    Args:
        init_kwargs : init parameters to the model
        model_name : model name
    """
    if model_name == 'lgb_reg':
        model = LGBM_REGRESSOR(**init_kwargs)
    elif model_name == 'rf_reg':
        model = RF_REGRESSOR(**init_kwargs)
    elif model_name == 'nn_reg':
        model = KERAS_REGRESSOR(**init_kwargs)
    elif model_name == 'nn_reg0':
        model = NN_REG0(**init_kwargs)
    elif model_name == 'nn_reg1':
        model = NN_REG1(**init_kwargs)
    elif model_name == 'nn_reg2':
        model = NN_REG2(**init_kwargs)
    elif model_name == 'nn_reg3':
        model = NN_REG3(**init_kwargs)
    elif model_name == 'nn_reg4':
        model = NN_REG4(**init_kwargs)
    elif model_name == 'nn_reg5':
        model = NN_REG5(**init_kwargs)
    elif model_name == 'nn_reg6':
        model = NN_REG6(**init_kwargs)
    elif model_name == 'nn_reg_2layer':
        model = NN_REG_2LAYER(**init_kwargs)
    elif model_name == 'nn_reg_4layer':
        model = NN_REG_4LAYER(**init_kwargs)
    else:
        raise ValueError('model_name is invalid.')
    return model


def save_krs_history(history, outdir='.'):
    fname = 'krs_history.csv'
    h = pd.DataFrame(history.history)
    h['epoch'] = np.asarray(history.epoch) + 1
    
    if 'lr' in hh:
        h['epoch']
        ax2 = ax1.twinx()
        ax2.plot(eps, hh['lr'][skp_ep:], color='g', marker='.', linestyle=':', linewidth=1,
                 alpha=0.6, markersize=5, label='LR')
    
    h.to_csv( Path(outdir)/fname, index=False )
    return h


# def get_keras_prfrm_metrics(history):
#     """ Extract names of all the recorded performance metrics from keras history for trina
#     and val sets. The performance metrics can be indentified as those starting with 'val'.
#     """
#     # all metrics including everything returned from callbacks
#     all_metrics = list(history.history.keys()) 
#     # performance metrics recorded for train and val such as 'loss', etc. (excluding callbacks)
#     pr_metrics = ['_'.join(m.split('_')[1:]) for m in all_metrics if 'val' in m]
#     return pr_metrics


def capitalize_metric(met):
    return ' '.join(s.capitalize() for s in met.split('_'))


def plot_prfrm_metrics(history, title=None, skp_ep=0, outdir='.', add_lr=False):
    """ Plots keras training curves history.
    Args:
        skp_ep: number of epochs to skip when plotting metrics 
        add_lr: add curve of learning rate progression over epochs
    """
    all_metrics = list(history.history.keys())
    pr_metrics = ['_'.join(m.split('_')[1:]) for m in all_metrics if 'val' in m]

    epochs = np.asarray(history.epoch) + 1
    if len(epochs) <= skp_ep: skp_ep = 0
    eps = epochs[skp_ep:]
    hh = history.history
        
    for p, m in enumerate(pr_metrics):
        metric_name = m
        metric_name_val = 'val_' + m

        y_tr = hh[metric_name][skp_ep:]
        y_vl = hh[metric_name_val][skp_ep:]
        
        ymin = min(set(y_tr).union(y_vl))
        ymax = max(set(y_tr).union(y_vl))
        lim = (ymax - ymin) * 0.1
        ymin, ymax = ymin - lim, ymax + lim

        # Start figure
        fig, ax1 = plt.subplots()
        
        # Plot metrics
        # ax1.plot(eps, y_tr, color='b', marker='.', linestyle='-', linewidth=1, alpha=0.6, label=metric_name)
        # ax1.plot(eps, y_vl, color='r', marker='.', linestyle='--', linewidth=1, alpha=0.6, label=metric_name_val)
        ax1.plot(eps, y_tr, color='b', marker='.', linestyle='-', linewidth=1, alpha=0.6, label=capitalize_metric(metric_name))
        ax1.plot(eps, y_vl, color='r', marker='.', linestyle='--', linewidth=1, alpha=0.6, label=capitalize_metric(metric_name_val))
        ax1.set_xlabel('Epoch')
        # ylabel = ' '.join(s.capitalize() for s in metric_name.split('_'))
        ax1.set_ylabel(capitalize_metric(metric_name))
        ax1.set_xlim([min(eps)-1, max(eps)+1])
        ax1.set_ylim([ymin, ymax])
        ax1.tick_params('y', colors='k')
        
        # ax1.tick_params(axis='both', which='major', labelsize=12)
        # ax1.tick_params(axis='both', which='minor', labelsize=12)        
        
        # Add learning rate
        if (add_lr is True) and ('lr' in hh):            
            ax2 = ax1.twinx()
            ax2.plot(eps, hh['lr'][skp_ep:], color='g', marker='.', linestyle=':', linewidth=1,
                     alpha=0.6, markersize=5, label='LR')
            ax2.set_ylabel('Learning rate', color='g', fontsize=12)
            
            ax2.set_yscale('log') # 'linear'
            ax2.tick_params('y', colors='g')
        
        ax1.grid(True)
        # plt.legend([metric_name, metric_name_val], loc='best')
        # medium.com/@samchaaa/how-to-plot-two-different-scales-on-one-plot-in-matplotlib-with-legend-46554ba5915a
        legend = ax1.legend(loc='best', prop={'size': 10})
        frame = legend.get_frame()
        frame.set_facecolor('0.95')
        if title is not None: plt.title(title)
        
        # fig.tight_layout()
        figpath = Path(outdir) / (metric_name+'.png')
        plt.savefig(figpath, bbox_inches='tight')
        plt.close()
        

def plot_metrics_from_logs(path_to_logs, title=None, name=None, skp_ep=0, outdir='.'):
    """ Plots keras training from logs.
    Args:
        path_to_logs : full path to log file
        skp_ep: number of epochs to skip when plotting metrics 
    """
    history = pd.read_csv(path_to_logs, sep=',', header=0)
    
    all_metrics = list(history.columns)
    pr_metrics = ['_'.join(m.split('_')[1:]) for m in all_metrics if 'val' in m]

    epochs = history['epoch'] + 1
    if len(epochs) <= skp_ep: skp_ep = 0
    eps = epochs[skp_ep:]
    hh = history
    
    for p, m in enumerate(pr_metrics):
        metric_name = m
        metric_name_val = 'val_' + m

        y_tr = hh[metric_name][skp_ep:]
        y_vl = hh[metric_name_val][skp_ep:]
        
        ymin = min(set(y_tr).union(y_vl))
        ymax = max(set(y_tr).union(y_vl))
        lim = (ymax - ymin) * 0.1
        ymin, ymax = ymin - lim, ymax + lim

        # Start figure
        fig, ax1 = plt.subplots()
        
        # Plot metrics
        # ax1.plot(eps, y_tr, color='b', marker='.', linestyle='-', linewidth=1, alpha=0.6, label=metric_name)
        # ax1.plot(eps, y_vl, color='r', marker='.', linestyle='--', linewidth=1, alpha=0.6, label=metric_name_val)
        ax1.plot(eps, y_tr, color='b', marker='.', linestyle='-', linewidth=1, alpha=0.6, label=capitalize_metric(metric_name))
        ax1.plot(eps, y_vl, color='r', marker='.', linestyle='--', linewidth=1, alpha=0.6, label=capitalize_metric(metric_name_val))        
        ax1.set_xlabel('Epoch')
        # ylabel = ' '.join(s.capitalize() for s in metric_name.split('_'))
        ax1.set_ylabel(capitalize_metric(metric_name))
        ax1.set_ylabel(ylabel)
        ax1.set_xlim([min(eps)-1, max(eps)+1])
        ax1.set_ylim([ymin, ymax])
        ax1.tick_params('y', colors='k')
        
        ax1.grid(True)
        # plt.legend([metric_name, metric_name_val], loc='best')
        # medium.com/@samchaaa/how-to-plot-two-different-scales-on-one-plot-in-matplotlib-with-legend-46554ba5915a
        legend = ax1.legend(loc='best', prop={'size': 10})
        frame = legend.get_frame()
        frame.set_facecolor('0.95')
        if title is not None: plt.title(title)
        
        # fig.tight_layout()
        if name is not None:
            fname = name + '_' + metric_name + '.png'
        else:
            fname = metric_name + '.png'
        figpath = Path(outdir) / fname
        plt.savefig(figpath, bbox_inches='tight')
        plt.close()
        
    return history
        
        
class Attention(keras.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(Attention, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(Attention, self).build(input_shape)
    
    def call(self, V):
        Q = keras.backend.dot(V, self.kernel)
        Q =  Q * V
        Q = Q / math.sqrt(self.output_dim)
        Q = keras.activations.softmax(Q)
        return Q
    
    def compute_output_shape(self, input_shape):
        return input_shape


class BaseMLModel():
    """ A parent class with some general methods for children ML classes.
    The children classes are specific ML models such random forest regressor, lightgbm regressor, etc.
    """
    def __adj_r2_score(self, ydata, preds):
        """ Calc adjusted r^2.
        https://en.wikipedia.org/wiki/Coefficient_of_determination#Adjusted_R2
        https://dziganto.github.io/data%20science/linear%20regression/machine%20learning/python/Linear-Regression-101-Metrics/
        https://stats.stackexchange.com/questions/334004/can-r2-be-greater-than-1
        """
        r2_score = sklearn.metrics.r2_score(ydata, preds)
        adj_r2 = 1 - (1 - r2_score) * (self.x_size[0] - 1)/(self.x_size[0] - self.x_size[1] - 1)
        return adj_r2


    def build_dense_block(self, layers, inputs, name=''):
        """ This function only applicable to keras NNs. """
        for i, l_size in enumerate(layers):
            if i == 0:
                x = Dense(l_size, kernel_initializer=self.initializer, name=f'{name}.fc{i+1}.{l_size}')(inputs)
            else:
                x = Dense(l_size, kernel_initializer=self.initializer, name=f'{name}.fc{i+1}.{l_size}')(x)
            x = BatchNormalization(name=f'{name}.bn{i+1}')(x)
            x = Activation('relu', name=f'{name}.a{i+1}')(x)
            x = Dropout(self.dr_rate, name=f'{name}.drp{i+1}.{self.dr_rate}')(x)        
        return x


class KERAS_REGRESSOR(BaseMLModel):
    """ Neural network regressor. """
    model_name = 'nn_reg'

    def __init__(self, input_dim, attn=False, dr_rate=0.2, opt_name='sgd', logger=None):
        inputs = Input(shape=(input_dim,))
        if attn:
            a = Dense(1000, activation='relu')(inputs)
            b = Dense(1000, activation='softmax')(inputs)
            x = keras.layers.multiply( [a, b] )
        else:
            x = Dense(1000, activation='relu')(inputs)
            
        x = Dense(1000, activation='relu')(x)
        x = Dropout(dr_rate)(x)
        
        x = Dense(500, activation='relu')(x)
        x = Dropout(dr_rate)(x)
        
        x = Dense(250, activation='relu')(x)
        x = Dropout(dr_rate)(x)
        
        x = Dense(125, activation='relu')(x)
        x = Dropout(dr_rate)(x)
        
        x = Dense(60, activation='relu')(x)
        x = Dropout(dr_rate)(x)
        
        x = Dense(30, activation='relu')(x)
        x = Dropout(dr_rate)(x)
        
        outputs = Dense(1, activation='relu')(x)
        model = Model(inputs=inputs, outputs=outputs)
        
        if opt_name == 'sgd':
            opt = SGD(lr=1e-4, momentum=0.9)
        elif opt_name == 'adam':
            opt = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False) # original
            # opt = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
            # opt = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
        else:
            opt = SGD(lr=1e-4, momentum=0.9) # for clr

        model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae', r2_krs])
        self.model = model


    #def dump_model(self, outdir='.'):
    #    """ Dump trained model. """        
    #    self.model.save( str(Path(outdir)/'model.h5') )
        
class NN_REG_2LAYER(BaseMLModel):
    """ Neural network regressor.
    Fully-connected NN.
    """
    model_name = 'nn_ref_2layer'

    def __init__(self, input_dim, dr_rate=0.2, opt_name='sgd', initializer='he_uniform', logger=None):
        self.input_dim = input_dim
        self.dr_rate = dr_rate
        self.opt_name = opt_name
        self.initializer = initializer

        layers = [1000, 500]
        inputs = Input(shape=(self.input_dim,), name='inputs')
        x = self.build_dense_block(layers, inputs)

        outputs = Dense(1, activation='relu', name='outputs')(x)
        model = Model(inputs=inputs, outputs=outputs)
        
        if self.opt_name == 'sgd':
            opt = SGD(lr=1e-4, momentum=0.9)
        elif self.opt_name == 'adam':
            opt = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        else:
            opt = SGD(lr=1e-4, momentum=0.9) # for clr

        model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae']) # r2_krs 
        self.model = model


class NN_REG_4LAYER(BaseMLModel):
    """ Neural network regressor.
    Fully-connected NN.
    """
    model_name = 'nn_reg_4layer'

    def __init__(self, input_dim, dr_rate=0.2, opt_name='sgd', initializer='he_uniform', logger=None):
        self.input_dim = input_dim
        self.dr_rate = dr_rate
        self.opt_name = opt_name
        self.initializer = initializer

        layers = [1000, 500, 250, 125]
        inputs = Input(shape=(self.input_dim,), name='inputs')
        x = self.build_dense_block(layers, inputs)

        outputs = Dense(1, activation='relu', name='outputs')(x)
        model = Model(inputs=inputs, outputs=outputs)
        
        if self.opt_name == 'sgd':
            opt = SGD(lr=1e-4, momentum=0.9)
        elif self.opt_name == 'adam':
            opt = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        else:
            opt = SGD(lr=1e-4, momentum=0.9) # for clr

        model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae']) # r2_krs 
        self.model = model



class NN_REG0(BaseMLModel):
    """ Neural network regressor.
    Fully-connected NN.
    """
    model_name = 'nn_reg0'

    def __init__(self, input_dim, dr_rate=0.2, opt_name='sgd', initializer='he_uniform', logger=None):
        self.input_dim = input_dim
        self.dr_rate = dr_rate
        self.opt_name = opt_name
        self.initializer = initializer

        layers = [1000, 1000, 500, 250, 125, 60, 30]
        inputs = Input(shape=(self.input_dim,), name='inputs')
        x = self.build_dense_block(layers, inputs)

        outputs = Dense(1, activation='relu', name='outputs')(x)
        model = Model(inputs=inputs, outputs=outputs)
        
        if self.opt_name == 'sgd':
            opt = SGD(lr=1e-4, momentum=0.9)
        elif self.opt_name == 'adam':
            opt = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        else:
            opt = SGD(lr=1e-4, momentum=0.9) # for clr

        model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae']) # r2_krs 
        self.model = model
        

class NN_REG4(BaseMLModel):
    """ Neural network regressor.
    Fully-connected NN.
    """
    model_name = 'nn_reg4'

    def __init__(self, input_dim, dr_rate=0.2, opt_name='sgd', initializer='he_uniform', logger=None):
        self.input_dim = input_dim
        self.dr_rate = dr_rate
        self.opt_name = opt_name
        self.initializer = initializer

        layers = [1000, 1000, 500, 500, 250, 250]
        inputs = Input(shape=(self.input_dim,), name='inputs')
        x = self.build_dense_block(layers, inputs)

        outputs = Dense(1, activation='relu', name='outputs')(x)
        model = Model(inputs=inputs, outputs=outputs)
        
        if self.opt_name == 'sgd':
            opt = SGD(lr=1e-4, momentum=0.9)
        elif self.opt_name == 'adam':
            opt = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        else:
            opt = SGD(lr=1e-4, momentum=0.9) # for clr

        model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae']) # r2_krs 
        self.model = model
        

class NN_REG1(BaseMLModel):
    """ Neural network regressor. 
    Fully-connected NN with attention layer.
    """
    model_name = 'nn_reg1'

    def __init__(self, input_dim, dr_rate=0.2, opt_name='sgd', logger=None):
        inputs = Input(shape=(input_dim,))
        #x = Lambda(lambda x: x, output_shape=(1000,))(inputs)
        # attn_lin = Dense(1000, activation='relu', name='attn_lin')(inputs)
        # attn_probs = Dense(1000, activation='softmax', name='attn_probs')(inputs)
        # x = keras.layers.multiply( [attn_lin, attn_probs], name='attn')
        
        # New attention layer (Rick, Austin)
        a = Dense(1000)(inputs)
        a = BatchNormalization()(a)
        a = Activation('relu')(a)
        b = Attention(1000)(a)
        x = keras.layers.multiply([b, a])

        x = Dense(1000)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(dr_rate)(x)

        x = Dense(500)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(dr_rate)(x)
        
        x = Dense(250)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(dr_rate)(x)
        
        x = Dense(125)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(dr_rate)(x)

        x = Dense(60)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(dr_rate)(x)

        x = Dense(30)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(dr_rate)(x)

        outputs = Dense(1, activation='relu')(x)
        model = Model(inputs=inputs, outputs=outputs)
        
        if opt_name == 'sgd':
            opt = SGD(lr=1e-4, momentum=0.9)
        elif opt_name == 'adam':
            opt = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        else:
            opt = SGD(lr=1e-4, momentum=0.9) # for clr

        model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae']) # r2_krs 
        self.model = model


class NN_REG2(BaseMLModel):
    """ Neural network regressor. """
    model_name = 'nn_reg2'
   
    def __init__(self, input_dim, dr_rate=0.2, opt_name='sgd', logger=None):
        inputs = Input(shape=(input_dim,))
        x = Dense(1000, activation='relu')(inputs)
        x = Dropout(dr_rate)(x)
            
        x = Dense(1000, activation='relu')(x)
        x = Dropout(dr_rate)(x)

        x = Dense(500, activation='relu')(x)
        x = Dropout(dr_rate)(x)
        
        attn_lin = Dense(250, activation='relu', name='attn_lin')(x)
        attn_probs = Dense(250, activation='softmax', name='attn_probs')(x)
        x = keras.layers.multiply( [attn_lin, attn_probs], name='attn' )
        x = Dropout(dr_rate)(x)
        
        x = Dense(125, activation='relu')(x)
        x = Dropout(dr_rate)(x)

        x = Dense(60, activation='relu')(x)
        x = Dropout(dr_rate)(x)

        outputs = Dense(1, activation='relu')(x)
        model = Model(inputs=inputs, outputs=outputs)
        
        if opt_name == 'sgd':
            opt = SGD(lr=1e-4, momentum=0.9)
        elif opt_name == 'adam':
            opt = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        else:
            opt = SGD(lr=1e-4, momentum=0.9) # for clr

        model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae']) # r2_krs 
        self.model = model


class NN_REG3(BaseMLModel):
    """ Neural network regressor.
    Uno-style network.
    """
    model_name = 'nn_reg3'

    def __init__(self, in_dim_rna, in_dim_dsc, dr_rate=0.2, opt_name='sgd', initializer='he_uniform', logger=None):
        # https://keras.io/getting-started/functional-api-guide/
        # Chollet book
        
        #def create_tower(layers, inputs, name):

        #    for i, l_size in enumerate(layers):
        #        if i == 0:
        #            x = Dense(l_size, kernel_initializer=initializer, name=f'{name}_FC{i+1}')(inputs)
        #        else:
        #            x = Dense(l_size, kernel_initializer=initializer, name=f'{name}_FC{i+1}')(x)
        #        x = BatchNormalization(name=f'{name}_BN{i+1}')(x)
        #        x = Activation('relu', name=f'{name}_A{i+1}')(x)
        #        x = Dropout(dr_rate, name=f'{name}_DRP{i+1}')(x)        

        #    x = Model(inputs=inputs, outputs=x, name=f'out_{name}')
        #    return x

        in_rna = Input(shape=(in_dim_rna,), name='in_rna')
        out_rna = self.build_dense_block(layers=[1000,800,600], inputs=in_rna, name='rna')
        rna = Model(inputs=in_rna, outputs=out_rna, name=f'out_rna')
        
        in_dsc = Input(shape=(in_dim_dsc,), name='in_dsc')
        out_dsc = self.build_dense_block(layers=[1000,800,600], inputs=in_dsc, name='dsc')
        dsc = Model(inputs=in_dsc, outputs=out_dsc, name=f'out_dsc')

        # merged = merge.concatenate([rna.output, dsc.output])
        # x = create_tower(layers=[1000,800,600], input_dim=rna.output_shape[-1] + dsc.output_shape[-1], name='merged')

        """
        # RNA
        in_rna = Input(shape=(in_dim_rna,), name='in_rna')
        layers = [1000, 800, 600]

        for i, l_size in enumerate(layers):
            if i == 0:
                x = Dense(l_size, kernel_initializer=initializer, name=f'FC{i+1}')(in_rna)
            else:
                x = Dense(l_size, kernel_initializer=initializer, name=f'FC{i+1}')(x)
            x = BatchNormalization(name=f'BN{i+1}')(x)
            x = Activation('relu', name=f'A{i+1}')(x)
            x = Dropout(dr_rate, name=f'DRP{i+1}')(x)        

        rna = Model(inputs=in_rna, outputs=x, name='out_rna')
        del x

        # DSC
        in_dsc = Input(shape=(in_dim_dsc,), name='in_dsc')
        layers = [1000, 800, 600]

        for i, l_size in enumerate(layers):
            if i == 0:
                x = Dense(l_size, kernel_initializer=initializer, name=f'FC{i+1}')(in_dsc)
            else:
                x = Dense(l_size, kernel_initializer=initializer, name=f'FC{i+1}')(x)
            x = BatchNormalization(name=f'BN{i+1}')(x)
            x = Activation('relu', name=f'A{i+1}')(x)
            x = Dropout(dr_rate, name=f'DRP{i+1}')(x)        

        dsc = Model(inputs=in_dsc, outputs=x, name='out_dsc')
        del x
        """

        """
        # Proc rna
        in_rna = Input(shape=(in_dim_rna,), name='in_rna')
        a = Dense(1000)(in_rna)
        a = BatchNormalization()(a)
        a = Activation('relu')(a)
        a = Dropout(dr_rate)(a)

        a = Dense(800)(a)
        a = BatchNormalization()(a)
        a = Activation('relu')(a)
        a = Dropout(dr_rate)(a)
        
        a = Dense(600)(a)
        a = BatchNormalization()(a)
        a = Activation('relu')(a)
        a = Dropout(dr_rate)(a)

        rna = Model(inputs=in_rna, outputs=a, name='out_rna')

        # Proc dsc
        in_dsc = Input(shape=(in_dim_dsc,), name='in_dsc')
        b = Dense(1000)(in_dsc)
        b = BatchNormalization()(b)
        b = Activation('relu')(b)
        b = Dropout(dr_rate)(b)

        b = Dense(800)(b)
        b = BatchNormalization()(b)
        b = Activation('relu')(b)
        b = Dropout(dr_rate)(b)

        b = Dense(600)(b)
        b = BatchNormalization()(b)
        b = Activation('relu')(b)
        b = Dropout(dr_rate)(b)
        
        dsc = Model(inputs=in_dsc, outputs=b, name='out_dsc')
        """
        
        # Merge layers
        merged = merge.concatenate([rna.output, dsc.output])
        
        # Dense layers
        x = Dense(1000, name='in_merged')(merged)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(dr_rate)(x)

        x = Dense(800)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(dr_rate)(x)

        x = Dense(600)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(dr_rate)(x)

        outputs = Dense(1, activation='relu', name='output')(x)
        model = Model(inputs=[in_rna, in_dsc], outputs=[outputs])
        
        if opt_name == 'sgd':
            opt = SGD(lr=1e-4, momentum=0.9)
        elif opt_name == 'adam':
            opt = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        else:
            opt = SGD(lr=1e-4, momentum=0.9) # for clr

        model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae']) # r2_krs 
        self.model = model


    def fit_cv(self,
            in_rna, in_dsc,
            epochs: int=150, batch_size: int=32,
            cv: int=5,
            cv_splits: tuple=None):
        # TODO: finish this!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        pass
        """
        for 
        model.fit({'in_rna': in_rna, 'in_dsc': in_dsc},
                  {'out_nn': ydata},
                    epochs=epochs, batch_size=batch_size)   

        if cv_splits is not None:
            tr_id = cv_splits[0]
            vl_id = cv_splits[1]
            assert tr_id.shape[1]==vl_id.shape[1], 'tr and vl must have the same of folds.'
            cv_folds = tr_id.shape[1]

            for i in range(tr_id.shape[1]):
                tr_dct[i] = tr_id.iloc[:, i].dropna().values.astype(int).tolist()
                vl_dct[i] = vl_id.iloc[:, i].dropna().values.astype(int).tolist()

            if tr_id.shape[1] == 1:
                vl_size = vl_id.shape[0]/(vl_id.shape[0] + tr_id.shape[0])

        # If pre-defined splits are not passed, then generate splits on the fly
        else:
            if isinstance(cv, int):
                cv_folds = cv
                cv = KFold(n_splits=cv_folds, shuffle=False, random_state=random_state)
            else:
                cv_folds = cv.get_n_splits() # cv is a sklearn splitter

            if cv_folds == 1:
                vl_size = cv.test_size
        """



class NN_REG5(BaseMLModel):
    """ Neural network regressor.
    Fully-connected NN.
    """
    model_name = 'nn_reg5'

    def __init__(self, input_dim, dr_rate=0.2, opt_name='sgd', initializer='he_uniform', logger=None):
        self.input_dim = input_dim
        self.dr_rate = dr_rate
        self.opt_name = opt_name
        self.initializer = initializer

        layers = [3000, 2000, 1000]
        inputs = Input(shape=(self.input_dim,), name='inputs')
        x = self.build_dense_block(layers)

        outputs = Dense(1, activation='relu', name='outputs')(x)
        model = Model(inputs=inputs, outputs=outputs)
        
        if self.opt_name == 'sgd':
            opt = SGD(lr=1e-4, momentum=0.9)
        elif self.opt_name == 'adam':
            opt = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        else:
            opt = SGD(lr=1e-4, momentum=0.9) # for clr

        model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae']) # r2_krs
        self.model = model
        

class NN_REG6(BaseMLModel):
    """ Neural network regressor.
    Embbeding for drugs.
    https://keras.io/getting-started/functional-api-guide/
    https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/
    https://medium.com/@chunduri11/deep-learning-part-1-fast-ai-rossman-notebook-7787bfbc309f
    """
    model_name = 'nn_reg6'

    def __init__(self, in_dim_rna, unq_drg_labels, dr_rate=0.2, opt_name='sgd', initializer='he_uniform', logger=None):
        # https://keras.io/getting-started/functional-api-guide/
        # Chollet book
        self.in_dim_rna = in_dim_rna
        self.unq_drg_labels = unq_drg_labels
        self.dr_rate = dr_rate
        self.opt_name = opt_name
        self.initializer = initializer

        # RNA
        in_rna = Input(shape=(in_dim_rna,), name='in_rna')
        out_rna = self.build_dense_block(layers=[1000, 1000], inputs=in_rna, name='rna')
        rna = Model(inputs=in_rna, outputs=out_rna, name=f'out_rna')

        # Drug embedding 
        # in_dim_drg_embd = len(drg_pdm)
        # in_dim_drg_embd = len(drg_common)
        # in_dim_drg_embd = len(drg)
        #dim_drg_embd = (in_dim_drg_lbl+1)//2
        dim_drg_embd = (unq_drg_labels+1)//2

        in_drg_lbl = Input(shape=(1,), name='in_drg_lbl')
        out_drg_embd = Embedding(input_dim = unq_drg_labels,
                                 output_dim = dim_drg_embd,
                                 input_length=1, name='out_drg_embd')(in_drg_lbl)
        out_drg_embd = Flatten()(out_drg_embd)
        drg_embd = Model(inputs=in_drg_lbl, outputs=out_drg_embd, name='out_drg')

        # Merge layers
        merged = merge.concatenate([rna.output, drg_embd.output])
        
        # Dense layers
        x = Dense(500, name='in_merged')(merged)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(dr_rate)(x)

        x = Dense(250)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(dr_rate)(x)

        x = Dense(125)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(dr_rate)(x)

        outputs = Dense(1, activation='relu', name='output')(x)
        model = Model(inputs=[in_rna, in_drg_lbl], outputs=[outputs])
        
        if opt_name == 'sgd':
            opt = SGD(lr=1e-4, momentum=0.9)
        elif opt_name == 'adam':
            opt = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        else:
            opt = SGD(lr=1e-4, momentum=0.9) # for clr

        model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae']) # r2_krs
        self.model = model


class LGBM_REGRESSOR(BaseMLModel):
    """ LightGBM regressor. """
    ml_objective = 'regression'
    model_name = 'lgb_reg'

    def __init__(self, n_estimators=100, eval_metric=['l2', 'l1'], n_jobs=1, random_state=None, logger=None):
        # TODO: use config file to set default parameters (like in candle)
        
        self.model = lgb.LGBMModel(
            objective = LGBM_REGRESSOR.ml_objective,
            n_estimators = n_estimators,
            n_jobs = n_jobs,
            random_state = random_state)


    # def fit(self, X, y, eval_set=None, **fit_params):
    #     #self.eval_set = eval_set
    #     #self.X = X
    #     #self.y = y
    #     #self.x_size = X.shape  # this is used to calc adjusteed r^2
        
    #     t0 = time.time()
    #     self.model.fit(X, y,
    #                    eval_metric=self.eval_metric,
    #                    eval_set=eval_set,
    #                    **fit_params)
    #     self.train_runtime = time.time() - t0

    #     if self.logger is not None:
    #         self.logger.info('Train time: {:.2f} mins'.format(self.train_runtime/60))


    def dump_model(self, outdir='.'):
        # lgb_reg.save_model(os.path.join(run_outdir, 'lgb_'+ml_type+'_model.txt'))
        joblib.dump(self.model, filename=Path(outdir)/('model.' + LGBM_REGRESSOR.model_name + '.pkl'))
        # lgb_reg_ = joblib.load(filename=os.path.join(run_outdir, 'lgb_reg_model.pkl'))

        
    def plot_fi(self, max_num_features=20, title='LGBMRegressor', outdir=None):
        lgb.plot_importance(booster=self.model, max_num_features=max_num_features, grid=True, title=title)
        plt.tight_layout()

        filename = LGBM_REGRESSOR.model_name + '_fi.png'
        if outdir is None:
            plt.savefig(filename, bbox_inches='tight')
        else:
            plt.savefig(Path(outdir)/filename, bbox_inches='tight')


    # # Plot training curves
    # # TODO: note, plot_metric didn't accept 'mae' although it's alias for 'l1' 
    # # TODO: plot_metric requires dict from train(), but train returns 'lightgbm.basic.Booster'??
    # for m in eval_metric:
    #     ax = lgb.plot_metric(booster=lgb_reg, metric=m, grid=True)
    #     plt.savefig(os.path.join(run_outdir, model_name+'_learning_curve_'+m+'.png'))
    

class RF_REGRESSOR(BaseMLModel):
    """ Random forest regressor. """
    # Define class attributes (www.toptal.com/python/python-class-attributes-an-overly-thorough-guide)
    model_name = 'rf_reg'

    def __init__(self, n_estimators=100, criterion='mse',
                 max_depth=None, min_samples_split=2,
                 max_features='sqrt',
                 bootstrap=True, oob_score=True, verbose=0, 
                 n_jobs=1, random_state=None,
                 logger=None):               

        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            max_features=max_features, bootstrap=bootstrap, oob_score=oob_score,
            verbose=verbose, random_state=random_state, n_jobs=n_jobs)


    def plot_fi(self):
        pass # TODO


    def dump_model(self, outdir='.'):
        joblib.dump(self.model, filename=os.path.join(outdir, 'model.' + RF_REGRESSOR.model_name + '.pkl'))
        # model_ = joblib.load(filename=os.path.join(run_outdir, 'lgb_reg_model.pkl'))


class LGBM_CLASSIFIER(BaseMLModel):
    # TODO: finish
    """ LightGBM classifier. """
    ml_objective = 'binary'
    model_name = 'lgb_cls'

    def __init__(self, eval_metric=['l2', 'l1'], n_jobs=1, random_state=None, logger=None):
        self.eval_metric = eval_metric
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.logger = logger

        # ----- lightgbm "sklearn API" - start
        # self.model = lgb.LGBMModel(objective=LGBM_REGRESSOR.ml_objective,
        #                            n_jobs=self.n_jobs,
        #                            random_state=self.random_state)
        # ----- lightgbm "sklearn API" - end

        # ----- lightgbm "sklearn API" - start
        self.model = lgb.LGBMClassifier(
            n_jobs=self.n_jobs,
            random_state=self.random_state)
        # ----- lightgbm "sklearn API" - end


class XGBM_REGRESSOR(BaseMLModel):
    """ xgboost regressor. """
    ml_objective = 'regression'
    model_name = 'xgb_reg'


# # ========================================================================
# #       Train models - the code below was in train_from_combined.py
# # ========================================================================
# from sklearn.metrics import r2_score, mean_absolute_error, median_absolute_error
# from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, ParameterGrid
# from sklearn.externals import joblib

# train_runtime = OrderedDict() # {}
# preds_filename_prefix = 'dev'

# # ---------------------
# # RandomForestRegressor
# # ---------------------
# if 'rf_reg' in ml_models:
#     model_name = 'rf_reg'
#     try:
#         from sklearn.ensemble import RandomForestRegressor
#     except ImportError:
#         # install??
#         logger.error(f'Module not found (RandomForestRegressor)')

#     logger.info('\nTrain RandomForestRegressor ...')
#     # ----- rf hyper-param start
#     rf_reg = RandomForestRegressor(max_features='sqrt', bootstrap=True, oob_score=True,
#                                 verbose=0, random_state=SEED, n_jobs=n_jobs)

#     random_search_params = {'n_estimators': [100, 500, 1000], # [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
#                             'max_depth': [None, 5, 10], # [None] + [int(x) for x in np.linspace(10, 110, num = 11)]
#                             'min_samples_split': [2, 5, 9]}
#     logger.info('hyper-params:\n{}'.format(random_search_params))

#     rf_reg_randsearch = RandomizedSearchCV(
#         estimator=rf_reg,
#         param_distributions=random_search_params,
#         n_iter=20,  # num of parameter settings that are sampled and used for training (num of models trained)
#         scoring=None, # string or callable used to evaluate the predictions on the test set
#         n_jobs=n_jobs,
#         cv=5,
#         refit=True,  # Refit an estimator using the best found parameters on the whole dataset
#         verbose=0)

#     # Run search
#     t0 = time.time()
#     rf_reg_randsearch.fit(xtr, ytr)
#     train_runtime[model_name+'_randsearch'] = time.time() - t0
#     logger.info('Runtime: {:.2f} mins'.format(train_runtime[model_name+'_randsearch']/60))

#     # Save best model
#     rf_reg = rf_reg_randsearch.best_estimator_
#     joblib.dump(rf_reg, filename=os.path.join(run_outdir, model_name+'_hypsearch_best_model.pkl'))

#     # Print preds
#     utils.print_scores(model=rf_reg, xdata=xvl, ydata=yvl, logger=logger)

#     # Save resutls
#     rf_reg_hypsearch = pd.DataFrame(rf_reg_randsearch.cv_results_)
#     rf_reg_hypsearch.to_csv(os.path.join(run_outdir, model_name+'_hypsearch_summary.csv'))  # save hyperparam search results

#     logger.info(f'{model_name} best score (random search): {rf_reg_randsearch.best_score_:.3f}')
#     logger.info('{} best params (random search): \n{}'.format(model_name, rf_reg_randsearch.best_params_))

#     # Dump preds
#     utils.dump_preds(model=rf_reg, df_data=vl_data, xdata=xvl, target_name=target_name,
#                     path=os.path.join(run_outdir, preds_filename_prefix+'_'+model_name+'_preds.csv'))
#     # ----- rf hyper-param end


# # ------------
# # XGBRegressor
# # ------------
# if 'xgb_reg' in ml_models:
#     try:
#         import xgboost as xgb
#     except ImportError:  # install??
#         logger.error('Module not found (xgboost)')

#     # https://xgboost.readthedocs.io/en/latest/python/python_api.html
#     # xgboost does not support categorical features!
#     # Rules of thumb
#     # 1. learning_rate should be 0.1 or lower (smaller values will require more trees).
#     # 2. tree_depth should be between 2 and 8 (where not much benefit is seen with deeper trees).
#     # 3. subsample should be between 30% and 80% of the training dataset, and compared to a value of 100% for no sampling.
#     logger.info('\nTrain XGBRegressor ...')
#     # xgb_tr = xgb.DMatrix(data=xtr, label=ytr, nthread=n_jobs)
#     # xgb_vl = xgb.DMatrix(data=xvl, label=yvl, nthread=n_jobs)
#     # ----- xgboost hyper-param start
#     xgb_reg = xgb.XGBRegressor(objective='reg:linear', # default: 'reg:linear', TODO: docs recommend funcs for different distributions (??)
#                             booster='gbtree', # default: gbtree (others: gblinear, dart)
#                             # max_depth=3, # default: 3
#                             # learning_rate=0.1, # default: 0.1
#                             # n_estimators=100, # default: 100
#                             n_jobs=n_jobs, # default: 1
#                             reg_alpha=0, # default=0, L1 regularization
#                             reg_lambda=1, # default=1, L2 regularization
#                             random_state=SEED)

#     random_search_params = {'n_estimators': [30, 50, 70],
#                             'learning_rate': [0.005, 0.01, 0.5],
#                             'subsample': [0.5, 0.7, 0.8],
#                             'max_depth': [2, 3, 5]}
#     logger.info('hyper-params:\n{}'.format(random_search_params))

#     xgb_reg_randsearch = RandomizedSearchCV(
#         estimator=xgb_reg,
#         param_distributions=random_search_params,
#         n_iter=20,  # num of parameter settings that are sampled and used for training (num of models trained)
#         scoring=None, # string or callable used to evaluate the predictions on the test set
#         n_jobs=n_jobs,
#         cv=5,
#         refit=True,  # Refit an estimator using the best found parameters on the whole dataset
#         verbose=False)   

#     # Start search
#     t0 = time.time()
#     xgb_reg_randsearch.fit(xtr, ytr)
#     train_runtime['xgb_reg_randsearch'] = time.time() - t0
#     logger.info('Runtime: {:.2f} mins'.format(train_runtime['xgb_reg_randsearch']/60))

#     # Save best model
#     xgb_reg = xgb_reg_randsearch.best_estimator_
#     joblib.dump(xgb_reg, filename=os.path.join(run_outdir, 'xgb_reg_hypsearch_best_model.pkl'))

#     # Print preds
#     utils.print_scores(model=xgb_reg, xdata=xvl, ydata=yvl, logger=logger)

#     # Save resutls
#     xgb_reg_hypsearch = pd.DataFrame(xgb_reg_randsearch.cv_results_)
#     xgb_reg_hypsearch.to_csv(os.path.join(run_outdir, 'xgb_reg_hypsearch_summary.csv'))  # save hyperparam search results

#     logger.info(f'rf_reg best score (random search): {xgb_reg_randsearch.best_score_:.3f}')
#     logger.info('rf_reg best params (random search): \n{}'.format(xgb_reg_randsearch.best_params_))

#     # Dump preds
#     utils.dump_preds(model=xgb_reg, df_data=vl_data, xdata=xvl, target_name=target_name,
#                     path=os.path.join(run_outdir, 'xgb_vl_preds.csv'))
#     # ----- xgboost hyper-param end

#     # ----- xgboost "Sklearn API" start
#     xgb_reg = xgb.XGBRegressor(objective='reg:linear', # default: 'reg:linear', TODO: docs recommend funcs for different distributions (??)
#                             booster='gbtree', # default: gbtree (others: gblinear, dart)
#                             max_depth=3, # default: 3
#                             learning_rate=0.1, # default: 0.1
#                             n_estimators=100, # default: 100
#                             n_jobs=n_jobs, # default: 1
#                             reg_alpha=0, # default=0, L1 regularization
#                             reg_lambda=1, # default=1, L2 regularization
#                             random_state=SEED
#     )
#     eval_metric = ['mae', 'rmse']
#     t0 = time.time()
#     xgb_reg.fit(xtr, ytr, eval_metric=eval_metric, eval_set=[(xtr, ytr), (xvl, yvl)],
#                 early_stopping_rounds=10, verbose=False, callbacks=None)
#     train_runtime['xgb_reg'] = time.time() - t0
#     logger.info('Runtime: {:.2f} mins'.format(train_runtime['xgb_reg']/60))

#     # Save model
#     # xgb_reg.save_model(os.path.join(run_outdir, 'xgb_reg.model'))
#     joblib.dump(xgb_reg, filename=os.path.join(run_outdir, 'xgb_reg_model.pkl'))
#     # xgb_reg_ = joblib.load(filename=os.path.join(run_outdir, 'xgb_reg_model.pkl'))

#     # Print preds
#     utils.print_scores(model=xgb_reg, xdata=xvl, ydata=yvl, logger=logger)

#     # Dump preds
#     utils.dump_preds(model=xgb_reg, df_data=vl_data, xdata=xvl, target_name=target_name,
#                     path=os.path.join(run_outdir, 'xgb_vl_preds.csv'))
#     # ----- xgboost "Sklearn API" end
        
#     # Plot feature importance
#     xgb.plot_importance(booster=xgb_reg, max_num_features=20, grid=True, title='XGBRegressor')
#     plt.tight_layout()
#     plt.savefig(os.path.join(run_outdir, 'xgb_reg_importances.png'))

#     # Plot learning curves
#     xgb_results = xgb_reg.evals_result()
#     epoch_vec = np.arange(1, len(xgb_results['validation_0'][eval_metric[0]])+1)
#     for m in eval_metric:
#         fig, ax = plt.subplots()
#         for i, s in enumerate(xgb_results):
#             label = 'Train' if i==0 else 'Val'
#             ax.plot(epoch_vec, xgb_results[s][m], label=label)
#         plt.xlabel('Epochs')
#         plt.title(m)
#         plt.legend()
#         plt.grid(True)
#         plt.tight_layout()
#         plt.savefig(os.path.join(run_outdir, 'xgb_reg_leraning_curve_'+m+'.png'))


# # -------------
# # LGBMRegressor
# # -------------
# if 'lgb_reg' in ml_models:
#     model_name = 'lgb_reg'
#     try:
#         import lightgbm as lgb
#     except ImportError:  # install??
#         logger.error('Module not found (lightgbm)')

#     # https://lightgbm.readthedocs.io/en/latest/Python-API.html
#     # TODO: use config file to set default parameters
#     logger.info('\nTrain LGBMRegressor ...')
#     ml_objective = 'regression'
#     eval_metric = ['l1', # aliases: regression_l1, mean_absolute_error, mae
#                    'l2', # aliases: regression, regression_l2, mean_squared_error, mse, and more
#                    ]

#     # ----- lightgbm "Training API" - start
#     # lgb_tr = lgb.Dataset(data=xtr, label=ytr, categorical_feature='auto')
#     # lgb_vl = lgb.Dataset(data=xvl, label=yvl, categorical_feature='auto')
#     # # https://lightgbm.readthedocs.io/en/latest/Parameters.html
#     # params = {'task': 'train', # default='train'
#     #         'objective': ml_objective, # default='regression' which alias for 'rmse' and 'mse' (but these are different??)
#     #         'boosting': 'gbdt', # default='gbdt'
#     #         'num_iterations': 100, # default=100 (num of boosting iterations)
#     #         'learning_rate': 0.1, # default=0.1
#     #         'num_leaves': 31, # default=31 (num of leaves in 1 tree)
#     #         'seed': SEED,
#     #         'num_threads': n_jobs, # default=0 (set to the num of real CPU cores)
#     #         'device_type': 'cpu', # default='cpu'
#     #         'metric': eval_metric # metric(s) to be evaluated on the evaluation set(s)
#     #         }
#     # t0 = time.time()
#     # lgb_reg = lgb.train(params=params, train_set=lgb_tr, valid_sets=lgb_vl, verbose_eval=False)
#     # # lgb_cv = lgb.train(params=params, train_set=lgb_tr, nfolds=5)
#     # train_runtime['lgb_reg'] = time.time() - t0
#     # logger.info('Runtime: {:.2f} mins'.format(train_runtime['lgb_reg']/60))
#     # ----- lightgbm "Training API" - end 

#     # ----- lightgbm "sklearn API" appraoch 1 - start
#     lgb_reg = lgb.LGBMModel(objective=ml_objective,
#                             n_jobs=n_jobs,
#                             random_state=SEED)
#     # lgb_reg = lgb.LGBMRegressor()
#     t0 = time.time()
#     lgb_reg.fit(xtr, ytr, eval_metric=eval_metric, eval_set=[(xtr, ytr), (xvl, yvl)],
#                 early_stopping_rounds=10, verbose=False, callbacks=None)
#     train_runtime[model_name] = time.time() - t0
#     logger.info('Runtime: {:.2f} mins'.format(train_runtime[model_name]/60))
#     # ----- lightgbm "sklearn API" appraoch 1 - end

#     # Save model
#     # lgb_reg.save_model(os.path.join(run_outdir, 'lgb_'+ml_type+'_model.txt'))
#     joblib.dump(lgb_reg, filename=os.path.join(run_outdir, model_name+'_model.pkl'))
#     # lgb_reg_ = joblib.load(filename=os.path.join(run_outdir, 'lgb_reg_model.pkl'))

#     # Print preds
#     # utils.print_scores(model=lgb_reg, xdata=xtr, ydata=ytr)
#     # utils.print_scores(model=lgb_reg, xdata=xvl, ydata=yvl)
#     utils.print_scores(model=lgb_reg, xdata=xvl, ydata=yvl, logger=logger)

#     # Dump preds
#     utils.dump_preds(model=lgb_reg, df_data=vl_data, xdata=xvl, target_name=target_name,
#                      path=os.path.join(run_outdir, preds_filename_prefix+'_'+model_name+'_preds.csv'))

#     # Plot feature importance
#     lgb.plot_importance(booster=lgb_reg, max_num_features=20, grid=True, title='LGBMRegressor')
#     plt.tight_layout()
#     plt.savefig(os.path.join(run_outdir, model_name+'_importances.png'))

#     # Plot learning curves
#     # TODO: note, plot_metric didn't accept 'mae' although it's alias for 'l1' 
#     # TODO: plot_metric requires dict from train(), but train returns 'lightgbm.basic.Booster'??
#     for m in eval_metric:
#         ax = lgb.plot_metric(booster=lgb_reg, metric=m, grid=True)
#         plt.savefig(os.path.join(run_outdir, model_name+'_learning_curve_'+m+'.png'))


# # -------------
# # TPOTRegressor
# # -------------
# # Total evaluation pipelines:  population_size + generations × offspring_size 
# if 'tpot_reg' in ml_models:
#     try:
#         import tpot
#     except ImportError:
#         logger.error('Module not found (tpot)')
    
#     tpot_checkpoint_folder = os.path.join(run_outdir, 'tpot_reg_checkpoints')
#     os.makedirs(tpot_checkpoint_folder)

#     logger.info('\nTrain TPOTRegressor ...')
#     tpot_reg = tpot.TPOTRegressor(generations=100,  # dflt: 100
#                                   population_size=100, # dflt: 100
#                                   offspring_size=100, # dflt: 100
#                                   scoring='neg_mean_squared_error', # dflt: 'neg_mean_squared_error'
#                                   cv=5,
#                                   n_jobs=n_jobs,
#                                   random_state=SEED,
#                                   warm_start=False,
#                                   periodic_checkpoint_folder=tpot_checkpoint_folder,
#                                   verbosity=2,
#                                   disable_update_check=True)
#     t0 = time.time()
#     tpot_reg.fit(xtr, ytr)
#     train_runtime['tpot_reg'] = time.time() - t0
#     logger.info('Runtime: {:.2f} mins'.format(ml_runtime['tpot_reg']/60))
    
#     # Export model as .py script
#     tpot_reg.export(os.path.join(run_outdir, 'tpot_reg_pipeline.py'))

#     # Print scores
#     utils.print_scores(model=tpot_reg, xdata=xvl, ydata=yvl, logger=logger)

#     # Dump preds
#     t0 = time.time()
#     utils.dump_preds(model=tpot_reg, df_data=vl_data, xdata=xvl, target_name=target_name,
#                     path=os.path.join(run_outdir, 'tpot_reg_vl_preds.csv'))
#     logger.info('Predictions runtime: {:.2f} mins'.format(time.time()/60))


