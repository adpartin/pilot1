""" 
Compares optimizers.
"""
from __future__ import print_function, division

import warnings
warnings.filterwarnings('ignore')

# import comet_ml
import os

import sys
from pathlib import Path
import itertools
import gzip
import argparse
import datetime
from time import time
from pprint import pprint

import sklearn
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import keras as ke
from keras import backend as K

from keras.layers import Input, Dense, Dropout, Activation, BatchNormalization
from keras import optimizers
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adamax, Nadam, Adam
from keras.models import Sequential, Model, model_from_json, model_from_yaml
from keras.utils import np_utils, multi_gpu_model
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard

from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit  
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, roc_auc_score, confusion_matrix, balanced_accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from sklearn.model_selection import ShuffleSplit, KFold
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

import lightgbm as lgb

SEED = None
t_start = time()


# Utils
import ml_models
from ml_models import r2_krs
import lrn_curve
import classlogger
import utils


# Import custom callbacks
keras_contrib = '/vol/ml/apartin/projects/keras-contrib/keras_contrib'
sys.path.append(keras_contrib)
from callbacks import *


# File path
# file_path = os.path.dirname(os.path.realpath(__file__))
file_path = Path(__file__).resolve().parent


# Create dir to dump results (AP)
PRJ_NAME = 'top6_cmpr_optimizers'
OUTDIR = file_path / '../../models' / PRJ_NAME
os.makedirs(OUTDIR, exist_ok=True)


# Utils
lib_path = '/vol/ml/apartin/Benchmarks/common/'  # (AP)
sys.path.append(lib_path)  # (AP)


# Arg parser
psr = argparse.ArgumentParser(description='input agg csv file')
psr.add_argument('--in', default=None)
psr.add_argument('--ep', type=int, default=250)
psr.add_argument('--batch', type=int, default=32)
psr.add_argument('--dr', type=float, default=0.2)
psr.add_argument('--cv_folds', type=int, default=1) # (AP)
psr.add_argument('--cv_method', type=str, default='simple') # (AP)
psr.add_argument('--attn', type=int, default=0, choices=[0, 1]) # (AP)
psr.add_argument('--mltype', type=str, default='reg', choices=['reg', 'cls']) # (AP)

psr.add_argument('--skip_ep', type=int, default=3, help='Number of epochs to skip when plotting training curves.')

args = vars(psr.parse_args())
pprint(args)


# Args
data_path = args['in']
attn = bool(args['attn'])
cv_folds = args['cv_folds']
cv_method = args['cv_method']
mltype = args['mltype']
skip_epochs = args['skip_ep']

DR = args['dr']
EPOCH = args['ep']
BATCH = args['batch']
nb_classes = 2

model_name = 'nn_reg'


# Set output dir
if ('nn' in model_name) and (attn is True):
    outdir_name = model_name + '_attn'
elif ('nn' in model_name) and (attn is False):
    outdir_name = model_name + '_fc'
else:
    outdir_name = model_name
outdir = OUTDIR / outdir_name
os.makedirs(outdir, exist_ok=True)


# Dump args
utils.dump_args(args, outdir=outdir)


# Create logger
logfilename = outdir / 'logfile.log'
lg = classlogger.Logger(logfilename=logfilename) 
    

# ---------
# Load data
# ---------
lg.logger.info(f'Loading data ... {data_path}')
t0 = time()
df = pd.read_parquet(data_path, engine='auto', columns=None)
df = df.sample(frac=1.0, axis=0, random_state=SEED).reset_index(drop=True)
lg.logger.info('Done ({:.2f} mins).\n'.format( (time()-t0)/60) )


# ---------------------------
# Extract features and target
# ---------------------------
dfy, dfx = df.iloc[:, 0], df.iloc[:, 1:];  del df

# Scale
scaler = StandardScaler()
dfx = pd.DataFrame( scaler.fit_transform(dfx) ).astype(np.float32)


# -----
# Comet
# -----
# args['comet_prj_name'] = str(PRJ_NAME)
# if 'nn' in model_name and attn is True:
#     args['comet_set_name'] = model_name + '_attn'
# elif 'nn' in model_name and attn is False:
#     args['comet_set_name'] = model_name + '_fc'
# else:
#     args['comet_set_name'] = model_name
    

# ---------
# CV scheme
# ---------
test_size = 0.2
if mltype == 'cls':
    # Classification
    if cv_method == 'simple':
        if cv_folds == 1:
            cv = ShuffleSplit(n_splits=cv_folds, test_size=test_size, random_state=SEED)
        else:
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=SEED)
    elif cv_method == 'stratify':
        if cv_folds == 1:
            cv = StratifiedShuffleSplit(n_splits=cv_folds, test_size=test_size, random_state=SEED)
        else:
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=SEED)

elif mltype == 'reg':
    # Regression
    if cv_folds == 1:
        cv = ShuffleSplit(n_splits=cv_folds, test_size=test_size, random_state=SEED)
    else:
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=SEED)

        
# Split the data        
id_tr, id_te = next(cv.split(dfx, dfy))
xtr, ytr = dfx.loc[id_tr,:], dfy.loc[id_tr]
xte, yte = dfx.loc[id_te,:], dfy.loc[id_te]
lg.logger.info('xtr.shape: {}'.format(xtr.shape))
lg.logger.info('xte.shape: {}'.format(xte.shape))
lg.logger.info('ytr.shape: {}'.format(ytr.shape))
lg.logger.info('yte.shape: {}'.format(yte.shape))


# Dump train and val sets
# (these can be used later for ensemble predictions)
df_tr = pd.concat([ytr, xtr], axis=1)
df_te = pd.concat([yte, xte], axis=1)
df_tr.to_csv(outdir/'df_tr.csv', index=False)
df_te.to_csv(outdir/'df_te.csv', index=False)


# Define various learning rates
# https://keras.io/optimizers/
# https://stats.stackexchange.com/questions/313278/no-change-in-accuracy-using-adam-optimizer-when-sgd-works-fine
opts = [
    ('CLR', CyclicLR(base_lr=1e-4, max_lr=1e-2, mode='triangular')),
    ('Adam', Adam(lr=1e-4, amsgrad=False)), # lr=0.001
    ('SGDMomentum', SGD(lr=1e-3, momentum=0.9, nesterov=False)),  # 0.01
    ('SGDMomentumNestrov', SGD(lr=1e-3, momentum=0.9, nesterov=True)),
    ('RMSprop', RMSprop(lr=1e-4)),  # lr=1e-3
    # ('Adagrad', Adagrad(lr=0.0001)),
    ('Adadelta', Adadelta(lr=1e-4)),
    ('AdamAmsgrad', Adam(lr=1e-4, amsgrad=True)),
    ('Nadam', Nadam(lr=1e-4)),
]


# Keras callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.75, patience=20, verbose=1, mode='auto',
                              min_delta=0.0001, cooldown=3, min_lr=0.000000001)
early_stop = EarlyStopping(monitor='val_loss', patience=60, verbose=1, mode='auto')

data_to_plot = {}
metric_to_plot = 'val_r2_krs'


# Iterate over optimizers
for opt_name, optimizer in opts:
    lg.logger.info(f'\nTraining with {opt_name} ...')
    opt_outdir = outdir / opt_name
    os.makedirs(opt_outdir, exist_ok=True)

    # Keras callbacks
    checkpointer = ModelCheckpoint(str(opt_outdir / f'{opt_name}.model.h5'), verbose=0, save_weights_only=False, save_best_only=True)
    csv_logger = CSVLogger(opt_outdir / f'{opt_name}.model.log')

    # Callbacks list
    callback_list = [checkpointer, csv_logger, early_stop, reduce_lr]
    
    if opt_name.lower() == 'clr':
        callback_list.append(optimizer)  # add the cyclical lr into callback
        optimizer = SGD(lr=0.001, momentum=0.9)
        
    # fit_params
    fit_params = {'batch_size': BATCH, 'epochs': EPOCH, 'verbose': 1}
    fit_params['validation_data'] = (xte, yte)
    fit_params['callbacks'] = callback_list

    # Get the estimator
    init_params = {'input_dim': dfx.shape[1], 'dr_rate': DR, 'attn': attn,
                   'optimizer': optimizer}
    model = ml_models.get_model(model_name=model_name, init_params=init_params)

    # Train model
    t0 = time()
    history = model.model.fit(xtr, ytr, **fit_params)
    fit_runtime = time() - t0
    lg.logger.info('fit_runtime: {:.3f} mins'.format(fit_runtime/60))

    # Print score
    score = model.model.evaluate(xte, yte, verbose=0)
    lg.logger.info('val_loss: {:.5f}'.format(score[0]))

    # Print plots
    model_plts_path = opt_outdir / f'{opt_name}_plts'
    os.makedirs(model_plts_path, exist_ok=True)
    ml_models.plot_prfrm_metrics(history=history, title=f'Training curves: {opt_name}',
                                 skip_epochs=skip_epochs, add_lr=True,
                                 outdir=model_plts_path)

    # Dump history
    pp = pd.DataFrame(history.history)
    pp['epoch'] = np.asarray(history.epoch) + 1
    pp.to_csv(outdir/f'{opt_name}_keras_history.csv', index=False)
    

    # Dump model
    model.dump_model(outpath=opt_outdir/f'model.{opt_name}.h5')
    
    # Define path to dump model and weights
#     model_path = opt_outdir / f'{opt_name}.model.json'
#     weights_path = opt_outdir / f'{opt_name}.weights.h5'

#     # Save model
#     model_json = model.model.to_json()
#     with open(model_path, 'w') as json_file:
#         json_file.write(model_json)

#     # Save weights
#     model.model.save_weights(weights_path)
    
    del model, init_params, fit_params

    # Data for plots
    data_to_plot[opt_name] = (pp['epoch'][skip_epochs:].values, pp[metric_to_plot][skip_epochs:].values)

    
fig, ax = plt.subplots()
for opt_name, v in data_to_plot.items():
    eps_vec = v[0]
    values_vec = v[1]
    ax.plot(eps_vec, values_vec, label=opt_name)
    
ax.grid(True)
ax.set_xlabel('epoch')
ax.set_ylabel(metric_to_plot)
plt.title(f'Top6: {metric_to_plot}')
plt.legend(loc='best')
plt.savefig(outdir/'train_curve_all_lr.png', bbox_inches='tight')

lg.logger.info('\nProgram runtime: {:.2f} mins'.format( (time() - t_start)/60 ))
lg.logger.info('Done.')
