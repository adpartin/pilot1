from __future__ import print_function, division

import warnings
warnings.filterwarnings('ignore')

import comet_ml
import os

import sys
import pathlib
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
from keras.optimizers import SGD, Adam, RMSprop, Adadelta
from keras.models import Sequential, Model, model_from_json, model_from_yaml
from keras.utils import np_utils, multi_gpu_model
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard

from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit  
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, roc_auc_score, confusion_matrix, balanced_accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from sklearn.model_selection import ShuffleSplit, KFold # (AP)
from sklearn.model_selection import GroupShuffleSplit, GroupKFold # (AP)
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold # (AP)

import lightgbm as lgb

SEED = None
t_start = time()


# Utils
import ml_models
import lrn_curve
import classlogger
import utils
SEED = None


# Import custom callbacks
keras_contrib = '/vol/ml/apartin/projects/keras-contrib/keras_contrib'
sys.path.append(keras_contrib)
from callbacks import *


# File path
# file_path = os.path.dirname(os.path.realpath(__file__))
file_path = pathlib.Path(__file__).resolve().parent
#lib_path = os.path.abspath(os.path.join(file_path, '..', '..', 'common')) # (AP)
#sys.path.append(lib_path) # (AP)


# Create dir to dump results (AP)
PRJ_NAME = pathlib.Path('top6_lrn_crv_cyclr')
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
# psr.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
psr.add_argument('--nrows', type=int, default=None) # (AP)
psr.add_argument('--cv_folds', type=int, default=1) # (AP)
psr.add_argument('--cv_method', type=str, default='simple') # (AP)
psr.add_argument('--attn', type=int, default=0, choices=[0, 1]) # (AP)
psr.add_argument('--n_jobs', type=int, default=4) # (AP)
psr.add_argument('--mltype', type=str, default='reg', choices=['reg', 'cls']) # (AP)
psr.add_argument('--ticks', type=int, default=5) # (AP)
psr.add_argument('-ml', '--model_name', type=str, default='nn_reg') # (AP)
psr.add_argument('-sc', '--scaler', type=str,  choices=['raw', 'stnd'], default='stnd') 

args = vars(psr.parse_args())
pprint(args)


# Args
data_path = args['in']
attn = bool(args['attn'])
cv_folds = args['cv_folds']
cv_method = args['cv_method']
n_jobs = args['n_jobs']
mltype = args['mltype']
ticks = args['ticks']
model_name = args['model_name']
scaler = args['scaler']

PL = 6213   # 38 + 60483
# PS = 6212   # 60483
DR = args['dr']
EPOCH = args['ep']
BATCH = args['batch']
# LR = args['lr']
nb_classes = 2


# Set output dir
t = datetime.datetime.now()
t = [t.year, '-', t.month, '-', t.day, '-', 'h', t.hour, '-', 'm', t.minute]
t = ''.join([str(i) for i in t])
if ('nn' in model_name) and (attn is True):
    outdir_name = 'lrn_crv_' + model_name + '_attn_' + t
elif ('nn' in model_name) and (attn is False):
    outdir_name = 'lrn_crv_' + model_name + '_fc_' + t
else:
    outdir_name = 'lrn_crv_' + model_name
outdir = OUTDIR / outdir_name
os.makedirs(outdir, exist_ok=True)


# Dump args
utils.dump_args(args, outdir=outdir)


# Create logger
logfilename = outdir / 'logfile.log'
lg = classlogger.Logger(logfilename=logfilename) 
    

# Custom metrics
def r2(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res/(SS_tot + K.epsilon()))

    
# ---------
# Load data
# ---------
print(f'Loading data ... {data_path}')
t0 = time()
if 'csv' in data_path:
    # df = (pd.read_csv(data_path, skiprows=1).values).astype('float32')
    df = pd.read_csv(data_path, skiprows=1, dtype='float32', nrows=args['nrows']).values # (AP)
elif 'parquet' in data_path:
    df = pd.read_parquet(data_path, engine='auto', columns=None) # (AP)
    df = df.sample(frac=1.0, axis=0, random_state=SEED).values # shuffle values
print('Done ({:.3f} mins).\n'.format( (time()-t0)/60 ))
    
if mltype == 'cls':
    df_y = df[:, 0].astype('int')
    Y_onehot = np_utils.to_categorical(df_y, nb_classes)
    print('Y_onehot.shape:', Y_onehot.shape)
    
    # y_integers = np.argmax(df_y, axis=1)
    y_integers = df_y
    class_weights = compute_class_weight('balanced', np.unique(y_integers), y_integers)
    d_class_weights = dict(enumerate(class_weights))
    # print(d_class_weights)

    # print('bincount(y):\n', pd.Series(df_y).value_counts())
else:
    df_y = df[:, 0]

df_x = df[:, 1:PL].astype(np.float32)


# Scale features
if scaler is not None:
    scaler = StandardScaler()
    df_x = scaler.fit_transform(df_x).astype(np.float32)

print('df_x.shape:', df_x.shape)
print('df_y.shape:', df_y.shape)


# -----
# Comet
# -----
args['comet_prj_name'] = str(PRJ_NAME)
if 'nn' in model_name and attn is True:
    args['comet_set_name'] = model_name + '_attn'
elif 'nn' in model_name and attn is False:
    args['comet_set_name'] = model_name + '_fc'
else:
    args['comet_set_name'] = model_name
    

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
        cv = ShuffleSplit(n_splits=cv_folds, test_size=0.2, random_state=SEED)
    else:
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=SEED)


# -------------------------
# Learning curve
# -------------------------
# ML model params
if model_name == 'lgb_reg':
    init_prms = {'n_jobs': n_jobs, 'random_state': SEED}
    fit_prms = {'verbose': False}  # 'early_stopping_rounds': 10,
if model_name == 'nn_reg':
    init_prms = {'input_dim': df_x.shape[1], 'dr_rate': DR, 'attn': attn}
    # init_prms = {'input_dim': df_x.shape[1], 'dr_rate': DR, 'attn': attn, 'lr': LR}
    fit_prms = {'batch_size': BATCH, 'epochs': EPOCH, 'verbose': 1}

print(f'\nLearning curve ({model_name}) ...')
outdir_ = outdir / model_name
os.makedirs(outdir_, exist_ok=True)

# Run learning curve
t0 = time()
lrn_curve_scores = lrn_curve.my_learning_curve(
    X=df_x, Y=df_y,
    mltype=mltype,
    model_name=model_name,
    fit_params=fit_prms,
    init_params=init_prms,
    args=args,
    lr_curve_ticks=ticks,
    data_sizes_frac=None,
    metrics=None,
    cv=cv,
    groups=None,
    n_jobs=n_jobs, random_state=SEED, logger=lg.logger, outdir=outdir_)
print('Runtime: {:.3f} mins'.format( (time()-t0)/60 ))

# Dump results
lrn_curve_scores.to_csv(outdir / (model_name + '_lrn_crv_scores.csv'), index=False)

lg.logger.info('\nProgram runtime: {:.2f} mins'.format( (time() - t_start)/60 ))
lg.logger.info('Done.')
