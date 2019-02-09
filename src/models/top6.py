import itertools
import pandas as pd
import numpy as np
import os
import sys
import gzip
import argparse
import sklearn
import time
import datetime

import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

import tensorflow as tf

import keras as ke
from keras import backend as K

from keras.layers import Input, Dense, Dropout, Activation, BatchNormalization
from keras import optimizers
from keras.optimizers import SGD, Adam, RMSprop, Adadelta
from keras.models import Sequential, Model, model_from_json, model_from_yaml
from keras.utils import np_utils, multi_gpu_model

from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping
from keras.callbacks import TensorBoard  # (AP)

from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit  
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, roc_auc_score, confusion_matrix, balanced_accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from sklearn.model_selection import ShuffleSplit, KFold # (AP)
from sklearn.model_selection import GroupShuffleSplit, GroupKFold # (AP)
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold # (AP)

from sklearn.model_selection import learning_curve
import lightgbm as lgb

file_path = os.path.dirname(os.path.realpath(__file__))
#lib_path = os.path.abspath(os.path.join(file_path, '..', '..', 'common')) # (AP)
#sys.path.append(lib_path) # (AP)

# Utils
lib_path = '/vol/ml/apartin/Benchmarks/common/'  # (AP)
sys.path.append(lib_path)  # (AP)
import attn_utils
import ml_models
import lrn_curve
import classlogger
SEED = None


# Arg parser
psr = argparse.ArgumentParser(description='input agg csv file')
psr.add_argument('--in',  default=None)
psr.add_argument('--ep',  type=int, default=300)
psr.add_argument('--batch',  type=int, default=32)
psr.add_argument('-dr', '--dr_rate',  type=float, default=0.2)
psr.add_argument('--nrows',  type=int, default=None) # (AP)
psr.add_argument('--cv_folds',  type=int, default=1) # (AP)
psr.add_argument('--cv_method',  type=str, default='simple') # (AP)
psr.add_argument('--attn',  type=int, default=0, choices=[0, 1]) # (AP)
psr.add_argument('--n_jobs',  type=int, default=4) # (AP)
psr.add_argument('--mltype',  type=str, default='reg', choices=['reg', 'cls']) # (AP)
psr.add_argument('--ticks',  type=int, default=5) # (AP)
args = vars(psr.parse_args())
print(args)


# Get args
data_path = args['in']
attn = bool(args['attn'])
cv_folds = args['cv_folds']
cv_method = args['cv_method']
n_jobs = args['n_jobs']
mltype = args['mltype']
ticks = args['ticks']


# Set output dir
t = datetime.datetime.now()
t = [t.year, '-', t.month, '-', t.day, '_', 'h', t.hour, '-', 'm', t.minute]
t = ''.join([str(i) for i in t])
outdir = os.path.join('./', 'lrn_curve_' + mltype + '_' + t)
os.makedirs(outdir, exist_ok=True)


# Create logger
logfilename = os.path.join(outdir, 'logfile.log')
lg = classlogger.Logger(logfilename=logfilename) 


EPOCH = args['ep']
BATCH = args['batch'] # 32
nb_classes = 2

PL = 6213   # 38 + 60483
PS = 6212   # 60483
DR = args['dr_rate']  # 0.2 # Dropout rate 
    
    
def r2(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res/(SS_tot + K.epsilon()))

    
# ---------
# Load data
# ---------
print(f'Loading data ... {data_path}')
t0 = time.time()
if 'csv' in data_path:
    # df = (pd.read_csv(data_path, skiprows=1).values).astype('float32')
    df = pd.read_csv(data_path, skiprows=1, dtype='float32', nrows=args['nrows']).values # (AP)
elif 'parquet' in data_path:
    df = pd.read_parquet(data_path, engine='auto', columns=None) # (AP)
    df = df.sample(frac=1.0, axis=0, random_state=SEED).values # shuffle values
print('Done ({:.3f} mins).\n'.format((time.time()-t0)/60))
    
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
scaler = StandardScaler()
df_x = scaler.fit_transform(df_x)

print('df_x.shape:', df_x.shape)
print('df_y.shape:', df_y.shape)


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


# # -------------------------
# # Learning curve (lightgbm)
# # -------------------------
# # ML model params
# model_name = 'lgb_reg'
# init_prms = {'n_jobs': n_jobs, 'random_state': SEED}
# fit_prms = {'verbose': False}  # 'early_stopping_rounds': 10,

# print(f'\nLearning curve ({model_name}) ...')
# outdir_ = os.path.join(outdir, model_name)
# os.makedirs(outdir_, exist_ok=True)

# # Run learning curve
# t0 = time.time()
# lrn_curve_scores = lrn_curve.my_learning_curve(
#     X=df_x, Y=df_y,
#     mltype=mltype,
#     model_name=model_name,
#     fit_params=fit_prms,
#     init_params=init_prms,
#     args=args,
#     lr_curve_ticks=ticks,
#     data_sizes_frac=None,
#     metrics=None,
#     cv=cv,
#     groups=None,
#     n_jobs=n_jobs, random_state=SEED, logger=None, outdir=outdir_)
# print('Runtime: {:.3f} mins'.format((time.time()-t0)/60))

# # Dump results
# lrn_curve_scores.to_csv(os.path.join(outdir, model_name + '_lrn_curve_scores.csv'), index=False)


# -------------------
# Learning curve (nn)
# -------------------
# ML model params
model_name = 'nn_reg'
init_prms = {'input_dim': df_x.shape[1], 'dr_rate': DR, 'attn': attn}
fit_prms = {'batch_size': BATCH, 'epochs': EPOCH, 'verbose': 1}
    
print(f'\nLearning curve ({model_name}) ...')
outdir_ = os.path.join(outdir, model_name)
os.makedirs(outdir_, exist_ok=True)

# Run learning curve
t0 = time.time()
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
    n_jobs=n_jobs, random_state=SEED, logger=None, outdir=outdir_)
print('Runtime: {:.3f} mins'.format((time.time()-t0)/60))

# Dump results
lrn_curve_scores.to_csv(os.path.join(outdir, model_name + '_lrn_curve_scores.csv'), index=False)


