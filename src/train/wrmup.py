from __future__ import print_function
from __future__ import division

import warnings
warnings.filterwarnings('ignore')

import comet_ml
import os

import sys
import itertools
import pandas as pd
import numpy as np
import gzip
import argparse
import sklearn
import time
import datetime
from pprint import pprint

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

# import tensorflow as tf

import keras as ke
from keras import backend as K

from keras.layers import Input, Dense, Dropout, Activation, BatchNormalization
from keras import optimizers
from keras.optimizers import SGD, Adam, RMSprop, Adadelta
from keras.models import Sequential, Model, model_from_json, model_from_yaml
from keras.utils import np_utils, multi_gpu_model

from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping

from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit  
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, roc_auc_score, confusion_matrix, balanced_accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit, KFold # (AP)
from sklearn.model_selection import GroupShuffleSplit, GroupKFold # (AP)
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold # (AP)

from sklearn.model_selection import learning_curve
import lightgbm as lgb

file_path = os.path.dirname(os.path.realpath(__file__))
#lib_path = os.path.abspath(os.path.join(file_path, '..', '..', 'common')) # (AP)
#sys.path.append(lib_path) # (AP)

# Create dir to dump results (AP)
PRJ_NAME = 'wrmup'
OUTDIR = os.path.join(file_path, '../../models', PRJ_NAME)
os.makedirs(OUTDIR, exist_ok=True)

# Utils
# lib_path = '/vol/ml/apartin/Benchmarks/common/'  # (AP)
# sys.path.append(lib_path)  # (AP)

# import attn_utils
import ml_models
import lrn_curve
import classlogger
import utils
SEED = None


# Arg parser
psr = argparse.ArgumentParser(description='input agg csv file')
psr.add_argument('--in',  default=None)
psr.add_argument('--batch',  type=int, default=32)
psr.add_argument('-dr', '--dr_rate',  type=float, default=0.2)
psr.add_argument('--nrows',  type=int, default=None) # (AP)
psr.add_argument('--cv_folds',  type=int, default=1) # (AP)
psr.add_argument('--cv_method',  type=str, default='simple') # (AP)
psr.add_argument('--attn',  type=int, default=0, choices=[0, 1]) # (AP)
psr.add_argument('--n_jobs',  type=int, default=4) # (AP)
psr.add_argument('--mltype',  type=str, default='reg', choices=['reg', 'cls']) # (AP)
psr.add_argument('--ticks',  type=int, default=5) # (AP)
psr.add_argument('-ml', '--model_name',  type=str, default='lgb_reg') # (AP)
psr.add_argument('-sc', '--scaler',  type=str,  choices=['stnd'], default=None)

psr.add_argument('--ep',  type=int, default=200, help='Total number epochs')
psr.add_argument('--epw', type=int, default=50, help='Number of warp-up epochs')

args = vars(psr.parse_args())
pprint(args)


# Get args
data_path = args['in']
attn = bool(args['attn'])
cv_folds = args['cv_folds']
cv_method = args['cv_method']
n_jobs = args['n_jobs']
mltype = args['mltype']
ticks = args['ticks']
model_name = args['model_name']
scaler = args['scaler']


# Set output dir
t = datetime.datetime.now()
t = [t.year, '-', t.month, '-', t.day, '-', 'h', t.hour, '-', 'm', t.minute]
t = ''.join([str(i) for i in t])
if ('nn' in model_name) and (attn is True):
    outdir_name = model_name + '_attn_' + t
elif ('nn' in model_name) and (attn is False):
    outdir_name = model_name + '_fc_' + t
else:
    outdir_name = model_name
outdir = os.path.join(OUTDIR, outdir_name)
os.makedirs(outdir, exist_ok=True)

# Dump args
utils.dump_args(args, outdir=outdir)

# Create logger
logfilename = os.path.join(outdir, 'logfile.log')
lg = classlogger.Logger(logfilename=logfilename) 


EPOCH = args['ep']
EPOCH_WRM = args['epw']
BATCH = args['batch'] # 32
# nb_classes = 2  # for classification task

# PL = 6213   # 38 + 60483
# PS = 6212   # 60483
DR = args['dr_rate']  # 0.2 # Dropout rate


# Custom metrics
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
    df = df.sample(frac=1.0, axis=0, random_state=SEED) # shuffle values
print('Done ({:.3f} mins).\n'.format((time.time()-t0)/60))


# Drop constant features (AP)
print('\nDrop constant features.')
print(df.shape)
col_idx = df.nunique(dropna=True).values==1 # col indexes to drop
df = df.iloc[:, ~col_idx]
print(df.shape)


# ---------------------------
# Split data (hard partition)
# ---------------------------
def split_cell_drug(dff):
    """ Split drug and cell. """
    dff = dff.copy()
    dd_cols = [c for c in df.columns if 'DD_' in c]
    ge_cols = [c for c in df.columns if 'GE_' in c]
    df_dd = dff[dd_cols]
    df_ge = dff[ge_cols]
    print('\ndf_dd.shape', df_dd.shape)
    print('df_ge.shape', df_ge.shape)
    return df_dd, df_ge


def add_lbl_dup(dff, label_name='lb', prffx='_'):
    """ Add col indicating with unique row (label). """
    dff = dff.copy()
    idx_org = dff.index.values
    
    # Sort rows (duplicated rows will be concateneted)
    dff = dff.sort_values(by=dff.columns.tolist())
    # Add boolean col indicating the start of new unique row
    dff = pd.concat([dff.duplicated(keep='first'), dff], axis=1).rename(columns={0: 'd'})

    # Add col indicating a unique row
    c = -1
    v = np.ones((len(dff),))
    for i, x in enumerate(dff['d']):
        # if i % 50000 == 0: print(i)
        if x is False:
            c += 1
            v[i] = int(c)
        else:
            v[i] = c

    dff.insert(loc=1, column=label_name, value=v) 
    dff = dff.reindex(idx_org)
    dff = dff.drop(columns=['d'])
    
    dff[label_name] = dff[label_name].map(lambda x: prffx + str(int(x)))
    return dff


df_dd, df_ge = split_cell_drug(dff=df)


# Get drug label vector
label_name = 'dlb'
df_dd = add_lbl_dup(df_dd, label_name='dlb', prffx='d')
# dlb = add_lbl_dup(df_dd, label_name='dlb', prffx='d')[label_name]

# Get cell label vector
label_name = 'clb'
df_ge = add_lbl_dup(df_ge, label_name='clb', prffx='c')
# clb = add_lbl_dup(df_ge, label_name='clb', prffx='c')[label_name]


# Split data into 2 datasets
split_by = 'c'

wrm_ratio = 0.5
test_size = 1 - wrm_ratio
cv = GroupShuffleSplit(n_splits=2, test_size=test_size, random_state=0)

if split_by == 'c':
    id_grp1, id_grp2 = next(cv.split(df, groups=df_ge[label_name]))  # hard split by cell
elif split_by == 'd':
    id_grp1, id_grp2 = next(cv.split(df, groups=df_dd[label_name]))  # hard split by drug


df1 = df.loc[id_grp1, :]
df2 = df.loc[id_grp2, :]

dfx1 = df1.iloc[:, 1:]
dfx2 = df2.iloc[:, 1:]
dfy1 = df1.iloc[:, 0]
dfy2 = df2.iloc[:, 0]

# idx = int(wrm_ratio * len(df))
# dfx1 = df.iloc[:idx, 1:].copy()
# dfx2 = df.iloc[idx:, 1:].copy()
# dfy1 = df.iloc[:idx, 0].copy()
# dfy2 = df.iloc[idx:, 0].copy()

# print('dfx1.shape:', dfx1.shape)
# print('dfx2.shape:', dfx2.shape)
# print('dfy1.shape:', dfy1.shape)
# print('dfy2.shape:', dfy2.shape)


# --------------
# Scale features
# --------------
scaler = StandardScaler()
dfx1 = scaler.fit_transform(dfx1)
dfx2 = scaler.transform(dfx2)


# -------------------------------
# Common settings for both models
# -------------------------------
# Add common keras callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.75, patience=20, verbose=1, mode='auto',
                              min_delta=0.0001, cooldown=3, min_lr=0.000000001)
early_stop = EarlyStopping(monitor='val_loss', patience=100, verbose=1, mode='auto')

# Model init params 
init_params = {'input_dim': dfx1.shape[1], 'dr_rate': DR, 'attn': attn}


# --------------------------------
# Train 1st model and dump weights
# --------------------------------
# cv = ShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
# tr_idx, vl_idx = next(cv.split(dfx1, dfy1))
xtr1, xte1, ytr1, yte1 = train_test_split(dfx1, dfy1)
print('xtr1.shape:', xtr1.shape)
print('xte1.shape:', xte1.shape)
print('ytr1.shape:', ytr1.shape)
print('yte1.shape:', yte1.shape)

# Define callbacks and fit_params for phase 1 (warm-up)
checkpointer = ModelCheckpoint(filepath=os.path.join(outdir, 'model1.wrm.{epoch:02d}-{val_loss:.2f}.h5'), verbose=0, save_weights_only=False, save_best_only=True)
csv_logger = CSVLogger(filename=os.path.join(outdir, 'model1.wrm.log'))
callback_list = [checkpointer, csv_logger, early_stop, reduce_lr]

fit_params = {'batch_size': BATCH, 'epochs': EPOCH_WRM, 'verbose': 1}
fit_params['validation_data'] = (xte1, yte1)
fit_params['callbacks'] = callback_list

# Get the estimator
model1 = ml_models.get_model(model_name=model_name, init_params=init_params)

# Train model phase 1
history_wrm = model1.model.fit(xtr1, ytr1, **fit_params)
score = model1.model.evaluate(xte1, yte1, verbose=0)
print('val_loss: {:.3f}'.format(score[0]))

# Print plots
model1_wrm_plts_path = os.path.join(outdir, 'model1_wrm_plts')
os.makedirs(model1_wrm_plts_path, exist_ok=True)
ml_models.plot_prfrm_metrics(history=history_wrm, title=f'Model 1: warm-up training',
                             outdir=model1_wrm_plts_path)


# Define path
model1_path = os.path.join(outdir, 'model1.wrm.json')
weights1_path = os.path.join(outdir, 'weights1.wrm.h5')

# wrm model
model_json = model1.model.to_json()
with open(model1_path, 'w') as json_file:
    json_file.write(model_json)

# wrm weights
model1.model.save_weights(weights1_path)
    
# Define callbacks and fit_params for phase 2 (continue training)
checkpointer = ModelCheckpoint(filepath=os.path.join(outdir, 'model1.cnt.h5'), verbose=0, save_weights_only=False, save_best_only=True)
csv_logger = CSVLogger(filename=os.path.join(outdir, 'model1.cnt.log'))
callback_list = [checkpointer, csv_logger, early_stop, reduce_lr]

fit_params = {'batch_size': BATCH, 'epochs': EPOCH - EPOCH_WRM, 'verbose': 1}
fit_params['validation_data'] = (xte1, yte1)
fit_params['callbacks'] = callback_list

# Train model phase 2
history_cnt = model1.model.fit(xtr1, ytr1, **fit_params)
score = model1.model.evaluate(xte1, yte1, verbose=0)
print('val_loss: {:.3f}'.format(score[0]))

# Print plots
model1_cnt_plts_path = os.path.join(outdir, 'model1_cnt_plts')
os.makedirs(model1_cnt_plts_path, exist_ok=True)
ml_models.plot_prfrm_metrics(history=history_cnt, title=f'Model 1: continue training',
                             outdir=model1_cnt_plts_path)



# --------------------------------
# Train 2nd model and dump weights
# --------------------------------
xtr2, xte2, ytr2, yte2 = train_test_split(dfx2, dfy2)
print('xtr2.shape:', xtr2.shape)
print('xte2.shape:', xte2.shape)
print('ytr2.shape:', ytr2.shape)
print('yte2.shape:', yte2.shape)

# Define callbacks and fit_params for phase 1 (warm-up)
checkpointer = ModelCheckpoint(filepath=os.path.join(outdir, 'model2.scratch.h5'), verbose=0, save_weights_only=False, save_best_only=True)
csv_logger = CSVLogger(filename=os.path.join(outdir, 'model2.scratch.log'))
callback_list = [checkpointer, csv_logger, early_stop, reduce_lr]

fit_params = {'batch_size': BATCH, 'epochs': EPOCH, 'verbose': 1}
fit_params['validation_data'] = (xte2, yte2)
fit_params['callbacks'] = callback_list

# Get the estimator
model2 = ml_models.get_model(model_name=model_name, init_params=init_params)

# Train model2 - from scratch
history_from_scratch = model2.model.fit(xtr2, ytr2, **fit_params)
score = model2.model.evaluate(xte2, yte2, verbose=0)
print('val_loss: {:.3f}'.format(score[0]))

# Print plots
model2_from_scratch_plts_path = os.path.join(outdir, 'model2_from_scratch_plts')
os.makedirs(model2_from_scratch_plts_path, exist_ok=True)
ml_models.plot_prfrm_metrics(history=history_from_scratch, title=f'Model 2: training from scratch',
                             outdir=model2_from_scratch_plts_path)


# ------------------
# Load model1 warmed 
# ------------------
# Define callbacks and fit_params for phase 1 (warm-up)
checkpointer = ModelCheckpoint(filepath=os.path.join(outdir, 'model2.from_wrm.h5'), verbose=0, save_weights_only=False, save_best_only=True)
csv_logger = CSVLogger(filename=os.path.join(outdir, 'model2.from_wrm.log'))
callback_list = [checkpointer, csv_logger, early_stop, reduce_lr]

fit_params = {'batch_size': BATCH, 'epochs': EPOCH - EPOCH_WRM, 'verbose': 1}
fit_params['validation_data'] = (xte2, yte2)
fit_params['callbacks'] = callback_list

# Load wamred model1
json_file = open(model1_path, 'r')
model1_wrm_json = json_file.read()
json_file.close()
model1_wrm_loaded = model_from_json(model1_wrm_json)

# Load weights into warmed model                                                                                                                 
model1_wrm_loaded.load_weights(weights1_path)

# Compile loaded model
model1_wrm_loaded.compile(loss='mean_squared_error',
                          optimizer=SGD(lr=0.0001, momentum=0.9),
                          metrics=['mae', r2])

# Train model2 - from wrm
history_from_wrm = model1_wrm_loaded.fit(xtr2, ytr2, **fit_params)
score = model1_wrm_loaded.evaluate(xte2, yte2, verbose=0)
print('val_loss: {:.3f}'.format(score[0]))

# Print plots
model2_from_wrm_plts_path = os.path.join(outdir, 'model2_from_wrm_plts')
os.makedirs(model2_from_wrm_plts_path, exist_ok=True)
ml_models.plot_prfrm_metrics(history=history_from_wrm, title=f'Model 2: training from warmed model1',
                             outdir=model2_from_wrm_plts_path)


print('Done')

