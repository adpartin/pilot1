from __future__ import print_function, division

import warnings
warnings.filterwarnings('ignore')

# from comet_ml import Experiment
import os

import sys
import pathlib
import argparse
import datetime
from time import time
from pprint import pprint
from glob import glob

import sklearn
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import keras as ke
from keras import backend as K

from keras.models import Sequential, Model, model_from_json, model_from_yaml, load_model
from keras.optimizers import SGD, Adam, RMSprop, Adadelta
from keras.utils import np_utils, multi_gpu_model
from keras.callbacks import Callback, ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping

from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit, KFold # (AP)
from sklearn.model_selection import GroupShuffleSplit, GroupKFold # (AP)
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold # (AP)

SEED = None
t_start = time()


# Utils
import ml_models
import classlogger
import utils


# Import custom callbacks
keras_contrib = '/vol/ml/apartin/projects/keras-contrib/keras_contrib'
sys.path.append(keras_contrib)
from callbacks import *


# File path
# file_path = os.path.dirname(os.path.realpath(__file__))
file_path = pathlib.Path(__file__).resolve().parent


# Path - create dir to dump results (AP)
PRJ_NAME = 'candle_challenge_prb'
TR_PHASE = 'wrm'
OUTDIR = file_path / '../../models' / PRJ_NAME / TR_PHASE
os.makedirs(OUTDIR, exist_ok=True)

DATADIR = file_path / '../../models' / PRJ_NAME / 'data'


# Arg parser
psr = argparse.ArgumentParser(description='input agg csv file')
psr.add_argument('--batch',  type=int, default=32)
psr.add_argument('-dr', '--dr_rate',  type=float, default=0.2)
psr.add_argument('--attn',  type=int, default=0, choices=[0, 1]) # (AP)
# psr.add_argument('--n_jobs',  type=int, default=4) # (AP)
psr.add_argument('-ml', '--model_name',  type=str, default='nn_reg') # (AP)
psr.add_argument('--ep', type=int, default=250, help='Total number epochs')
# psr.add_argument('--lr',  type=float, default=0.0005, help='Learning rate')
psr.add_argument('--split_method', type=str, choices=['rnd', 'hrd'], default='hrd')
psr.add_argument('--skip_ep', type=int, default=3, help='Number of epochs to skip when plotting training curves.')

args = vars(psr.parse_args())
pprint(args)


# Args
EPOCH = args['ep']
BATCH = args['batch']
DR = args['dr_rate']
# LR = args['lr']
attn = bool(args['attn'])
model_name = args['model_name']
split_method = args['split_method']
skip_epochs = args['skip_ep']


# Data path
if split_method == 'rnd':
    data_path = DATADIR / 'rnd' / 'df_wrm.parquet'
    outdir = OUTDIR / 'rnd'
elif split_method == 'hrd':
    data_path = DATADIR / 'hrd' / 'df_wrm.parquet'
    outdir = OUTDIR / 'hrd'
os.makedirs(outdir, exist_ok=True)


# Dump args
utils.dump_args(args, outdir=outdir)


# Logger
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
lg.logger.info(f'Loading data ... {data_path}')
t0 = time()
df = pd.read_parquet(data_path, engine='auto', columns=None)
df = df.sample(frac=1.0, axis=0, random_state=SEED)
lg.logger.info('Done ({:.2f} mins).\n'.format( (time()-t0)/60) )


# ---------------------------
# Extract features and target
# ---------------------------
dfy, dfx = df.iloc[:, 0], df.iloc[:, 1:];  del df

# Scale
scaler = StandardScaler()
dfx = pd.DataFrame( scaler.fit_transform(dfx) ).astype(np.float32)
joblib.dump(scaler, outdir / 'scaler.pkl')


# -----
# Comet
# -----
# comet_api_key = os.environ.get('COMET_API_KEY')
# comet_prg_name = PRJ_NAME
# comet_set_name = TR_PHASE

# experiment = Experiment(api_key=comet_api_key, project_name=comet_prg_name)
# experiment.set_name(comet_set_name)
# experiment.add_tag(TR_PHASE)


# -------------------
# Split to train/text
# -------------------
xtr, xte, ytr, yte = train_test_split(dfx, dfy)
lg.logger.info('xtr.shape: {}'.format(xtr.shape))
lg.logger.info('xte.shape: {}'.format(xte.shape))
lg.logger.info('ytr.shape: {}'.format(ytr.shape))
lg.logger.info('yte.shape: {}'.format(yte.shape))


# ---------------------
# Train 'warm-up' model
# ---------------------
tag = 'wrm'

# Callbacks (custom)
clr_triangular = CyclicLR(base_lr=0.0001, max_lr=0.001, mode='triangular')

# Keras callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.75, patience=20, verbose=1, mode='auto',
                              min_delta=0.0001, cooldown=3, min_lr=0.000000001)
early_stop = EarlyStopping(monitor='val_loss', patience=60, verbose=1, mode='auto')
checkpointer = ModelCheckpoint(str(outdir / 'model.wrm.ep_{epoch:d}-val_loss_{val_loss:.5f}.h5'), verbose=0, save_weights_only=False, save_best_only=False)
csv_logger = CSVLogger(outdir / f'model.{tag}.log')

# Callbacks list
callback_list = [checkpointer, csv_logger, early_stop, reduce_lr,  # keras callbacks
                 clr_triangular]  # custom callbacks

# fit_params
fit_params = {'batch_size': BATCH, 'epochs': EPOCH, 'verbose': 1}
fit_params['validation_data'] = (xte, yte)
fit_params['callbacks'] = callback_list

# Get the estimator
init_params = {'input_dim': dfx.shape[1], 'dr_rate': DR, 'attn': attn}
# init_params = {'input_dim': dfx.shape[1], 'dr_rate': DR, 'attn': attn, 'lr': LR}
model = ml_models.get_model(model_name=model_name, init_params=init_params)

# Train model - warm-up phase
t0 = time()
history = model.model.fit(xtr, ytr, **fit_params)
fit_runtime = time() - t0
lg.logger.info('fit_runtime: {:.3f} mins'.format(fit_runtime/60))

# Print score
score = model.model.evaluate(xte, yte, verbose=0)
lg.logger.info('val_loss: {:.5f}'.format(score[0]))

# Print plots
model_plts_path = outdir / f'model_{tag}_plts'
os.makedirs(model_plts_path, exist_ok=True)
ml_models.plot_prfrm_metrics(history=history,
                             title=f'Warm-up training',  # LR: {LR}
                             skip_epochs=skip_epochs,
                             add_lr=True,
                             outdir=model_plts_path)

# Dump history
pd.DataFrame(history.history).to_csv(outdir/'keras_history.csv', index=False)

# Define path to dump model and weights
model_path = outdir / f'model.{tag}.json'
weights_path = outdir / f'weights.{tag}.h5'

# Save model
model_json = model.model.to_json()
with open(model_path, 'w') as json_file:
    json_file.write(model_json)

# Save weights
model.model.save_weights(weights_path)

lg.logger.info('\nProgram runtime: {:.2f} mins'.format( (time() - t_start)/60 ))
lg.logger.info('Done.')



















# # --------------------------------
# # Train 1st model and dump weights
# # --------------------------------
# # cv = ShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
# # tr_idx, vl_idx = next(cv.split(dfx1, dfy1))
# xtr1, xte1, ytr1, yte1 = train_test_split(dfx1, dfy1)
# print('xtr1.shape:', xtr1.shape)
# print('xte1.shape:', xte1.shape)
# print('ytr1.shape:', ytr1.shape)
# print('yte1.shape:', yte1.shape)

# # Define callbacks and fit_params for phase 1 (warm-up)
# checkpointer = ModelCheckpoint(os.path.join(outdir, 'model1.wrm.{epoch:02d}-{val_loss:.2f}.h5'), verbose=0, save_weights_only=False, save_best_only=True)
# csv_logger = CSVLogger(filename=os.path.join(outdir, 'model1.wrm.log'))
# callback_list = [checkpointer, csv_logger, early_stop, reduce_lr]

# fit_params = {'batch_size': BATCH, 'epochs': EPOCH_WRM, 'verbose': 1}
# fit_params['validation_data'] = (xte1, yte1)
# fit_params['callbacks'] = callback_list

# # Get the estimator
# model1 = ml_models.get_model(model_name=model_name, init_params=init_params)

# # Train model phase 1
# history_wrm = model1.model.fit(xtr1, ytr1, **fit_params)
# score = model1.model.evaluate(xte1, yte1, verbose=0)
# print('val_loss: {:.3f}'.format(score[0]))

# # Print plots
# model1_wrm_plts_path = os.path.join(outdir, 'model1_wrm_plts')
# os.makedirs(model1_wrm_plts_path, exist_ok=True)
# ml_models.plot_prfrm_metrics(history=history_wrm, title=f'Model 1: warm-up training',
#                              outdir=model1_wrm_plts_path)


# # Define path
# model1_path = os.path.join(outdir, 'model1.wrm.json')
# weights1_path = os.path.join(outdir, 'weights1.wrm.h5')

# # wrm model
# model_json = model1.model.to_json()
# with open(model1_path, 'w') as json_file:
#     json_file.write(model_json)

# # wrm weights
# model1.model.save_weights(weights1_path)
    
# # Define callbacks and fit_params for phase 2 (continue training)
# checkpointer = ModelCheckpoint(filepath=os.path.join(outdir, 'model1.cnt.h5'), verbose=0, save_weights_only=False, save_best_only=True)
# csv_logger = CSVLogger(filename=os.path.join(outdir, 'model1.cnt.log'))
# callback_list = [checkpointer, csv_logger, early_stop, reduce_lr]

# fit_params = {'batch_size': BATCH, 'epochs': EPOCH - EPOCH_WRM, 'verbose': 1}
# fit_params['validation_data'] = (xte1, yte1)
# fit_params['callbacks'] = callback_list

# # Train model phase 2
# history_cnt = model1.model.fit(xtr1, ytr1, **fit_params)
# score = model1.model.evaluate(xte1, yte1, verbose=0)
# print('val_loss: {:.3f}'.format(score[0]))

# # Print plots
# model1_cnt_plts_path = os.path.join(outdir, 'model1_cnt_plts')
# os.makedirs(model1_cnt_plts_path, exist_ok=True)
# ml_models.plot_prfrm_metrics(history=history_cnt, title=f'Model 1: continue training',
#                              outdir=model1_cnt_plts_path)



# # --------------------------------
# # Train 2nd model and dump weights
# # --------------------------------
# xtr2, xte2, ytr2, yte2 = train_test_split(dfx2, dfy2)
# print('xtr2.shape:', xtr2.shape)
# print('xte2.shape:', xte2.shape)
# print('ytr2.shape:', ytr2.shape)
# print('yte2.shape:', yte2.shape)

# # Define callbacks and fit_params for phase 1 (warm-up)
# checkpointer = ModelCheckpoint(filepath=os.path.join(outdir, 'model2.scratch.h5'), verbose=0, save_weights_only=False, save_best_only=True)
# csv_logger = CSVLogger(filename=os.path.join(outdir, 'model2.scratch.log'))
# callback_list = [checkpointer, csv_logger, early_stop, reduce_lr]

# fit_params = {'batch_size': BATCH, 'epochs': EPOCH, 'verbose': 1}
# fit_params['validation_data'] = (xte2, yte2)
# fit_params['callbacks'] = callback_list

# # Get the estimator
# model2 = ml_models.get_model(model_name=model_name, init_params=init_params)

# # Train model2 - from scratch
# history_from_scratch = model2.model.fit(xtr2, ytr2, **fit_params)
# score = model2.model.evaluate(xte2, yte2, verbose=0)
# print('val_loss: {:.3f}'.format(score[0]))

# # Print plots
# model2_from_scratch_plts_path = os.path.join(outdir, 'model2_from_scratch_plts')
# os.makedirs(model2_from_scratch_plts_path, exist_ok=True)
# ml_models.plot_prfrm_metrics(history=history_from_scratch, title=f'Model 2: training from scratch',
#                              outdir=model2_from_scratch_plts_path)


# # ------------------
# # Load model1 warmed 
# # ------------------
# # Define callbacks and fit_params for phase 1 (warm-up)
# checkpointer = ModelCheckpoint(filepath=os.path.join(outdir, 'model2.from_wrm.h5'), verbose=0, save_weights_only=False, save_best_only=True)
# csv_logger = CSVLogger(filename=os.path.join(outdir, 'model2.from_wrm.log'))
# callback_list = [checkpointer, csv_logger, early_stop, reduce_lr]

# fit_params = {'batch_size': BATCH, 'epochs': EPOCH - EPOCH_WRM, 'verbose': 1}
# fit_params['validation_data'] = (xte2, yte2)
# fit_params['callbacks'] = callback_list

# # Load wamred model1
# json_file = open(model1_path, 'r')
# model1_wrm_json = json_file.read()
# json_file.close()
# model1_wrm_loaded = model_from_json(model1_wrm_json)

# # Load weights into warmed model                                                                                                                 
# model1_wrm_loaded.load_weights(weights1_path)

# # Compile loaded model
# model1_wrm_loaded.compile(loss='mean_squared_error',
#                           optimizer=SGD(lr=0.0001, momentum=0.9),
#                           metrics=['mae', r2])

# # Train model2 - from wrm
# history_from_wrm = model1_wrm_loaded.fit(xtr2, ytr2, **fit_params)
# score = model1_wrm_loaded.evaluate(xte2, yte2, verbose=0)
# print('val_loss: {:.3f}'.format(score[0]))

# # Print plots
# model2_from_wrm_plts_path = os.path.join(outdir, 'model2_from_wrm_plts')
# os.makedirs(model2_from_wrm_plts_path, exist_ok=True)
# ml_models.plot_prfrm_metrics(history=history_from_wrm, title=f'Model 2: training from warmed model1',
#                              outdir=model2_from_wrm_plts_path)

# print('Done')



