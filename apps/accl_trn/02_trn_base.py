"""
Train base model. This model will be used to continue training.
"""
from __future__ import print_function, division

import warnings
warnings.filterwarnings('ignore')

import os
import sys
from pathlib import Path
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
from sklearn.model_selection import ShuffleSplit, KFold
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

SEED = 42
t_start = time()


# Utils
import ml_models
import classlogger
import utils


# Import custom callbacks
keras_contrib = '/vol/ml/apartin/projects/keras-contrib/keras_contrib/callbacks'
sys.path.append(keras_contrib)
from cyclical_learning_rate import CyclicLR

# File path
# file_path = os.path.dirname(os.path.realpath(__file__))
file_path = Path(__file__).resolve().parent


# Path
PRJ_NAME = file_path.name
PRJ_DIR = file_path / '../../out' / PRJ_NAME
DATADIR = PRJ_DIR / 'data'


# Arg parser
psr = argparse.ArgumentParser(description='input agg csv file')
psr.add_argument('--ep', type=int, default=350, help='Total number of epochs.')
psr.add_argument('--tr_phase',  type=str, choices=['wrm', 'ref'], default='wrm')
psr.add_argument('--attn', action='store_true', default=False, help='whether to use attention layer.')
psr.add_argument('--split_by', type=str, choices=['cell', 'drug', 'both', 'none'], default='cell',
                 help='Specify how to disjointly partition the dataset: \
                 `cell` (disjoint on cell), `drug` (disjoint on drug), \
                 `both` (disjoint on cell and drug), `none` (random split).')

psr.add_argument('-ml', '--model_name', type=str, default='nn_reg')
psr.add_argument('--batch', type=int, default=32)
psr.add_argument('--dr_rate', type=float, default=0.2)
psr.add_argument('--base_clr', type=float, default=1e-4, help='Base learning rate for cyclical learning rate.')
psr.add_argument('--max_clr', type=float, default=1e-3, help='Max learning rate for cyclical learning rate.')
psr.add_argument('--skp_ep', type=int, default=10, help='Number of epochs to skip when plotting training curves.')


args = vars(psr.parse_args())
pprint(args)


# Args
EPOCH = args['ep']
BATCH = args['batch']
DR = args['dr_rate']
attn = args['attn']
model_name = args['model_name']
split_by = args['split_by']
skp_ep = args['skp_ep']
base_clr = args['base_clr']
max_clr = args['max_clr']
tr_phase = args['tr_phase']


if attn is True:
    nn_type = 'attn'
else:
    nn_type = 'fc'    


# Path and outdir
# data_path = DATADIR / ('split_by_' + split_by) / f'df_{tr_phase}.parquet'
data_path = DATADIR / ('split_by_' + split_by)
outdir = PRJ_DIR / (tr_phase + '_' + nn_type) / ('split_by_' + split_by)
os.makedirs(outdir, exist_ok=True)


# Dump args
utils.dump_args(args, outdir=outdir)


# Logger
logfilename = outdir/'logfile.log'
lg = classlogger.Logger(logfilename=logfilename)


# ---------
# Load data
# ---------
def load_data(datapath):
    data = pd.read_parquet(datapath, engine='auto', columns=None)
    data = data.sample(frac=1.0, axis=0, random_state=SEED).reset_index(drop=True)
    return data
    
lg.logger.info(f'Loading data ... {data_path}')
t0 = time()
# df = pd.read_parquet(data_path, engine='auto', columns=None)
# df = df.sample(frac=1.0, axis=0, random_state=SEED).reset_index(drop=True)
df_tr = load_data(data_path/f'df_{tr_phase}_tr.parquet')
df_te = load_data(data_path/f'df_{tr_phase}_te.parquet')
lg.logger.info('Done ({:.2f} mins)\n'.format( (time()-t0)/60) )


# -----
# Comet
# -----
# comet_api_key = os.environ.get('COMET_API_KEY')
# comet_prg_name = PRJ_NAME
# comet_set_name = TR_PHASE

# experiment = Experiment(api_key=comet_api_key, project_name=comet_prg_name)
# experiment.set_name(comet_set_name)
# experiment.add_tag(TR_PHASE)


# --------------------
# Split data and scale
# --------------------
# df_tr, df_te = train_test_split(df);  del df
lg.logger.info('df_tr.shape: {}'.format(df_tr.shape))
lg.logger.info('df_te.shape: {}'.format(df_te.shape))

# if tr_phase == 'ref':
#     lg.logger.info('\nDump ref dfs ...')
#     df_tr.to_parquet(outdir/'df_tr.parquet', engine='auto', compression='snappy')
#     df_te.to_parquet(outdir/'df_te.parquet', engine='auto', compression='snappy')

ytr, xtr = df_tr.iloc[:, 0], df_tr.iloc[:, 1:];  del df_tr
yte, xte = df_te.iloc[:, 0], df_te.iloc[:, 1:];  del df_te

# Scale
scaler = StandardScaler()
xtr = pd.DataFrame( scaler.fit_transform(xtr) ).astype(np.float32)
xte = pd.DataFrame( scaler.transform(xte) ).astype(np.float32)
joblib.dump(scaler, outdir/'scaler.pkl')


# -----------------
# Define callbackls
# -----------------
# Callbacks (custom)
tr_iters = xtr.shape[0] / BATCH
# step_size = int(3 * iterations) # num of training iterations per half cycle. Smith suggests to set step_size = (2-8) x (training iterations in epoch).
clr = CyclicLR(base_lr=base_clr, max_lr=max_clr, mode='triangular')

# Keras callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.75, patience=15, verbose=1, mode='auto',
                              min_delta=0.0001, cooldown=3, min_lr=1e-9)
early_stop = EarlyStopping(monitor='val_loss', patience=50, verbose=1, mode='auto')
csv_logger = CSVLogger(outdir/f'krs_logger.log')

model_checkpoint_dir = outdir/'models'
os.makedirs(model_checkpoint_dir, exist_ok=True)
# checkpointer = ModelCheckpoint(str(outdir/'model.ep_{epoch:d}-val_loss_{val_loss:.5f}.h5'),
#                                verbose=0, save_weights_only=False, save_best_only=False)
checkpointer = ModelCheckpoint(str(model_checkpoint_dir/'model.ep_{epoch:d}-val_loss_{val_loss:.5f}.h5'),
                               verbose=0, save_weights_only=False, save_best_only=False)

# Callbacks list
callback_list = [checkpointer, csv_logger, early_stop, reduce_lr, clr]


# -----
# Train
# -----
# fit_params
fit_params = {'batch_size': BATCH, 'epochs': EPOCH, 'verbose': 1}
fit_params['validation_data'] = (xte, yte)
fit_params['callbacks'] = callback_list

# Get the estimator
init_params = {'input_dim': xtr.shape[1], 'dr_rate': DR, 'attn': attn}
model = ml_models.get_model(model_name=model_name, init_params=init_params)

# Train model
t0 = time()
history = model.model.fit(xtr, ytr, **fit_params)
fit_runtime = time() - t0
lg.logger.info('Train runtime: {:.1f} mins'.format(fit_runtime/60))

# Print score
score = model.model.evaluate(xte, yte, verbose=0)
lg.logger.info('val_loss: {:.5f}'.format(score[0]))


# -----------------
# Summarize results
# -----------------
# Plots
plts_path = outdir/'plts'
os.makedirs(plts_path, exist_ok=True)
ml_models.plot_prfrm_metrics(history=history, title=f'{tr_phase} training',
                             skp_ep=skp_ep, add_lr=True, outdir=plts_path)

# Save keras history
ml_models.save_krs_history(history, outdir)


lg.logger.info('\nProgram runtime: {:.2f} mins'.format( (time() - t_start)/60 ))
lg.logger.info('Done.')


