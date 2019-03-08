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


# epoch, loss, mean_absolute_error, r2, val_loss, val_mean_absolute_error, val_r2
# 200, 0.012034249995582483, 0.08081748884650404,  0.5621227001456715, 0.011678791823905396, 0.07943350637974293, 0.5800033041099545
# EARLY_STOP_VAL_LOSS_THRES = 0.011678791823905396  # val_loss after ~200 epochs
EARLY_STOP_VAL_LOSS_THRES = 0.009638553218112529  # val_loss after ~150 epochs

# ep_vec = [int(x) for x in np.linspace(25, 175, 7)]
# ep_vec = [190, 160, 120, 80, 40]
ep_vec = [300, 200, 150, 80, 40]


# File path
# file_path = os.path.dirname(os.path.realpath(__file__))
file_path = pathlib.Path(__file__).resolve().parent


# Path - create dir to dump results (AP)
PRJ_NAME = 'candle_challenge_prb'
PHASE = 'cnt'
OUTDIR = file_path / '../../models' / PRJ_NAME / PHASE
os.makedirs(OUTDIR, exist_ok=True)

DATADIR = file_path / '../../models' / PRJ_NAME / 'data'
WRMDIR = file_path / '../../models' / PRJ_NAME / 'wrm'  # 'wrm_lr_0.0001_new'
BASELINEDIR = file_path / '../../models' / PRJ_NAME / 'baseline'  # 'baseline_lr_0.0001_new'


# Arg parser
psr = argparse.ArgumentParser(description='input agg csv file')
psr.add_argument('--batch', type=int, default=32)
psr.add_argument('--dr', type=float, default=0.2)
psr.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
psr.add_argument('--ep', type=int, default=200, help='Total number epochs')
psr.add_argument('--split_method', type=str, choices=['rnd', 'hrd'], default='hrd')

args = vars(psr.parse_args())
pprint(args)


# Args
EPOCH = args['ep']
BATCH = args['batch']
DR = args['dr']
LR = args['lr']
split_method = args['split_method']
skip_epochs=0


# Data path
if split_method == 'rnd':
    data_path_tr = BASELINEDIR / 'rnd' / 'df_tr.parquet'
    data_path_te = BASELINEDIR / 'rnd' / 'df_te.parquet'    
    outdir = OUTDIR / 'rnd'
    wrmdir = WRMDIR / 'rnd'
    baselinedir = BASELINEDIR / 'rnd'
elif split_method == 'hrd':
    data_path_tr = BASELINEDIR / 'hrd' / 'df_tr.parquet'
    data_path_te = BASELINEDIR / 'hrd' / 'df_te.parquet'        
    outdir = OUTDIR / 'hrd'
    wrmdir = WRMDIR / 'hrd'
    baselinedir = BASELINEDIR / 'hrd'
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
# We'll use the same data that was used to train the baseline model
def load_data_prqt(data_path):
    df = pd.read_parquet(data_path, engine='auto', columns=None)
    df = df.sample(frac=1.0, axis=0, random_state=SEED)
    return df
lg.logger.info(f'Loading data ... {data_path_tr}')
t0 = time()
df_tr = load_data_prqt(data_path=data_path_tr)
df_te = load_data_prqt(data_path=data_path_te)
lg.logger.info('Done ({:.2f} mins).\n'.format((time()-t0)/60))


# ---------------------------
# Extract features and target
# ---------------------------
ytr, xtr = df_tr.iloc[:, 0], df_tr.iloc[:, 1:];
yte, xte = df_te.iloc[:, 0], df_te.iloc[:, 1:];
lg.logger.info('xtr.shape: {}'.format(xtr.shape))
lg.logger.info('xte.shape: {}'.format(xte.shape))
lg.logger.info('ytr.shape: {}'.format(ytr.shape))
lg.logger.info('yte.shape: {}'.format(yte.shape))

# Scale (use the scaler from the warm-up model)
scaler_path = wrmdir/'scaler.pkl'
scaler = joblib.load(scaler_path)
xtr = pd.DataFrame( scaler.transform(xtr) ).astype(np.float32)
xte = pd.DataFrame( scaler.transform(xte) ).astype(np.float32)


# ----------------------
# Train 'continue' model
# ----------------------
tag = 'cnt'


# Custom callback to stop training after reaching a target val_loss
class EarlyStoppingByLossVal(Callback):
    """ Custom callback that terminates training if a specific `monitor` metric reaches
    a specific value indicated by `value`. For example: we want to terminate training reaches
    val_loss of 0.05.
    
    https://stackoverflow.com/questions/37293642/how-to-tell-keras-stop-training-based-on-loss-value
    """
    def __init__(self, monitor='val_loss', value=0.00001, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if current < self.value:
            if self.verbose > 0:
                print(f'Epoch {epoch:4d}: early stopping threshold of {self.value}.')
            self.model.stop_training = True
            self.stopped_epoch = epoch  # (AP)


summary = {}
lg.logger.info('\n___ Start iterating over weps ___')
for i, weps in enumerate(ep_vec):
    t0 = time()
    
    # Load warm model
    lg.logger.info(f'\nLoad warm-up model with {weps} weps')
    modelpath = glob(str(wrmdir/f'*ep_{weps}-*.h5'))[0]
    model = load_model(modelpath, custom_objects={'r2': r2})  # https://github.com/keras-team/keras/issues/5916
    
    # Compute val_loss of the wrm model
    lg.logger.info('Learning rate (wrm): {}'.format( K.eval(model.optimizer.lr)) )
    score_wrm = model.evaluate(xte, yte, verbose=0)
    lg.logger.info('val_loss (wrm): {:.5f}'.format(score_wrm[0]))
    
    # Reset learning rate to a new value
    lr_new = LR
    model.compile(loss='mean_squared_error',
                  optimizer=SGD(lr=lr_new, momentum=0.9),
                  metrics=['mae', r2])
    
    # Create outdir for a specific value of weps
    ep_dir = outdir / ('weps_'+str(weps))
    os.makedirs(ep_dir, exist_ok=True)

    
    # Callbacks (custom)
    clr_triangular = CyclicLR(base_lr=0.0001, max_lr=0.001, mode='triangular')
    early_stop_custom = EarlyStoppingByLossVal(monitor='val_loss', value=EARLY_STOP_VAL_LOSS_THRES, verbose=True)
    
    # Keras callbacks
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.75, patience=5, verbose=1, mode='auto',
                                  min_delta=0.0001, min_lr=0.000000001)  # patience=20, cooldown=3
    early_stop = EarlyStopping(monitor='val_loss', patience=60, verbose=1, mode='auto')
    checkpointer = ModelCheckpoint(str(ep_dir / 'model.cnt.h5'), verbose=0, save_weights_only=False, save_best_only=True)
    csv_logger = CSVLogger(ep_dir / f'model.{tag}.log')
    
    # Callbacks list
    callback_list = [checkpointer, csv_logger, early_stop, reduce_lr,  # keras callbacks
                     clr_triangular, early_stop_custom]  # custom callbacks
    
    # fit_params
    fit_params = {'batch_size': BATCH, 'epochs': EPOCH, 'verbose': 1}
    fit_params['validation_data'] = (xte, yte)
    fit_params['callbacks'] = callback_list

    # Train model - continue phase
    history = model.fit(xtr, ytr, **fit_params)    
    score_cnt = model.evaluate(xte, yte, verbose=0)
    lg.logger.info('val_loss (cnt): {:.5f}'.format(score_cnt[0]))

    # Compute ceps runtime
    runtime_per_weps = time() - t0
    lg.logger.info('Runtime (for {} weps): {:.2f} mins'.format(weps, runtime_per_weps/60))    
    
    # Get ceps
    ceps = len(history.epoch)
    
    # Update summary table
    summary[i] = (weps, ceps, score_wrm[0], score_cnt[0], runtime_per_weps)    
    
    # Print plots
    model_plts_path = ep_dir / 'model_cnt_plts'
    os.makedirs(model_plts_path, exist_ok=True)
    ml_models.plot_prfrm_metrics(history=history,
                                 title=f'Continue training',
                                 skip_epochs=skip_epochs,
                                 add_lr=True,
                                 outdir=model_plts_path)

    # Define path to dump model and weights
    model_path = outdir / f'model.{tag}.json'
    weights_path = outdir / f'weights.{tag}.h5'

    # Save model
    model_json = model.to_json()
    with open(model_path, 'w') as json_file:
        json_file.write(model_json)

    # Save weights
    model.save_weights(weights_path)
    
    
lg.logger.info('\n_________________________________')
summary = pd.DataFrame.from_dict(summary, orient='index', columns=['weps', 'ceps', 'val_loss_wrm', 'val_loss_cnt', 'runtime_sec'])
summary.to_csv(outdir/'summary.csv', index=False)

# Final plot
fig, ax1 = plt.subplots()
ax1.plot(summary['weps'], summary['ceps'], '-or')
ax1.set_xlabel('weps')
ax1.set_ylabel('ceps', color='r')
ax1.tick_params('y', colors='r')
ax1.grid(True)

ax2 = ax1.twinx()
ax2.plot(summary['weps'], summary['runtime_sec'], '-ob')
ax2.set_ylabel('runtime', color='b')
ax2.tick_params('y', colors='b')

fig.tight_layout()
plt.savefig(outdir/'summary_plot.png', bbox_inches='tight')

lg.logger.info('\nProgram runtime: {:.2f} mins'.format( (time() - t_start)/60 ))
lg.logger.info('Done.')


