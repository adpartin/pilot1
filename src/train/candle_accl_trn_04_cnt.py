"""
TODO: Note that at this point only val_loss this available as the ref_metric. However,
straightforward changes in the code should suppot other metrics.
"""
from __future__ import print_function, division

import warnings
warnings.filterwarnings('ignore')

# from comet_ml import Experiment
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

from sklearn.externals import joblib

from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit, KFold
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

SEED = None
t_start = time()


# Utils
import ml_models
from ml_models import r2_krs
import classlogger
import utils


# Import custom callbacks
keras_contrib = '/vol/ml/apartin/projects/keras-contrib/keras_contrib'
sys.path.append(keras_contrib)
from callbacks import *


# epoch, loss, mean_absolute_error, r2, val_loss, val_mean_absolute_error, val_r2
# 200, 0.012034249995582483, 0.08081748884650404,  0.5621227001456715, 0.011678791823905396, 0.07943350637974293, 0.5800033041099545
# EARLY_STOP_VAL_LOSS_THRES = 0.011678791823905396  # val_loss after ~200 epochs
# VAL_LOSS_REF = 0.009638553218112529  # val_loss after ~150 epochs

# ep_vec = [int(x) for x in np.linspace(25, 175, 7)]
# ep_vec = [190, 160, 120, 80, 40]
# ep_vec = [280, 200, 150, 120, 90, 70, 40]
ep_vec = [280, 240, 200, 150, 120, 90, 70, 40]


# File path
# file_path = os.path.dirname(os.path.realpath(__file__))
file_path = Path(__file__).resolve().parent


# Path - create dir to dump results (AP)
PRJ_NAME = 'candle_accl_trn'
PRJ_DIR = file_path / '../../models' / PRJ_NAME
DATADIR = PRJ_DIR / 'data'


# Arg parser
psr = argparse.ArgumentParser(description='input agg csv file')
psr.add_argument('--batch', type=int, default=32)
psr.add_argument('--dr', type=float, default=0.2)
psr.add_argument('--ep', type=int, default=300, help='Total number epochs')
psr.add_argument('--ref_ep', type=int, default=250, help='Reference epoch')
psr.add_argument('--ref_met', type=str, default='val_loss', help='Reference metric')
psr.add_argument('--split_by', type=str, choices=['cell', 'drug', 'both', 'none'], default='cell',
                 help='Specify what datasets to load in terms of disjoint partition: `cell`, `drug`, `both`, `none` (random split).')
psr.add_argument('--skip_ep', type=int, default=0, help='Number of epochs to skip when plotting training curves.')
psr.add_argument('--base_clr', type=float, default=1e-4, help='Base learning rate for cyclical learning rate.')
psr.add_argument('--max_clr', type=float, default=1e-3, help='Max learning rate for cyclical learning rate.')

args = vars(psr.parse_args())
pprint(args)


# Args
EPOCH = args['ep']
BATCH = args['batch']
DR = args['dr']
split_by = args['split_by']
ref_ep = args['ref_ep']
ref_metric = args['ref_met']
skip_epochs = args['skip_ep']
base_clr = args['base_clr']
max_clr = args['max_clr']
tr_phase = 'cnt'


# Path and outdir
wrmdir = PRJ_DIR / 'wrm' / ('split_by_' + split_by)
refdir = PRJ_DIR / 'ref' / ('split_by_' + split_by)
data_path_tr = refdir / 'df_tr.parquet'
data_path_te = refdir / 'df_te.parquet' 
outdir = PRJ_DIR / ( tr_phase + '_' + ref_metric + '_at_' + str(ref_ep) ) / ('split_by_' + split_by)
os.makedirs(outdir, exist_ok=True)


# Data path
# if split_method == 'rnd':  
#     outdir = OUTDIR / 'rnd'
#     wrmdir = WRMDIR / 'rnd'
#     refdir = REFDIR / 'rnd'
# elif split_method == 'hrd':
#     outdir = OUTDIR / 'hrd'
#     wrmdir = WRMDIR / 'hrd'
#     refdir = REFDIR / 'hrd'
# data_path_tr = refdir / 'df_tr.parquet'
# data_path_te = refdir / 'df_te.parquet'    
# os.makedirs(outdir, exist_ok=True)   


# Dump args
utils.dump_args(args, outdir=outdir)


# Plot WRM vd REF curves
h_wrm = pd.read_csv(wrmdir/'keras_history.csv')
h_ref = pd.read_csv(refdir/'keras_history.csv')
val_cols_names = [c for c in h_ref.columns if 'val_' in c]
for c in val_cols_names:
    fig, ax = plt.subplots()
    ax.plot(h_ref[c], label=c+'_ref')
    ax.plot(h_wrm['epoch'], h_wrm[c], label=c+'_wrm')
    ax.set_xlabel('epoch')
    ax.set_ylabel(c)
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(outdir/f'ref_vs_wrm_{c}.png', bbox_inches='tight')


# Create logger
logfilename = outdir/'logfile.log'
lg = classlogger.Logger(logfilename=logfilename) 


# -----------------
# Get the ref score
# -----------------
aa = pd.read_csv(refdir/'model.ref.log')
ref_value = aa.loc[ref_ep-1, ref_metric]
lg.logger.info(f'\n{ref_metric} at ref epoch {ref_ep}: {ref_value}\n')

    
# ---------
# Load data
# ---------
# We'll use the same data that was used to train the reference model
def load_data_prqt(data_path):
    df = pd.read_parquet(data_path, engine='auto', columns=None)
    df = df.sample(frac=1.0, axis=0, random_state=SEED)
    return df

df_tr = load_data_prqt(data_path=data_path_tr)
df_te = load_data_prqt(data_path=data_path_te)
lg.logger.info('df_tr.shape: {}'.format(df_tr.shape))
lg.logger.info('df_te.shape: {}'.format(df_te.shape))

# Extract target and features
ytr, xtr = df_tr.iloc[:, 0], df_tr.iloc[:, 1:];
yte, xte = df_te.iloc[:, 0], df_te.iloc[:, 1:];

# Scale (use the scaler from the warm-up model)
scaler_path = wrmdir/'scaler.pkl'
scaler = joblib.load(scaler_path)
xtr = pd.DataFrame( scaler.transform(xtr) ).astype(np.float32)
xte = pd.DataFrame( scaler.transform(xte) ).astype(np.float32)


# ----------------------
# Train 'continue' model
# ----------------------
# Custom callback to stop training after reaching a target val_loss
class EarlyStoppingByMetric(Callback):
    """ Custom callback that terminates training if a specific `monitor` metric reaches
    a specific value indicated by `value`. For example: we want to terminate training reaches
    val_loss of 0.05.
    
    https://stackoverflow.com/questions/37293642/how-to-tell-keras-stop-training-based-on-loss-value
    """
    def __init__(self, monitor='val_loss', value=0.00001, stop_when_below=True, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose
        self.stop_when_below = stop_when_below

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)
        
        if self.stop_when_below:
            if current < self.value:
                if self.verbose > 0:
                    print(f'Epoch {epoch:4d}: early stopping threshold of {self.value}.')
                self.model.stop_training = True
                self.stopped_epoch = epoch
        else:
            if current > self.value:
                if self.verbose > 0:
                    print(f'Epoch {epoch:4d}: early stopping threshold of {self.value}.')
                self.model.stop_training = True
                self.stopped_epoch = epoch

summary = {}
lg.logger.info('\n___ Start iterate over weps ___')
for i, weps in enumerate(ep_vec):
    t0 = time()
    
    # Load warm model
    lg.logger.info(f'\nLoad warmed-up model with {weps} weps')
    modelpath = glob(str(wrmdir/f'*ep_{weps}-*.h5'))[0]
    model = load_model(modelpath, custom_objects={'r2_krs': r2_krs})  # https://github.com/keras-team/keras/issues/5916
    
    # Compute val_loss of the wrm model
    # lg.logger.info('Learning rate (wrm): {}'.format( K.eval(model.optimizer.lr)) )
    score_wrm = model.evaluate(xte, yte, verbose=0)
    lg.logger.info('val_loss (wrm): {:.5f}'.format(score_wrm[0]))
    
    # Reset learning rate to a new value
    lr_new = 0.0005  # LR
    model.compile(loss='mean_squared_error',
                  optimizer=SGD(lr=lr_new, momentum=0.9),
                  metrics=['mae', r2_krs])
    
    # Create outdir for a specific value of weps
    ep_dir = outdir/('weps_'+str(weps))
    os.makedirs(ep_dir, exist_ok=True)

    
    # Callbacks (custom)
    tr_iters = xtr.shape[0]/BATCH
    clr = CyclicLR(base_lr=base_clr, max_lr=max_clr, mode='triangular')
    early_stop_custom = EarlyStoppingByMetric(monitor=ref_metric, value=ref_value,
                                              stop_when_below=True, verbose=True)
    
    # Keras callbacks
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.75, patience=5, verbose=1, mode='auto',
                                  min_delta=0.0001, min_lr=1e-9)  # patience=20, cooldown=3
    early_stop = EarlyStopping(monitor='val_loss', patience=60, verbose=1, mode='auto')
    checkpointer = ModelCheckpoint(str(ep_dir / 'model.cnt.h5'), verbose=0, save_weights_only=False, save_best_only=True)
    csv_logger = CSVLogger(ep_dir / f'model.{tr_phase}.log')
    
    # Callbacks list
    callback_list = [checkpointer, csv_logger, early_stop, reduce_lr,  # keras callbacks
                     clr, early_stop_custom]  # custom callbacks
    
    # -----
    # Train
    # -----    
    # fit_params
    fit_params = {'batch_size': BATCH, 'epochs': EPOCH, 'verbose': 1}
    fit_params['validation_data'] = (xte, yte)
    fit_params['callbacks'] = callback_list

    # Train model - continue phase
    # lg.logger.info('Continue training ...')
    history = model.fit(xtr, ytr, **fit_params)    
    score_cnt = model.evaluate(xte, yte, verbose=0)
    # lg.logger.info('val_loss (cnt): {:.5f}'.format(score_cnt[0]))
    scr_cnt = pd.DataFrame(history.history)
    scr_cnt = scr_cnt.loc[len(scr_cnt)-1, ref_metric]
    lg.logger.info('{} (cnt): {:.5f}'.format(ref_metric, scr_cnt))
    
    # Compute ceps runtime
    runtime_per_weps = time() - t0
    lg.logger.info('Runtime: {:.2f} mins'.format(runtime_per_weps/60))    
    
    # ----------------------------------
    # Results for the summary table/plot
    # ----------------------------------
    # Get ceps
    ceps = len(history.epoch)
    
    # Update summary table
    summary[i] = (weps, ceps, score_wrm[0], score_cnt[0], runtime_per_weps)
    #summary[i] = (weps, ceps, scr_wrm, scr_cnt, runtime_per_weps)
    
    # ---------------------------
    # Save history and make plots
    # ---------------------------
    # Print plots
    model_plts_path = ep_dir/'model_cnt_plts'
    os.makedirs(model_plts_path, exist_ok=True)
    ml_models.plot_prfrm_metrics(history=history, title=f'Continue training',
                                 skip_epochs=skip_epochs, add_lr=True,
                                 outdir=model_plts_path)  

    # Dump keras history
    h = ml_models.dump_keras_history(history, ep_dir)      
    
    # Plot reference training with continue training
    skp_ep = 10
    for c in val_cols_names:
        fig, ax = plt.subplots()
                        
        ax.plot(h_ref['epoch'][skp_ep:], h_ref[c][skp_ep:], 'b-', alpha=0.8, label='ref')
        ax.plot(h['epoch'][skp_ep:]+weps, h[c][skp_ep:], '-', 'r-', alpha=0.8, label=f'weps_{weps}')
        
        # Add line that dindicates the min value of ref curve
        x = np.array(range(skp_ep, h_ref['epoch'].values[-1] + 1))
        y = np.ones((len(x))) * min(h_ref[c])        
        ax.plot(x, y, 'g--', alpha=0.6)
        
        ax.set_xlabel('epoch')
        ax.set_ylabel(c)
        plt.legend(loc='best')
        plt.grid(True)
        plt.savefig(ep_dir/f'ref_vs_cnt_({c} weps).png', bbox_inches='tight')    
    
    # ----------
    # Save model
    # ----------
    # Define path to dump model and weights
    model_path = outdir/f'model.{tr_phase}.json'
    weights_path = outdir/f'weights.{tr_phase}.h5'

    # Save model
    model_json = model.to_json()
    with open(model_path, 'w') as json_file:
        json_file.write(model_json)

    # Save weights
    model.save_weights(weights_path)    
    
    
lg.logger.info('_________________________________')
summary = pd.DataFrame.from_dict(summary, orient='index', columns=['weps', 'ceps', 'val_loss_wrm', 'val_loss_cnt', 'runtime_sec'])
summary.to_csv(outdir/'summary.csv', index=False)
lg.logger.info(summary)

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
plt.title('Reference {} at {} epoch'.format(ref_metric, ref_ep))
plt.savefig(outdir/'summary_plot.png', bbox_inches='tight')

lg.logger.info('\nMaximum speed-up of {}/{}={}'.format( ref_ep, summary['ceps'].min(), ref_ep/summary['ceps'].min()) )
lg.logger.info('Epochs reduced: {}-{}={}'.format( ref_ep, summary['ceps'].min(), ref_ep-summary['ceps'].min()) )

lg.logger.info('\nProgram runtime: {:.2f} mins'.format( (time() - t_start)/60 ))
lg.logger.info('Done.')


