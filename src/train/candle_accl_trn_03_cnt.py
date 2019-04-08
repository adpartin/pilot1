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


# ep_vec = [int(x) for x in np.linspace(25, 175, 7)]
ep_vec = [280, 240, 200, 160, 120, 80, 40]


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
psr.add_argument('--ep', type=int, default=500, help='Total number of epochs.')
psr.add_argument('--ref_ep', type=int, default=300, help='Reference epoch.')
psr.add_argument('--ref_met', type=str, choices=['val_loss', 'val_mean_absolute_error'],
                 default='val_mean_absolute_error', help='Reference metric.')
psr.add_argument('--split_by', type=str, choices=['cell', 'drug', 'both', 'none'],
                 default='cell',
                 help='Specify how to disjointly partition the dataset: \
                 `cell` (disjoint on cell), `drug` (disjoint on drug), \
                 `both` (disjoint on cell and drug), `none` (random split).')
psr.add_argument('--skp_ep', type=int, default=10, help='Number of epochs to skip when plotting training curves.')
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
skp_ep = args['skp_ep']
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


# Dump args
utils.dump_args(args, outdir=outdir)


# Plot WRM vd REF curves
h_wrm = pd.read_csv(wrmdir/'keras_history.csv')
h_ref = pd.read_csv(refdir/'keras_history.csv')
val_cols_names = [c for c in h_ref.columns if 'val_' in c]
for c in val_cols_names:
    fig, ax = plt.subplots()
    ax.plot(h_ref['epoch'][skp_ep:], h_ref[c][skp_ep:], label=c+'_ref')
    ax.plot(h_wrm['epoch'][skp_ep:], h_wrm[c][skp_ep:], label=c+'_wrm')
    ax.set_xlabel('epoch')
    ax.set_ylabel(c)
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(outdir/f'ref_vs_wrm_{c}.png', bbox_inches='tight')


# Create logger
logfilename = outdir/'logfile.log'
lg = classlogger.Logger(logfilename=logfilename) 

    
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


# -----------------
# Get the ref score
# -----------------
# aa = pd.read_csv(refdir/'model.ref.log')
# score_ref = aa.loc[ref_ep-1, ref_metric]
# score_ref = aa[ref_metric].min()
# lg.logger.info(f'\n{ref_metric} at ref epoch {ref_ep}: {score_ref}')
x = h_ref[ref_metric].min()
lg.logger.info(f'\n{ref_metric} (min): {x.min()}')
prct_diff = 2
score_ref = x + x * prct_diff/100
lg.logger.info(f'{ref_metric} ({prct_diff}% from min): {score_ref}')


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
            if current <= self.value:
                if self.verbose > 0:
                    print(f'Epoch {epoch:4d}: Early stopping, {self.monitor} threshold of {self.value}.')
                self.model.stop_training = True
                self.stopped_epoch = epoch
        else:
            if current >= self.value:
                if self.verbose > 0:
                    print(f'Epoch {epoch:4d}: Early stopping, {self.monitor} threshold of {self.value}.')
                self.model.stop_training = True
                self.stopped_epoch = epoch

                
summary = {}
lg.logger.info('\n___ Iterate over weps ___')
for i, weps in enumerate(ep_vec):
    
    # Load warm model
    lg.logger.info(f'\nLoad warmed-up model with {weps} weps')
    modelpath = glob(str(wrmdir/f'*ep_{weps}-*.h5'))[0]
    model = load_model(modelpath, custom_objects={'r2_krs': r2_krs})  # https://github.com/keras-team/keras/issues/5916
    
    # Compute ref_metric of wrm model
    # lg.logger.info('Learning rate (wrm): {}'.format( K.eval(model.optimizer.lr)) )
    score_wrm = model.evaluate(xte, yte, verbose=0)
    score_wrm = score_wrm[ int(np.argwhere(h_ref.columns == ref_metric)) ]
    lg.logger.info('{} (wrm): {}'.format(ref_metric, score_wrm))
    
    # Reset learning rate to a new value
    lr_new = 0.0005
    model.compile(loss='mean_squared_error',
                  optimizer=SGD(lr=lr_new, momentum=0.9),
                  metrics=['mae', r2_krs])
    
    # Create outdir for a specific value of weps
    ep_dir = outdir/('weps_'+str(weps))
    os.makedirs(ep_dir, exist_ok=True)

    
    # Callbacks (custom)
    tr_iters = xtr.shape[0]/BATCH
    clr = CyclicLR(base_lr=base_clr, max_lr=max_clr, mode='triangular')
    early_stop_custom = EarlyStoppingByMetric(monitor=ref_metric, value=score_ref,
                                              stop_when_below=True, verbose=True)
    
    # Keras callbacks
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.75, patience=15, verbose=1, mode='auto',
                                  min_delta=0.0001, min_lr=1e-9)
#     reduce_lr = ReduceLROnPlateau(monitor=ref_metric, factor=0.75, patience=10, verbose=1, mode='auto',
#                                   min_delta=0.0001, min_lr=1e-9)
    early_stop = EarlyStopping(monitor=ref_metric, patience=50, verbose=1, mode='auto')
    checkpointer = ModelCheckpoint(str(ep_dir/'model.cnt.h5'), verbose=0, save_weights_only=False, save_best_only=True)
    csv_logger = CSVLogger(ep_dir/f'model.{tr_phase}.log')
    
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
    
    # Train model (continue phase)
    t0 = time()
    history = model.fit(xtr, ytr, **fit_params)
    runtime_ceps = time() - t0  
    
    # Dump keras history
    kh = ml_models.dump_keras_history(history, ep_dir) 


    # ----------------------------------
    # Results for the summary table/plot
    # ----------------------------------
    # Get ceps
    ceps = len(history.epoch)
    
    # Compute ref_metric of cnt model
    score_cnt = kh.loc[len(kh)-1, ref_metric]
    lg.logger.info('{} (cnt): {}'.format( ref_metric, score_cnt ))
    #score_cnt = model.evaluate(xte, yte, verbose=0)
    #score_cnt = score_cnt[ int(np.argwhere(h_ref.columns == ref_metric)) ]    
    
    # Bool that indicates if wrm model was converged to score_ref
    # TODO: this is correct for error (not R^2)
    converge = True if score_cnt <= score_ref else False
        
    # Update summary table
    summary[i] = (weps, ceps, score_wrm, score_cnt, runtime_ceps, converge)
    
    lg.logger.info('converge: {}'.format(converge))
    lg.logger.info('ceps: {} ({:.2f} mins)'.format( ceps, runtime_ceps/60 ))
    
    
    # ----------
    # Make plots
    # ----------
    # Plots
    model_plts_path = ep_dir/'model_cnt_plts'
    os.makedirs(model_plts_path, exist_ok=True)
    ml_models.plot_prfrm_metrics(history=history, title=f'Continue training',
                                 skp_ep=skp_ep, add_lr=True, outdir=model_plts_path)  
    
    # Plot reference training with continue training
    for c in val_cols_names:
        fig, ax = plt.subplots(figsize=(8, 6))

        x1 = list(h_ref['epoch'])
        x2 = list(kh['epoch'] + weps)
        
        ax.plot(x1, h_ref[c], 'b-', alpha=0.7, label='ref')
        ax.plot(x2, kh[c], 'ro-', markersize=2, alpha=0.7, label=f'weps_{weps}')        
        
        if c == ref_metric:
            x = range(0, max(x1+x2))
            ymin = np.ones(len(x)) * min(h_ref[c])
            yref = np.ones(len(x)) * score_ref
            ax.plot(x, ymin, 'c--', linewidth=1, alpha=0.7, label='ref_min')
            ax.plot(x, yref, 'g--', linewidth=1, alpha=0.7, label=f'ref_min_{prct_diff}%')
        
        ax.set_xlabel('epoch')
        ax.set_ylabel(c)
        plt.title(f'ceps: {ceps}')
        plt.legend(loc='best')
        plt.grid(True)
        plt.savefig(ep_dir/f'ref_vs_cnt_({c}).png', bbox_inches='tight')
        del fig, ax


lg.logger.info('\n' + '_'*70)
columns = ['weps', 'ceps', f'{ref_metric}_wrm', f'{ref_metric}_cnt', 'runtime_sec', 'converge']
summary = pd.DataFrame.from_dict(summary, orient='index', columns=columns)
summary.to_csv(outdir/'summary.csv', index=False)
lg.logger.info(summary)


# Final plot
fig, ax1 = plt.subplots()

# Add ceps plot
ax1.plot(summary['weps'], summary['ceps'], '-ob')
ax1.set_xlabel('weps')
ax1.set_ylabel('ceps', color='b')
ax1.tick_params('y', colors='b')
ax1.grid(True)

# Add runtime plot
# ax2 = ax1.twinx()
# ax2.plot(summary['weps'], summary['runtime_sec'], '-om')
# ax2.set_ylabel('runtime', color='m')
# ax2.tick_params('y', colors='m')

fig.tight_layout()
# plt.title('Reference {} at {} epoch'.format(ref_metric, ref_ep))
plt.title(f'Ref epochs: {ref_ep};  Diff from min score: {prct_diff}%')
plt.savefig(outdir/'summary_plot.png', bbox_inches='tight')


lg.logger.info('\nMax speed-up: {}/{}={}'.format( ref_ep, summary['ceps'].min(), ref_ep/summary['ceps'].min()) )
lg.logger.info('Max epochs reduced: {}-{}={}'.format( ref_ep, summary['ceps'].min(), ref_ep-summary['ceps'].min()) )

lg.logger.info('\nProgram runtime: {:.2f} mins'.format( (time() - t_start)/60 ))
lg.logger.info('Done.')


