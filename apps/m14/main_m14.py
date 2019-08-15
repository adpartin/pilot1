""" 
Milestone 14.
"""
from __future__ import print_function, division

import warnings
warnings.filterwarnings('ignore')

import os
import sys
from pathlib import Path
import argparse
from datetime import datetime
from time import time
from pprint import pprint, pformat
from collections import OrderedDict
from glob import glob

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sklearn
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

import keras
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard
from keras.utils import plot_model

SEED = 42


# File path
file_path = Path(__file__).resolve().parent


# Utils
utils_path = file_path / '../../utils'
sys.path.append(str(utils_path))
import utils
import ml_models
from classlogger import Logger


# Path
PRJ_NAME = file_path.name 
OUTDIR = file_path / '../../out/' / PRJ_NAME


def parse_args(args):
    parser = argparse.ArgumentParser(description="Generate learning curves.")

    # Input data
    parser.add_argument('--dirpath', default=None, type=str, help='Full path to data and splits (default: None).')

    # Select target to predict
    parser.add_argument('-t', '--target_name', default='AUC', type=str, choices=['AUC', 'AUC1', 'IC50'], help='Name of target variable (default: AUC).')

    # Select feature types
    parser.add_argument('-cf', '--cell_fea', nargs='+', default=['GE'], choices=['GE'], help='Cell line features (default: rna).')
    parser.add_argument('-df', '--drug_fea', nargs='+', default=['DD'], choices=['DD'], help='Drug features (default: dsc).')

    # Data split methods
    parser.add_argument('-cvm', '--cv_method', default='simple', type=str, choices=['simple', 'group'], help='CV split method (default: simple).')
    parser.add_argument('-cvf', '--cv_folds', default=5, type=str, help='Number cross-val folds (default: 5).')
    
    # ML models
    # parser.add_argument('-frm', '--framework', default='lightgbm', type=str, choices=['keras', 'lightgbm', 'sklearn'], help='ML framework (default: lightgbm).')
    parser.add_argument('-ml', '--model_name', default='lgb_reg', type=str, help='ML model for training (default: lgb_reg).')

    # NN hyper_params
    parser.add_argument('-ep', '--epochs', default=200, type=int, help='Number of epochs (default: 200).')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size (default: 32).')
    parser.add_argument('--dr_rate', default=0.2, type=float, help='Dropout rate (default: 0.2).')
    parser.add_argument('-sc', '--scaler', default='stnd', type=str, choices=['stnd', 'minmax', 'rbst'], help='Feature normalization method (default: stnd).')

    parser.add_argument('--opt', default='sgd', type=str, choices=['sgd', 'adam'], help='Optimizer name (default: sgd).')

    parser.add_argument('--clr_mode', default=None, type=str, choices=['trng1', 'trng2', 'exp'], help='CLR mode (default: trng1).')
    parser.add_argument('--clr_base_lr', type=float, default=1e-4, help='Base lr for cycle lr.')
    parser.add_argument('--clr_max_lr', type=float, default=1e-3, help='Max lr for cycle lr.')
    parser.add_argument('--clr_gamma', type=float, default=0.999994, help='Gamma parameter for learning cycle LR.')

    # Define n_jobs
    parser.add_argument('--skp_ep', default=10, type=int, help='Default: 10.')
    parser.add_argument('--n_jobs', default=4, type=int, help='Default: 4.')

    # Parse args
    args = parser.parse_args(args)
    return args
        
    
def create_outdir(outdir, args, src):
    t = datetime.now()
    t = [t.year, '-', t.month, '-', t.day, '_', 'h', t.hour, '-', 'm', t.minute]
    t = ''.join([str(i) for i in t])
    
    l = [('cvf'+str(args['cv_folds']))] + args['cell_fea'] + args['drug_fea'] + [args['target_name']] 
    if args['clr_mode'] is not None: l = [args['clr_mode']] + l
    if 'nn' in args['model_name']: l = [args['opt']] + l
                
    name_sffx = '.'.join( [src] + [args['model_name']] + l )
    # outdir = Path(outdir) / (name_sffx + '_' + t)
    outdir = Path(outdir) / name_sffx
    # os.makedirs(outdir)
    os.makedirs(outdir, exist_ok=True)
    return outdir


def define_keras_callbacks(outdir):
    checkpointer = ModelCheckpoint(str(outdir/'model_best.h5'), verbose=0, save_weights_only=False, save_best_only=True)
    csv_logger = CSVLogger(outdir/'training.log')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.75, patience=20, verbose=1, mode='auto',
                                  min_delta=0.0001, cooldown=3, min_lr=0.000000001)
    # early_stop = EarlyStopping(monitor='val_loss', patience=100, verbose=1)
    early_stop = EarlyStopping(monitor='val_loss', patience=60, verbose=1)
    return [checkpointer, csv_logger, early_stop, reduce_lr]


def calc_preds(model, x, y, mltype):
    """ Calc predictions. """
    if mltype == 'cls':    
        if y.ndim > 1 and y.shape[1] > 1:
            y_pred = model.predict_proba(x)
            y_pred = np.argmax(y_pred, axis=1)
            y_true = np.argmax(ydata, axis=1)
        else:
            y_pred = model.predict_proba(x)
            y_pred = np.argmax(y_pred, axis=1)
            y_true = y
            
    elif mltype == 'reg':
        y_pred = model.predict(x)
        y_true = y

    return y_pred, y_true


def calc_scores(y_true, y_pred, mltype, metrics=None):
    """ Create dict of scores. """
    scores = {}

    if mltype == 'cls':    
        scores['auroc'] = sklearn.metrics.roc_auc_score(y_true, y_pred)
        scores['f1_score'] = sklearn.metrics.f1_score(y_true, y_pred, average='micro')
        scores['acc_blnc'] = sklearn.metrics.balanced_accuracy_score(y_true, y_pred)

    elif mltype == 'reg':
        scores['r2'] = sklearn.metrics.r2_score(y_true=y_true, y_pred=y_pred)
        scores['mean_absolute_error'] = sklearn.metrics.mean_absolute_error(y_true=y_true, y_pred=y_pred)
        scores['median_absolute_error'] = sklearn.metrics.median_absolute_error(y_true=y_true, y_pred=y_pred)
        scores['mean_squared_error'] = sklearn.metrics.mean_squared_error(y_true=y_true, y_pred=y_pred)

    return scores


def run(args):
    dirpath = Path(args['dirpath'])
   
    # Target
    target_name = args['target_name']

    # Data split 
    cv_folds = args['cv_folds']

    # Features 
    cell_fea = args['cell_fea']
    drug_fea = args['drug_fea']
    fea_list = cell_fea + drug_fea # + other_fea

    # NN params
    epochs = args['epochs']
    batch_size = args['batch_size']
    dr_rate = args['dr_rate']

    # Optimizer
    opt_name = args['opt']
    clr_keras_kwargs = {'mode': args['clr_mode'], 'base_lr': args['clr_base_lr'],
                        'max_lr': args['clr_max_lr'], 'gamma': args['clr_gamma']}

    # Other params
    # framework = args['framework']
    model_name = args['model_name']
    skp_ep = args['skp_ep']
    n_jobs = args['n_jobs']

    # ML type ('reg' or 'cls')
    if 'reg' in model_name:
        mltype = 'reg'
    elif 'cls' in model_name:
        mltype = 'cls'
    else:
        raise ValueError("model_name must contain 'reg' or 'cls'.")

    src = dirpath.name.split('_')[0]
    
    
    # -----------------------------------------------
    #       Load data and pre-proc
    # -----------------------------------------------
    def get_file(fpath):
        return pd.read_csv(fpath, header=None).squeeze().values if fpath.is_file() else None

    def read_data_file(fpath, file_format='csv'):
        fpath = Path(fpath)
        if fpath.is_file():
            if file_format=='csv':
                df = pd.read_csv( fpath )
            elif file_format=='parquet':
                df = pd.read_parquet( fpath )
        else:
            df = None
        return df
    

    # Data splits
    tr_id = pd.read_csv( dirpath/f'{cv_folds}fold_tr_id.csv' )
    vl_id = pd.read_csv( dirpath/f'{cv_folds}fold_vl_id.csv' )
    te_id = pd.read_csv( dirpath/'te_id.csv' )

    tr_dct = {}
    vl_dct = {}    

    for fold in range(tr_id.shape[1]):
        tr_dct[fold] = tr_id.iloc[:, fold].dropna().values.astype(int).tolist()
        vl_dct[fold] = vl_id.iloc[:, fold].dropna().values.astype(int).tolist()
        
    te_id = te_id.iloc[:, 0].dropna().values.astype(int).tolist() 
        

    # Load data    
    xdata = read_data_file( dirpath/'xdata.parquet', 'parquet' )
    meta  = read_data_file( dirpath/'meta.parquet', 'parquet' )
    ydata = meta[[target_name]]
        

    # Scale
    scaler = args['scaler']
    if scaler is not None:
        if scaler == 'stnd':
            scaler = StandardScaler()
        elif scaler == 'minmax':
            scaler = MinMaxScaler()
        elif scaler == 'rbst':
            scaler = RobustScaler()
        
    cols = xdata.columns
    xdata = pd.DataFrame(scaler.fit_transform(xdata), columns=cols, dtype=np.float32)

    # Test set
    xte = xdata.iloc[te_id, :]
    yte = np.squeeze(ydata.iloc[te_id, :])   


    # -----------------------------------------------
    #       Create outdir and logger
    # -----------------------------------------------
    run_outdir = create_outdir(OUTDIR, args, src)
    lg = Logger(run_outdir/'logfile.log')
    lg.logger.info(f'File path: {file_path}')
    lg.logger.info(f'\n{pformat(args)}')

    # Dump args to file
    utils.dump_dict(args, outpath=run_outdir/'args.txt')        


    # -----------------------------------------------
    #      ML model configs
    # -----------------------------------------------
    if model_name == 'lgb_reg':
        framework = 'lightgbm'
        init_kwargs = {'n_jobs': n_jobs, 'random_state': SEED, 'logger': lg.logger}
        fit_kwargs = {'verbose': False}
    elif model_name == 'nn_reg':
        framework = 'keras'
        init_kwargs = {'input_dim': xdata.shape[1], 'dr_rate': dr_rate, 'opt_name': opt_name, 'attn': attn, 'logger': lg.logger}
        fit_kwargs = {'batch_size': batch_size, 'epochs': epochs, 'verbose': 1} 
    elif model_name == 'nn_reg0' or 'nn_reg1' or 'nn_reg2':
        framework = 'keras'
        init_kwargs = {'input_dim': xdata.shape[1], 'dr_rate': dr_rate, 'opt_name': opt_name, 'logger': lg.logger}
        fit_kwargs = {'batch_size': batch_size, 'epochs': epochs, 'verbose': 1}  # 'validation_split': 0.1
    elif model_name == 'nn_reg3' or 'nn_reg4':
        framework = 'keras'
        init_kwargs = {'in_dim_rna': None, 'in_dim_dsc': None, 'dr_rate': dr_rate, 'opt_name': opt_name, 'logger': lg.logger}
        fit_kwargs = {'batch_size': batch_size, 'epochs': epochs, 'verbose': 1}  # 'validation_split': 0.1


    # -----------------------------------------------
    #      Train
    # -----------------------------------------------
    lg.logger.info('\n\n{}'.format('=' * 50))
    lg.logger.info(f'Train {src} ...')
    lg.logger.info('=' * 50)

    # CV loop
    for fold, (tr_k, vl_k) in enumerate(zip( tr_dct.keys(), vl_dct.keys() )):
        if lg.logger is not None: lg.logger.info(f'Fold {fold+1}/{cv_folds}')

        tr_id = tr_dct[tr_k]
        vl_id = vl_dct[vl_k]

        # Samples from this dataset are randomly sampled for training
        xtr = xdata.iloc[tr_id, :]
        ytr = ydata.iloc[tr_id, :]

        # A fixed set of validation samples for the current CV split
        xvl = xdata.iloc[vl_id, :]
        yvl = np.squeeze(ydata.iloc[vl_id, :])   
        
        # Get the estimator
        estimator = ml_models.get_model(model_name, init_kwargs=init_kwargs)
        model = estimator.model
        
        keras.utils.plot_model(model, to_file=run_outdir/'nn_model.png')
        keras_callbacks = define_keras_callbacks(run_outdir)
        
        if clr_keras_kwargs['mode'] is not None:
            keras_callbacks.append( ml_models.clr_keras_callback(**clr_keras_kwargs) )        
            
        # Fit params
        fit_kwargs['validation_data'] = (xvl, yvl)
        fit_kwargs['callbacks'] = keras_callbacks            
        
        # Train
        t0 = time()
        history = model.fit(xtr, ytr, **fit_kwargs)
        lg.logger.info('Runtime: {:.1f} hrs'.format( (time()-t0)/360) )
       
        # Multi-gpu training
        # keras.utils.multi_gpu_model(model, gpus=[0, 1], cpu_merge=True, cpu_relocation=False)

        # Log
        ml_models.save_krs_history(history, outdir=run_outdir)
        ml_models.plot_prfrm_metrics(history, title=f'Training', skp_ep=skp_ep, add_lr=True, outdir=run_outdir)

        # Calc preds and scores
        # ... training set
        y_pred, y_true = calc_preds(model, x=xtr, y=ytr, mltype=mltype)
        tr_scores = calc_scores(y_true=y_true, y_pred=y_pred, mltype=mltype, metrics=None)
        pd.DataFrame({'y_true': y_true, 'y_pred': y_pred.reshape(-1)}).to_csv(run_outdir/'tr_preds.csv', index=False)
        # ... val set
        y_pred, y_true = calc_preds(model, x=xvl, y=yvl, mltype=mltype)
        vl_scores = calc_scores(y_true=y_true, y_pred=y_pred, mltype=mltype, metrics=None)        
        pd.DataFrame({'y_true': y_true, 'y_pred': y_pred.reshape(-1)}).to_csv(run_outdir/'vl_preds.csv', index=False)

        
    # Calc preds and scores for test set
    y_pred, y_true = calc_preds(model, x=xte, y=yte, mltype=mltype)
    te_scores = calc_scores(y_true=y_true, y_pred=y_pred, mltype=mltype, metrics=None)        
    pd.DataFrame({'y_true': y_true, 'y_pred': y_pred.reshape(-1)}).to_csv(run_outdir/'te_preds.csv', index=False)

        
    lg.kill_logger()
    del xdata, ydata
        
    print('Done.')


def main(args):
    args = parse_args(args)
    args = vars(args)
    run(args)
    

if __name__ == '__main__':
    main(sys.argv[1:])
