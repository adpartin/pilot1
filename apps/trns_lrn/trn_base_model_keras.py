"""
Train base model. This model will be used to continue training.
"""
from __future__ import print_function, division

import warnings
warnings.filterwarnings('ignore')

import os
import sys
import platform
from pathlib import Path
import psutil
import argparse
from datetime import datetime
from time import time
from pprint import pprint, pformat
from collections import OrderedDict
from glob import glob

import sklearn
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.externals import joblib
from sklearn.metrics import r2_score, mean_absolute_error, median_absolute_error, explained_variance_score

import keras as ke
from keras import backend as K

from keras.models import Sequential, Model, model_from_json, model_from_yaml, load_model
from keras.optimizers import SGD, Adam, RMSprop, Adadelta
from keras.utils import np_utils, multi_gpu_model, plot_model
from keras.callbacks import Callback, ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping

import lightgbm as lgb

SEED = 42
t_start = time()

# File path
file_path = Path(__file__).resolve().parent

# Utils
utils_path = file_path / '../../utils'
sys.path.append(str(utils_path))
import utils
from classlogger import Logger
import ml_models

# Import custom callbacks
keras_contrib = '/vol/ml/apartin/projects/keras-contrib/keras_contrib/callbacks'
sys.path.append(keras_contrib)
from cyclical_learning_rate import CyclicLR

# Path
PRJ_NAME = Path(file_path.name)/'base_model_keras'
OUTDIR = file_path / '../../out/' / PRJ_NAME


def parse_args(args):
    parser = argparse.ArgumentParser(description="Train base NN model.")

    # Input data
    parser.add_argument('--dirpath', default=None, type=str, help='Full path to data and split (default: None).')

    # parser.add_argument('--dname', default='ytn', type=str, choices=['top6', 'ytn'], help='Dataset name (default: ytn).')
    # parser.add_argument('--frm', default='trch', type=str, choices=['krs', 'trch'], help='DL framework (default: trch).')
    # parser.add_argument('--src', default='GDSC', type=str, help='Data source (default: GDSC).')

    parser.add_argument('-ml', '--model_name', type=str, default='nn_reg0')
    parser.add_argument('-ep', '--epochs', default=250, type=int, help='Epochs (default: 250).')
    parser.add_argument('-b', '--batch_size', default=32, type=float, help='Batch size (default: 32).')
    parser.add_argument('--dr_rate', default=0.2, type=float, help='Dropout rate (default: 0.2).')

    parser.add_argument('--opt', default='sgd', type=str, choices=['sgd', 'adam', 'clr_trng1', 'clr_trng2', 'clr_exp'], help='Optimizer name (default: `sgd`).')
    parser.add_argument('--base_lr', type=float, default=1e-4, help='Base lr for cycle lr.')
    parser.add_argument('--max_lr', type=float, default=1e-3, help='Max lr for cycle lr.')
    parser.add_argument('--gamma', type=float, default=0.99994, help='Gamma parameter for learning cycle LR.')
    parser.add_argument('--skp_ep', type=int, default=10, help='Number of epochs to skip when plotting training curves.')

    parser.add_argument('--n_jobs', default=4, type=int, help='Number of cpu workers (default: 4).')
    args = parser.parse_args(args)
    return args


def create_outdir(outdir, args):
    t = datetime.now()
    t = [t.year, '-', t.month, '-', t.day, '_', 'h', t.hour, '-', 'm', t.minute]
    t = ''.join([str(i) for i in t])
    
    l = [args['src']] + [args['model_name']] + [args['opt']] +  ['ep'+str(args['epochs'])] + ['drp'+str(args['dr_rate'])]
        
    name_sffx = '.'.join( l )
    outdir = Path(outdir) / (name_sffx + '_' + t)
    os.makedirs(outdir)
    return outdir


def run(args):
    dirpath = Path(args['dirpath'])
    epochs = args['epochs']
    batch_size = args['batch_size']
    dr_rate = args['dr_rate']
    model_name = args['model_name']
    skp_ep = args['skp_ep']
    opt_name = args['opt']
    base_lr = args['base_lr']
    max_lr = args['max_lr']
    gamma = args['gamma']
    n_jobs = args['n_jobs']

    verbose = True
    args['src'] = dirpath.name.split('.')[0]
    src = args['src'] 
    
    # Split of the dataset used for for taining the BASE MODEL
    fold = 0
    

    # =====================================================
    #       Logger
    # =====================================================
    run_outdir = create_outdir(OUTDIR, args)
    logfilename = run_outdir/'logfile.log'
    lg = Logger(logfilename)
    lg.logger.info(datetime.now())
    lg.logger.info(f'\nFile path: {file_path}')
    lg.logger.info(f'Machine: {platform.node()} ({platform.system()}, {psutil.cpu_count()} CPUs)')
    lg.logger.info(f'\n{pformat(args)}')

    # Dump args to file
    utils.dump_args(args, run_outdir)      


    # =====================================================
    #       Load data
    # =====================================================
    """
    xdata_fpath = Path((glob(str(dirpath/'*xdata.parquet')))[0])
    ydata_fpath = Path((glob(str(dirpath/'*ydata.parquet')))[0])
    meta_fpath = Path((glob(str(dirpath/'*meta.parquet')))[0])
    if xdata_fpath.is_file():
        xdata = pd.read_parquet( xdata_fpath, engine='auto', columns=None )
    if ydata_fpath.is_file():
        ydata = pd.read_parquet( ydata_fpath, engine='auto', columns=None )
    if meta_fpath.is_file():
        meta = pd.read_parquet( meta_fpath, engine='auto', columns=None )
    """
    # dirpath = '/vol/ml/apartin/projects/pilot1/data/yitan/Data/tidy/GDSC.geneGE.DD'
    data_fpath = Path((glob(str(dirpath/'*data.parquet')))[0])
    assert data_fpath.is_file(), '*data.parquet file was not found.'
    data = pd.read_parquet( data_fpath )
    lg.logger.info('\ndata {}'.format(data.shape))

    # Path to data splits
    datadir = Path(file_path/'../../data/yitan/Data')
    ccl_folds_dir = Path(file_path/'../../data/yitan/CCL_10Fold_Partition')
    pdm_folds_dir = Path(file_path/'../../data/yitan/PDM_10Fold_Partition')

    ccl_fea_list = ['geneGE']
    drg_fea_list = ['DD']
    fea_sep = '_'

    ids_path = ccl_folds_dir/f'{src}/cv_{fold}' # 'TestList.txt'
    tr_ids_list = pd.read_csv(ids_path/'TrainList.txt', header=None).squeeze().values
    vl_ids_list = pd.read_csv(ids_path/'ValList.txt', header=None).squeeze().values
    te_ids_list = pd.read_csv(ids_path/'TestList.txt', header=None).squeeze().values

    data_tr = data[ data['cclname'].isin( tr_ids_list ) ]
    data_vl = data[ data['cclname'].isin( vl_ids_list ) ]
    data_te = data[ data['cclname'].isin( te_ids_list ) ]

    lg.logger.info('data_tr {}'.format(data_tr.shape))
    lg.logger.info('data_vl {}'.format(data_vl.shape))
    lg.logger.info('data_te {}'.format(data_te.shape))

    def extract_subset_fea(df, fea_list, fea_sep='_'):
        """ Extract features based feature prefix name. """
        fea = [c for c in df.columns if (c.split(fea_sep)[0]) in fea_list]
        df = df[fea]
        return df

    def extract_data(df, fea_list):
        """ ... """
        X = extract_subset_fea(df, fea_list=fea_list, fea_sep='_')
        Y = df[['auc']]
        meta = df.drop(columns=X.columns)
        meta = meta.drop(columns=['auc'])
        return X, Y, meta

    xtr, ytr, mtr = extract_data(data_tr, fold=fold, fea_list = ccl_fea_list + drg_fea_list)
    xvl, yvl, mvl = extract_data(data_vl, fold=fold, fea_list = ccl_fea_list + drg_fea_list)
    xte, yte, mte = extract_data(data_te, fold=fold, fea_list = ccl_fea_list + drg_fea_list)


    # =====================================================
    #       Scale (CV can start here)
    # =====================================================
    # Scale
    cols = xtr.columns
    scaler = MinMaxScaler()

    cols = xtr.columns
    xtr = pd.DataFrame(scaler.fit_transform(xtr), columns=cols, dtype=np.float32)
    xvl = pd.DataFrame(scaler.transform(xvl), columns=cols, dtype=np.float32)
    xte = pd.DataFrame(scaler.transform(xte), columns=cols, dtype=np.float32)


    # =====================================================
    #       Train NN keras
    # =====================================================
    # Create output dir
    out_nn_model = run_outdir / ('cv'+str(fold))
    os.makedirs(out_nn_model, exist_ok=False)
    
    # Dump scaler
    joblib.dump(scaler, out_nn_model/'scaler.pkl')
                
    # Params
    init_kwargs = {'input_dim': xtr.shape[1], 'dr_rate': dr_rate, 'opt_name': opt_name, 'logger': lg.logger}
    fit_kwargs = {'batch_size': batch_size, 'epochs': epochs, 'verbose': 1}

    # Get NN model
    estimator = ml_models.get_model(model_name, init_kwargs=init_kwargs)
    plot_model(estimator.model, to_file=run_outdir/f'{model_name}.png')

    # CycleLR
    lg.logger.info('Iterations per epoch: {:.1f}'.format( xtr.shape[0]/batch_size ))
    if opt_name == 'clr_trng1':
        clr = CyclicLR(base_lr=base_lr, max_lr=max_lr, mode='triangular')
    elif opt_name == 'clr_trng2':
        clr = CyclicLR(base_lr=base_lr, max_lr=max_lr, mode='triangular2')
    elif opt_name == 'clr_exp':
        clr = CyclicLR(base_lr=base_lr, max_lr=max_lr, mode='exp_range', gamma=gamma) # 0.99994; 0.99999994; 0.999994
                
    # Checkpoint
    model_checkpoint_dir = out_nn_model/'models'
    os.makedirs(model_checkpoint_dir, exist_ok=True)
    checkpointer = ModelCheckpoint(str(model_checkpoint_dir/'model.ep_{epoch:d}-val_loss_{val_loss:.5f}.h5'),
                                   verbose=0, save_weights_only=False, save_best_only=False)

    # Keras callbacks
    csv_logger = CSVLogger(out_nn_model/'training.log')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.75, patience=20, verbose=1, mode='auto',
                                  min_delta=0.0001, cooldown=3, min_lr=0.000000001)
    early_stop = EarlyStopping(monitor='val_loss', patience=100, verbose=1, mode='auto')
                
    # Callbacks list
    callback_list = [checkpointer, csv_logger, early_stop, reduce_lr]
    if 'clr' in opt_name: callback_list = callback_list + [clr]

    # Fit params
    fit_kwargs['validation_data'] = (xvl, yvl)
    # fit_kwargs['validation_split'] = 0.2
    fit_kwargs['callbacks'] = callback_list

    # Train model
    t0 = time()
    history = estimator.model.fit(xtr, ytr, **fit_kwargs)
    fit_runtime = time() - t0
    lg.logger.info('Train runtime: {:.1f} mins'.format(fit_runtime/60))

    # Dump model
    estimator.dump_model(out_nn_model/'final_model.h5')

    # Get the model
    model = estimator.model

    # Predict
    pred_ytr = model.predict(xtr)
    pred_yvl = model.predict(xvl)
    pred_yte = model.predict(xte)

    # Calc scores
    lg.logger.info(f'\nScores {model_name}:')
    scores = {}
    scores['r2_tr'] = r2_score(ytr, pred_ytr)
    scores['r2_vl'] = r2_score(yvl, pred_yvl)
    scores['r2_te'] = r2_score(yte, pred_yte)
    scores['mae_tr'] = mean_absolute_error(ytr, pred_ytr)
    scores['mae_vl'] = mean_absolute_error(yvl, pred_yvl)
    scores['mae_te'] = mean_absolute_error(yte, pred_yte)
    for k, v, in scores.items(): lg.logger.info(f'{k}: {v}')
    utils.dump_dict(scores, outpath=out_nn_model/'nn_scores.txt')

    # -----------------
    # Summarize results
    # -----------------
    # Plots
    # plts_path = outdir/'plts'
    # os.makedirs(plts_path, exist_ok=True)
    ml_models.plot_prfrm_metrics(history, title=f'Train base model {model_name}',
                                 skp_ep=skp_ep, add_lr=True, outdir=out_nn_model)

    # Save keras history
    ml_models.save_krs_history(history, out_nn_model)


    # =====================================================
    #       Train LGBM
    # =====================================================
     
    lg.logger.info('\n{}'.format('=' * 50))
    lg.logger.info('Train LGBM ...')
    
    # Define model
    init_kwargs = {'objective': 'regression', 'n_estimators': 100, 'n_jobs': n_jobs, 'random_state': SEED}    
    model = lgb.LGBMModel(**init_kwargs)

    # Train
    fit_kwargs = {'verbose': verbose}
    t0 = time()
    model.fit(xtr, ytr, **fit_kwargs)
    lg.logger.info('Train time: {:.1f} mins'.format( (time()-t0)/60 ))

    # Predict
    pred_ytr = model.predict(xtr)
    pred_yvl = model.predict(xvl)
    pred_yte = model.predict(xte)

    # Calc scores
    lg.logger.info('\nScores LGBM:')
    lgbm_scores = {}
    lgbm_scores['r2_tr'] = r2_score(ytr, pred_ytr)
    lgbm_scores['r2_vl'] = r2_score(yvl, pred_yvl)
    lgbm_scores['r2_te'] = r2_score(yte, pred_yte)
    lgbm_scores['mae_tr'] = mean_absolute_error(ytr, pred_ytr)
    lgbm_scores['mae_vl'] = mean_absolute_error(yvl, pred_yvl)
    lgbm_scores['mae_te'] = mean_absolute_error(yte, pred_yte)
    for k, v, in lgbm_scores.items(): lg.logger.info(f'{k}: {v}')
    utils.dump_dict(lgbm_scores, outpath=run_outdir/'lgbm_scores.txt')



    # Finish and kill logger
    lg.kill_logger()



def main(args):
    args = parse_args(args)
    args = vars(args)
    ret = run(args)
    

if __name__ == '__main__':
    """ __name__ == '__main__' explained: www.youtube.com/watch?v=sugvnHA7ElY """
    main(sys.argv[1:])


