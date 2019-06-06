""" This is the main script. """
from __future__ import print_function, division

import warnings
warnings.filterwarnings('ignore')

# from comet_ml import Experiment
import os

import sys
from pathlib import Path
import psutil
import argparse
from datetime import datetime
from time import time
from pprint import pprint
from collections import OrderedDict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sklearn
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split

from pandas.api.types import is_string_dtype
from sklearn.preprocessing import LabelEncoder

# Import custom callbacks
keras_contrib = '/vol/ml/apartin/projects/keras-contrib/keras_contrib/callbacks'
sys.path.append(keras_contrib)
#from callbacks import *
from cyclical_learning_rate import CyclicLR

SEED = None


# File path
# file_path = os.path.dirname(os.path.realpath(__file__))
file_path = Path(__file__).resolve().parent


# Utils
utils_path = file_path / '../../utils'
sys.path.append(str(utils_path))
import utils
from utils_tidy import load_tidy_combined, get_data_by_src, break_src_data 
import argparser
from classlogger import Logger
import ml_models
from cv_splitter import cv_splitter, plot_ytr_yvl_dist
from lrn_crv import my_learning_curve

from models import nn_model1 #,nn_model2,nn_ model3

# Path
PRJ_NAME = file_path.name 
OUTDIR = file_path / '../../out/' / PRJ_NAME
CONFIGFILENAME = 'config_prms.txt'


def create_outdir(outdir, args, src):
    t = datetime.now()
    t = [t.year, '-', t.month, '-', t.day, '_', 'h', t.hour, '-', 'm', t.minute]
    t = ''.join([str(i) for i in t])
    
    l = [args['opt']] + [args['cv_method']] + [('cvf'+str(args['cv_folds']))] + args['cell_features'] + args['drug_features'] + [args['target_name']]
    name_sffx = '.'.join( src + [args['model_name']] + l )

    outdir = Path(outdir) / (name_sffx + '_' + t)
    os.makedirs(outdir)
    return outdir


def run(args):
    outdir = args['outdir']
    target_name = args['target_name']
    target_transform = args['target_transform']    
    tr_sources = args['train_sources']
    te_sources = args['test_sources']
    row_sample = args['row_sample']
    col_sample = args['col_sample']
    tissue_type = args['tissue_type']
    cell_fea = args['cell_features']
    drug_fea = args['drug_features']
    other_fea = args['other_features']
    model_name = args['model_name']
    cv_method = args['cv_method']
    cv_folds = args['cv_folds']
    retrain = args['retrain']
    n_jobs = args['n_jobs']

    epochs = args['epochs']
    batch_size = args['batch_size']
    dr_rate = args['dr_rate']
    opt_name = args['opt']
    attn = args['attn']

    # Extract ml type ('reg' or 'cls')
    #mltype = args['model_name'].split('_')[-1]
    #assert mltype in ['reg', 'cls'], "mltype should be either 'reg' or 'cls'."    
    mltype = 'reg'
    
    # Feature list
    fea_list = cell_fea + drug_fea + other_fea

    # Define names
    tr_sources_name = '_'.join(tr_sources)
        
    # Define custom metric to calc auroc from regression
    # scikit-learn.org/stable/modules/model_evaluation.html#scoring
    def reg_auroc(y_true, y_pred):
        y_true = np.where(y_true < 0.5, 1, 0)
        y_score = np.where(y_pred < 0.5, 1, 0)
        auroc = sklearn.metrics.roc_auc_score(y_true, y_score)
        return auroc
    reg_auroc_score = sklearn.metrics.make_scorer(score_func=reg_auroc, greater_is_better=True)

    # Define metrics
    # TODO: find way to pass metrics to calc_scores in ml_models.py
    metrics = {'r2': 'r2', #sklearn.metrics.r2_score,
               'neg_mean_absolute_error': 'neg_mean_absolute_error', #sklearn.metrics.neg_mean_absolute_error,
               'neg_median_absolute_error': 'neg_median_absolute_error', #sklearn.metrics.neg_median_absolute_error,
               'neg_mean_squared_error': 'neg_mean_squared_error', #sklearn.metrics.neg_mean_squared_error,
               'reg_auroc_score': reg_auroc_score,
    }


    # ========================================================================
    #       Logger
    # ========================================================================
    run_outdir = create_outdir(outdir, args=args, src=tr_sources)
    logfilename = run_outdir/'logfile.log'
    lg = Logger(logfilename)

    lg.logger.info(f'File path: {file_path}')
    lg.logger.info(f'System CPUs: {psutil.cpu_count(logical=True)}')
    lg.logger.info(f'n_jobs: {n_jobs}')

    # Dump args to file
    utils.dump_args(args, run_outdir)
    

    # ========================================================================
    #       Load data
    # ========================================================================
    DATADIR = file_path / '../../data/processed/from_combined/tidy_drop_fibro'
    DATAFILENAME = 'tidy_data.parquet'
    datapath = DATADIR / DATAFILENAME

    dataset = load_tidy_combined(
            datapath, fea_list=fea_list, logger=lg.logger, random_state=SEED)

    tr_data = get_data_by_src(
            dataset, src_names=tr_sources, logger=lg.logger)
    
    xdata, ydata, meta, tr_scaler = break_src_data(
            tr_data, target=args['target_name'],
            scaler_method=args['scaler'], logger=lg.logger)


    # ========================================================================
    #       Define CV split
    # ========================================================================
    cv = cv_splitter(cv_method=cv_method, cv_folds=cv_folds, test_size=0.2,
                     mltype=mltype, shuffle=True, random_state=SEED)
    if cv_method=='simple':
        groups = None
    elif cv_method=='group':
        groups = tr_data['CELL'].copy()


    # ========================================================================
    #       CV training
    # ========================================================================
    lg.logger.info('\n{}'.format('='*50))
    lg.logger.info(f'CV training ... {tr_sources}')
    lg.logger.info('='*50)

    # ML model params
    """
    if model_name == 'lgb_reg':
        init_prms = {'n_jobs': n_jobs, 'random_state': SEED, 'logger': lg.logger}
        fit_prms = {'verbose': False}  # 'early_stopping_rounds': 10, 'sample_weight': sample_weight
    elif model_name == 'nn_reg':
        init_prms = {'input_dim': xdata.shape[1], 'dr_rate': dr_rate, 'opt_name': opt_name, 'attn': attn, 'logger': lg.logger}
        fit_prms = {'batch_size': batch_size, 'epochs': epochs, 'verbose': 1}  # 'validation_split': 0.1
    """
    X = pd.DataFrame(xdata).values
    Y = pd.DataFrame(ydata).values

    # TODO: didn't test!
    if isinstance(cv, int) and groups is None:
        cv_folds = cv
        cv = KFold(n_splits=cv_folds, shuffle=False, random_state=random_state)
    if isinstance(cv, int) and groups is not None:
        cv_folds = cv
        cv = GroupKFold(n_splits=cv_folds)
    else:
        cv_folds = cv.get_n_splits()

    if is_string_dtype(groups):
        group_encoder = LabelEncoder()
        groups = group_encoder.fit_transform(groups)
    
    # ... Now start a nested loop of train size and cv folds ...
    tr_scores_all = [] # list dicts
    vl_scores_all = [] # list dicts

    if mltype == 'cls':
        if Y.ndim > 1 and Y.shape[1] > 1:
            splitter = cv.split(X, np.argmax(Y, axis=1), groups=groups)
        else:
            splitter = cv.split(X, Y, groups=groups)
    elif mltype == 'reg':
        splitter = cv.split(X, Y, groups=groups)

    # Placeholder to save the best model
    best_model = None
    best_score = -np.Inf

    # Start CV iters
    for fold_id, (tr_idx, vl_idx) in enumerate(splitter):
        if lg.logger is not None:
            lg.logger.info(f'Fold {fold_id+1}/{cv_folds}')

        # Samples from this dataset are sampled for training
        xtr = X[tr_idx, :]
        ytr = np.squeeze(Y[tr_idx, :])

        # A fixed set of validation samples for the current CV split
        xvl = X[vl_idx, :]
        yvl = np.squeeze(Y[vl_idx, :])        

        # Get the estimator
        #estimator = ml_models.get_model(model_name, init_params)
        if model_name == 'nn_model1':
            init_prms = {'input_dim': xdata.shape[1], 'dr_rate': dr_rate, 'opt_name': opt_name, 'logger': lg.logger}
            model = nn_model1(**init_prms)
        elif model_name == 'nn_model2':
            init_prms = {}
            model = nn_model2(**init_prms)
        elif model_name == 'nn_model3':
            init_prms = {}
            model = nn_model3(**init_prms)
        fit_params = {'batch_size': batch_size, 'epochs': epochs, 'verbose': 1}  # 'validation_split': 0.1

        if 'nn' in model_name:
            from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard

            # Create output dir
            #out_nn_model = outdir / ('cv'+str(fold_id+1))
            out_nn_model = run_outdir / ('cv'+str(fold_id+1))
            os.makedirs(out_nn_model, exist_ok=False)
            
            # Callbacks (custom)
            clr = CyclicLR(base_lr=0.0001, max_lr=0.001, mode='triangular')
                
            # Keras callbacks
            checkpointer = ModelCheckpoint(str(out_nn_model/'autosave.model.h5'), verbose=0, save_weights_only=False, save_best_only=True)
            csv_logger = CSVLogger(out_nn_model/'training.log')
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.75, patience=20, verbose=1, mode='auto',
                                          min_delta=0.0001, cooldown=3, min_lr=0.000000001)
            early_stop = EarlyStopping(monitor='val_loss', patience=60, verbose=1, mode='auto')
                
            # Callbacks list
            if (args is not None) and (args['opt']=='clr'):
                callback_list = [checkpointer, csv_logger, early_stop, reduce_lr, clr]
            else:
                callback_list = [checkpointer, csv_logger, early_stop, reduce_lr]

            # Fit params
            fit_params['validation_data'] = (xvl, yvl)
            fit_params['callbacks'] = callback_list

        # Train model
        history = model.fit(xtr, ytr, **fit_params)

        # Calc preds and scores TODO: dump preds
        # ... training set
        y_preds, y_true = utils.calc_preds(estimator=model, x=xtr, y=ytr, mltype=mltype)
        tr_scores = utils.calc_scores(y_true=y_true, y_preds=y_preds, mltype=mltype)
        # ... val set
        y_preds, y_true = utils.calc_preds(estimator=model, x=xvl, y=yvl, mltype=mltype)
        vl_scores = utils.calc_scores(y_true=y_true, y_preds=y_preds, mltype=mltype)

        # Save the best model
        if mltype == 'cls':
            vl_scores['f1_score'] > best_score
            best_score = vl_scores['f1_score']
            best_model = estimator
        elif mltype == 'reg':
            vl_scores['r2'] > best_score
            best_score = vl_scores['r2']
            best_model = estimator

        # Plot training curves
        if 'nn' in model_name:
            ml_models.plot_prfrm_metrics(history=history, title=f'cv fold: {fold_id+1}',
                                         skp_ep=7, add_lr=True, outdir=out_nn_model)

        # Add info
        tr_scores['tr_set'] = True
        vl_scores['tr_set'] = False
        tr_scores['fold'] = 'f'+str(fold_id)
        vl_scores['fold'] = 'f'+str(fold_id)

        # Aggregate scores
        tr_scores_all.append(tr_scores)
        vl_scores_all.append(vl_scores)

        # Delete the estimator/model
        del estimator, history

        # Comet log fold scores
        # https://medium.com/comet-ml/building-reliable-machine-learning-models-with-cross-validation-20b2c3e32f3e
#         if (args is not None) and ('comet' in args):
#             experiment = args['comet']
#             experiment.log_metric('Fold {}'.format(fold_id), vl_scores['r2'])
            
    tr_df = scores_to_df(tr_scores_all)
    vl_df = scores_to_df(vl_scores_all)
    scores_all_df = pd.concat([tr_df, vl_df], axis=0)


    # Comet log fold scores
#     if (args is not None) and ('comet' in args):
#         experiment = args['comet']
#         experiment.log_metric('Best score', best_score)

    return scores_all_df, best_model

    """
    # ------------
    # My CV method - (works with keras)
    # ------------
    t0 = time()
    cv_scores, best_model = my_cross_validate(
        X=xdata,
        Y=ydata,
        mltype=mltype,
        model_name=model_name,
        fit_params=fit_prms,
        init_params=init_prms,
        args=args,
        cv=cv,
        groups=groups,
        n_jobs=n_jobs, random_state=SEED, logger=lg.logger, outdir=run_outdir)
    lg.logger.info('Runtime: {:.1f} mins'.format( (time()-t0)/60) )

    # Dump results
    # cv_scores = cv_scores.round(3)
    cv_scores.to_csv( run_outdir/('cv_scores_' + tr_sources_name + '.csv'), index=False )
    lg.logger.info(f'\ncv_scores\n{cv_scores}')
    """


def main(args):
    config_fname = file_path / CONFIGFILENAME
    args = argparser.get_args(args=args, config_fname=config_fname)
    ## pprint(vars(args))
    args = vars(args)
    if args['outdir'] is None:
        args['outdir'] = OUTDIR
    
    #args = None
    ret = run(args)


if __name__ == '__main__':
    main(sys.argv[1:])
 
