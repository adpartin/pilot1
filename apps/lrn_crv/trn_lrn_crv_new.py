""" ... """
from __future__ import print_function, division

import warnings
warnings.filterwarnings('ignore')

from comet_ml import Experiment
import os

import sys
from pathlib import Path 
import psutil
from datetime import datetime
from time import time
from pprint import pprint

import sklearn
import numpy as np
import pandas as pd

import matplotlib
# matplotlib.use('TkAgg')
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy import stats
np.set_printoptions(precision=3)

from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import learning_curve
from sklearn.model_selection import cross_val_score, cross_validate, learning_curve

from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit, KFold
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

SEED = None
t_start = time()


# File path
# file_path = os.path.dirname(os.path.realpath(__file__))
file_path = Path(__file__).resolve().parent


# Utils
# utils_path = os.path.abspath(os.path.join(file_path, 'utils'))
# sys.path.append(utils_path)
utils_path = file_path / '../../utils'
sys.path.append(str(utils_path))
import utils
from utils_tidy import load_tidy_combined, get_data_by_src, break_src_data 
import argparser
from classlogger import Logger
import ml_models
from cv_splitter import cv_splitter, plot_ytr_yvl_dist
from lrn_crv import my_learning_curve


# Path
PRJ_NAME = file_path.name
OUTDIR = file_path / '../../out/' / PRJ_NAME
# DATADIR = file_path / '../../data/processed/from_combined/tidy_drop_fibro'
# DATAFILENAME = 'tidy_data.parquet'
CONFIGFILENAME = 'config_prms.txt'
COMET_PRJ_NAME = 'trn_lrn_curves'
os.makedirs(OUTDIR, exist_ok=True)


#def run(args):
def trn_lrn_crv_new(xdata, ydata, args):
    # Data name
    dname = args['dname']
    if dname == 'top6':
        args['train_sources'] = ['top6']
        args['test_sources'] = ['top6']
        args['target_name'] = 'AUC1'

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
    lrn_crv_ticks = args['lc_ticks']
    n_jobs = args['n_jobs']

    epochs = args['epochs']
    batch_size = args['batch_size']
    dr_rate = args['dr_rate']
    opt_name = args['opt']
    attn = args['attn']

    # Extract ml type ('reg' or 'cls')
    mltype = args['model_name'].split('_')[-1]
    assert mltype in ['reg', 'cls'], "mltype should be either 'reg' or 'cls'."    
    
    # Feature list
    fea_list = cell_fea + drug_fea + other_fea

    # Define names
    tr_sources_name = '_'.join(tr_sources)
    
    # Print args
    pprint(args)
    
    # Define custom metric to calc auroc from regression
    # scikit-learn.org/stable/modules/model_evaluation.html#scoring
    def reg_auroc(y_true, y_pred):
        y_true = np.where(y_true < 0.5, 1, 0)
        y_score = np.where(y_pred < 0.5, 1, 0)
        auroc = sklearn.metrics.roc_auc_score(y_true, y_score)
        return auroc
    reg_auroc_score = sklearn.metrics.make_scorer(score_func=reg_auroc, greater_is_better=True)

    # Define metrics
    metrics = {'r2': 'r2',
               'neg_mean_absolute_error': 'neg_mean_absolute_error', #sklearn.metrics.neg_mean_absolute_error,
               'neg_median_absolute_error': 'neg_median_absolute_error', #sklearn.metrics.neg_median_absolute_error,
               'neg_mean_squared_error': 'neg_mean_squared_error', #sklearn.metrics.neg_mean_squared_error,
               'reg_auroc_score': reg_auroc_score,
    }


    # ========================================================================
    #       Logger
    # ========================================================================
    run_outdir = utils.create_outdir(outdir, args=args)
    logfilename = run_outdir/'logfile.log'
    lg = Logger(logfilename)

    lg.logger.info(f'File path: {file_path}')
    lg.logger.info(f'System CPUs: {psutil.cpu_count(logical=True)}')
    lg.logger.info(f'n_jobs: {n_jobs}')

    # Dump args to file
    utils.dump_args(args, run_outdir)

    
    # ========================================================================
    #       Comet
    # ========================================================================
    # comet.ml/docs/python-sdk/Experiment
    COMET_API_KEY = os.environ.get('COMET_API_KEY')
    if COMET_API_KEY is not None:
        # args['comet_prj_name'] = COMET_PRJ_NAME
        # args['comet_set_name'] = tr_sources_name # model_name
        comet_dict = {'comet_prj_name': COMET_PRJ_NAME,
                      'comet_set_name': tr_sources_name}
        args['comet'] = comet_dict
#         lg.logger.info('\ncomet_api_key:        {}'.format(COMET_API_KEY))
#         # lg.logger.info('comet_workspace_name: {}'.format(PRJ_NAME))
#         lg.logger.info('comet_project_name:   {}'.format(COMET_PRJ_NAME))
#         experiment = Experiment(api_key=COMET_API_KEY,
#                                 # workspace=PRJ_NAME,
#                                 project_name=COMET_PRJ_NAME)
#         experiment.set_name(train_sources_name)  # set experiment name
#         args['comet'] = experiment
    
    
    # ========================================================================
    #       Load data and pre-proc
    # ========================================================================
#     if dname == 'combined':
#         DATADIR = file_path / '../../data/processed/from_combined/tidy_drop_fibro'
#         DATAFILENAME = 'tidy_data.parquet'
#         datapath = DATADIR / DATAFILENAME
    
#         dataset = load_tidy_combined(
#                 datapath, fea_list=fea_list, logger=lg.logger, random_state=SEED)

#         tr_data = get_data_by_src(
#                 dataset, src_names=tr_sources, logger=lg.logger)
    
#         xdata, ydata, _, _ = break_src_data(
#                 tr_data, target=args['target_name'],
#                 scaler_method=args['scaler'], logger=lg.logger)
#         del tr_data

#     elif dname == 'top6':
#         DATADIR = file_path / '../../data/raw/'
#         DATAFILENAME = 'uniq.top6.reg.parquet'
#         datapath = DATADIR / DATAFILENAME
        
#         df = pd.read_parquet(datapath, engine='auto', columns=None)
#         df = df.sample(frac=1.0, axis=0, random_state=SEED).reset_index(drop=True)

#         scaler_method = args['scaler']
#         if  scaler_method is not None:
#             if scaler_method == 'stnd':
#                 scaler = StandardScaler()
#             elif scaler_method == 'minmax':
#                 scaler = MinMaxScaler()
#             elif scaler_method == 'rbst':
#                 scaler = RobustScaler()

#         xdata = df.iloc[:, 1:]
#         ydata = df.iloc[:, 0]
#         xdata = scaler.fit_transform(xdata).astype(np.float32)


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
    #       Learning curves
    # ========================================================================
    lg.logger.info('\n\n{}'.format('='*50))
    if dname == 'combined':
        lg.logger.info(f'Learning curves ... {tr_sources}')
    elif dname == 'top6':
        lg.logger.info('Learning curves ... (Top6)')
    lg.logger.info('='*50)

    # ML model params
    if model_name == 'lgb_reg':
        init_prms = {'n_jobs': n_jobs, 'random_state': SEED, 'logger': lg.logger}
        fit_prms = {'verbose': False}  # 'early_stopping_rounds': 10,
    elif model_name == 'nn_reg':
        init_prms = {'input_dim': xdata.shape[1], 'dr_rate': dr_rate, 'opt_name': opt_name, 'attn': attn, 'logger': lg.logger}
        #fit_prms = {'batch_size': batch_size, 'epochs': epochs, 'verbose': 1, 'validation_split': 0.2} 
        fit_prms = {'batch_size': batch_size, 'epochs': epochs, 'verbose': 1} 

    # -----------------------------------------------
    # Generate learning curve 
    # -----------------------------------------------
    lg.logger.info('\nStart learning curve (my method) ...')

    # Run learning curve
    t0 = time()
    lrn_crv_scores = my_learning_curve(
        X=xdata,
        Y=ydata,
        lrn_crv_ticks=lrn_crv_ticks,
        data_sizes_frac=None,
        mltype=mltype,
        model_name=model_name,
        fit_params=fit_prms,
        init_params=init_prms,
        args=args,
        metrics=metrics,
        cv=cv,
        groups=groups,
        n_jobs=n_jobs, random_state=SEED, logger=lg.logger, outdir=run_outdir)
    lg.logger.info('Runtime: {:.1f} mins'.format( (time()-t0)/60) )

    # Dump results
    lrn_crv_scores.to_csv( run_outdir/('lrn_crv_scores_' + tr_sources_name + '.csv'), index=False) 
    lg.logger.info(f'\nlrn_crv_scores\n{lrn_crv_scores}')

    # -------------------------------------------------
    # Generate learning curve - complete sklearn method
    # (*) Problem: can't generate multiple metrics.
    # -------------------------------------------------
    """
    # Define params
    metric_name = 'neg_mean_absolute_error'
    base = 10
    train_sizes_frac = np.logspace(0.0, 1.0, lrn_crv_ticks, endpoint=True, base=base)/base

    # Run learning curve
    t0 = time()
    lrn_curve_scores = learning_curve(
        estimator=model.model, X=xdata, y=ydata,
        train_sizes=train_sizes_frac, cv=cv, groups=groups,
        scoring=metric_name,
        n_jobs=n_jobs, exploit_incremental_learning=False,
        random_state=SEED, verbose=1, shuffle=False)
    lg.logger.info('Runtime: {:.1f} mins'.format( (time()-t0)/60) )

    # Dump results
    # lrn_curve_scores = utils.cv_scores_to_df(lrn_curve_scores, decimals=3, calc_stats=False) # this func won't work
    # lrn_curve_scores.to_csv(os.path.join(run_outdir, 'lrn_curve_scores_auto.csv'), index=False)
    
    # Plot learning curves
    lrn_crv.plt_learning_curve(rslt=lrn_curve_scores, metric_name=metric_name,
        title='Learning curve (target: {}, data: {})'.format(target_name, tr_sources_name),
        path=os.path.join(run_outdir, 'auto_learning_curve_' + target_name + '_' + metric_name + '.png'))
    """

    # Kill logger
    lg.kill_logger()
    del xdata, ydata
    return lrn_crv_scores
   

# def main(args):
#     config_fname = file_path / CONFIGFILENAME
#     args = argparser.get_args(args=args, config_fname=config_fname)
#     # pprint(vars(args))
#     args = vars(args)
#     if args['outdir'] is None:
#         args['outdir'] = OUTDIR
#     lrn_crv_scores = run(args)
#     return lrn_crv_scores, args
    

# if __name__ == '__main__':
#     # python -m pdb apps/lrn_crv/trn_lrn_crv.py -tr gcsi
#     """ __name__ == '__main__' explained:
#     www.youtube.com/watch?v=sugvnHA7ElY
#     """
#     """
#     stackoverflow.com/questions/14500183/in-python-can-i-call-the-main-of-an-imported-module
#     How to run code with input args from another code?
#     This will be used with multiple train and test sources.
#     For example: in launch_model_transfer.py
#         import trn_from_combined.py
#         train_from_combined.main([tr_src, tst_src])
#     """
#     # python -m pdb apps/lrn_crv/trn_lrn_crv.py -te ccle gcsi -tr gcsi
#     main(sys.argv[1:])
