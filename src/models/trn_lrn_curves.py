from __future__ import print_function
from __future__ import division

import warnings
warnings.filterwarnings('ignore')

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
import time
import datetime
import logging
import psutil
import re
from pprint import pprint
from collections import OrderedDict
import numpy as np
import pandas as pd

import argparse
import configparser

import matplotlib
# matplotlib.use('TkAgg')
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats
import sklearn
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import learning_curve
from sklearn.model_selection import cross_val_score, cross_validate, learning_curve

from sklearn.model_selection import ShuffleSplit, KFold
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

# Get file path
# ... manutally run ...
# file_path = os.getcwd()
# file_path = os.path.join(file_path, 'src/models')
# os.chdir(file_path)
# ... uato ...
file_path = os.path.dirname(os.path.realpath(__file__))  # os.path.dirname(os.path.abspath(__file__))

# Utils
# utils_path = os.path.abspath(os.path.join(file_path, 'utils'))
# sys.path.append(utils_path)
import utils
import utils_tidy
import argparser
import classlogger
import lrn_curve
import ml_models
from cvsplitter import GroupSplit, SimpleSplit, plot_ytr_yvl_dist

DATADIR = os.path.join(file_path, '../../data/processed/from_combined')
OUTDIR = os.path.join(file_path, '../../models/from_combined')
DATAFILENAME = 'tidy_data_no_fibro.parquet'
# DATAFILENAME = 'tidy_data.parquet'
CONFIGFILENAME = 'config_params.txt'
os.makedirs(OUTDIR, exist_ok=True)

SEED = 0

# Feature prefix (some already present in the tidy dataframe)
fea_prfx_dict = {'rna': 'cell_rna.', 'cnv': 'cell_cnv.',
                 'dsc': 'drug_dsc.', 'fng': 'drug_fng.',
                 'clb': 'cell_lbl.', 'dlb': 'drug_lbl.'}

np.set_printoptions(precision=3)


def run(args):
    target_name = args['target_name']
    target_transform = args['target_transform']    
    train_sources = args['train_sources']
    test_sources = args['test_sources']
    row_sample = args['row_sample']
    col_sample = args['col_sample']
    tissue_type = args['tissue_type']
    cell_features = args['cell_features']
    drug_features = args['drug_features']
    other_features = args['other_features']
    mltype = args['mltype']
    mlmodel = args['mlmodel']
    cv_method = args['cv_method']
    cv_folds = args['cv_folds']
    lr_curve_ticks = args['lc_ticks']
    verbose = args['verbose']
    n_jobs = args['n_jobs']

    epochs = args['epochs']
    batch_size = args['batch_size']
    dr_rate = args['dr_rate']
    attn = args['attn']

    outdir = args['outdir']

    # Feature list
    feature_list = cell_features + drug_features + other_features

    # Define names
    train_sources_name = '_'.join(train_sources)
    
    # Define custom metric to calc auroc from regression
    # https://scikit-learn.org/stable/modules/model_evaluation.html#scoring
    from sklearn.metrics import make_scorer
    from sklearn.metrics import roc_auc_score
    def reg_auroc(y_true, y_pred):
            y_true = np.where(y_true < 0.5, 1, 0)
            y_score = np.where(y_pred < 0.5, 1, 0)
            auroc = roc_auc_score(y_true, y_score)
            return auroc
    reg_auroc_score = make_scorer(score_func=reg_auroc, greater_is_better=True)

    # Define metrics
    # TODO: find way to pass metrics to calc_scores in ml_models.py
    # metrics = ['r2', 'neg_mean_absolute_error', 'neg_median_absolute_error',
    #            'neg_mean_squared_error']
    metrics = {'r2': 'r2', #sklearn.metrics.r2_score,
               'neg_mean_absolute_error': 'neg_mean_absolute_error', #sklearn.metrics.neg_mean_absolute_error,
               'neg_median_absolute_error': 'neg_median_absolute_error', #sklearn.metrics.neg_median_absolute_error,
               'neg_mean_squared_error': 'neg_mean_squared_error', #sklearn.metrics.neg_mean_squared_error,
               'reg_auroc_score': reg_auroc_score,
    }


    # ========================================================================
    #       Logger
    # ========================================================================
    run_outdir = utils.create_outdir(outdir=outdir, args=args)
    logfilename = os.path.join(run_outdir, 'logfile.log')
    lg = classlogger.Logger(logfilename=logfilename)

    lg.logger.info(f'File path: {file_path}')
    lg.logger.info(f'System CPUs: {psutil.cpu_count()}')
    lg.logger.info(f'n_jobs: {n_jobs}')


    # ========================================================================
    #       Load data and pre-proc
    # ========================================================================
    datapath = os.path.join(DATADIR, DATAFILENAME)
    data, te_data = utils_tidy.load_data(datapath=datapath, fea_prfx_dict=fea_prfx_dict,
                                         args=args, logger=lg.logger, random_state=SEED)


    # ========================================================================
    #       Keep a subset of training features
    # ========================================================================
    data = utils_tidy.extract_subset_features(data=data, feature_list=feature_list, fea_prfx_dict=fea_prfx_dict)


    # ========================================================================
    #       Define CV split
    # ========================================================================
    # Split tr/vl data
    if mltype == 'cls':
        if cv_method == 'simple':
            if cv_folds == 1:
                cv = ShuffleSplit(n_splits=cv_folds, test_size=0.2, random_state=SEED)
            else:
                cv = KFold(n_splits=cv_folds, shuffle=True, random_state=SEED)
            groups = None
        elif cv_method == 'stratify':
            if cv_folds == 1:
                cv = StratifiedShuffleSplit(n_splits=cv_folds, test_size=0.2, random_state=SEED)
            else:
                cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=SEED)
            groups = None
    elif mltype == 'reg':
        if cv_method == 'group':
            if cv_folds == 1:
                cv = GroupShuffleSplit(random_state=SEED)
            else:
                cv = GroupKFold(n_splits=cv_folds)
            groups = data['CELL'].copy()
        elif cv_method == 'simple':
            if cv_folds == 1:
                cv = ShuffleSplit(n_splits=cv_folds, test_size=0.2, random_state=SEED)
            else:
                cv = KFold(n_splits=cv_folds, shuffle=True, random_state=SEED)
            groups = None


    # ========================================================================
    #       Learning curves
    # ========================================================================
    lg.logger.info('\n\n{}'.format('='*50))
    lg.logger.info('Learning curves ...')
    lg.logger.info('='*50)

    # Get the data
    xdata, _ = utils_tidy.split_features_and_other_cols(data, fea_prfx_dict=fea_prfx_dict)
    ydata = utils_tidy.extract_target(data=data, target_name=target_name)
    utils_tidy.print_feature_shapes(df=xdata, logger=lg.logger)

    # ML model params
    if mlmodel == 'lgb_reg':
        init_prms = {'n_jobs': n_jobs, 'random_state': SEED, 'logger': lg.logger}
        fit_prms = {'verbose': False}  # 'early_stopping_rounds': 10,
    elif mlmodel == 'nn_reg':
        init_prms = {'input_dim': xdata.shape[1], 'dr_rate': dr_rate, 'attn': attn, 'logger': lg.logger}
        fit_prms = {'batch_size': batch_size, 'epochs': epochs, 'verbose': 2, 'validation_split': 0.2}

    # Define ML model
    model = ml_models.get_model(mlmodel=mlmodel, init_params=init_prms)  


    # -----------------------------------------------
    # Generate learning curve - semi automatic method
    # (*) uses cross_validate from sklearn.
    # -----------------------------------------------
    lg.logger.info('\nStart learning curve (my method) ...')

    # Run learning curve
    t0 = time.time()
    lrn_curve_scores = lrn_curve.my_learning_curve(
        X=xdata, Y=ydata,
        mltype=mltype,
        mlmodel=mlmodel,
        fit_params=fit_prms,
        init_params=init_prms,
        args=args,
        lr_curve_ticks=lr_curve_ticks,
        data_sizes_frac=None,
        metrics=metrics,
        cv=cv,
        groups=groups,
        n_jobs=n_jobs, random_state=SEED, logger=lg.logger, outdir=run_outdir)
    lg.logger.info('Runtime: {:.3f} mins'.format((time.time()-t0)/60))

    # Dump results
    lrn_curve_scores.to_csv(os.path.join(run_outdir, 'lrn_curve_scores.csv'), index=False) 


    # -----------------------------------------------
    # Generate learning curve - my method
    # (*) ...
    # -----------------------------------------------
    # from cvrun import my_cv_run
    # df_tr = []
    # df_vl = []
    # data_sizes_frac = np.linspace(0.1, 1.0, lr_curve_ticks)
    # data_sizes = [int(n) for n in data.shape[0]*data_sizes_frac]
    
    # for d_size in data_sizes:
    #     lg.logger.info(f'Data size: {d_size}')
    #     data_sample = data.sample(n=d_size)

    #     tr_cv_scores, vl_cv_scores = my_cv_run(
    #         data=data_sample,
    #         target_name=target_name,
    #         fea_prfx_dict=fea_prfx_dict,
    #         model=model.model, #mlmodel=mlmodel,
    #         cv_method=cv_method, cv_folds=cv_folds,
    #         logger=lg.logger, random_state=SEED, outdir=run_outdir)

    #     # Add col that indicates d_size
    #     tr_cv_scores.insert(loc=1, column='data_size', value=data_sample.shape[0])
    #     vl_cv_scores.insert(loc=1, column='data_size', value=data_sample.shape[0])
        
    #     # Append results to master dfs
    #     df_tr.append(tr_cv_scores)
    #     df_vl.append(vl_cv_scores)

    # # Concat the results for all the data_sizes
    # df_tr = pd.concat(df_tr, axis=0)
    # df_vl = pd.concat(df_vl, axis=0)

    # lrn_curve.plt_learning_curve_multi_metric(df_tr=df_tr, df_vl=df_vl,
    #                                           cv_folds=cv_folds, target_name=target_name,
    #                                           outdir=run_outdir)


    # -------------------------------------------------
    # Generate learning curve - complete sklearn method
    # (*) can't generate multiple metrics.
    # -------------------------------------------------
    # lg.logger.info("\nStart learning_curve (sklearn) ...")

    # # Define params
    # metric_name = 'neg_mean_absolute_error'
    # # train_sizes_frac = np.linspace(0.1, 1.0, lr_curve_ticks)
    # base = 10
    # train_sizes_frac = np.logspace(0.0, 1.0, lr_curve_ticks, endpoint=True, base=base)/base

    # # Run learning curve
    # t0 = time.time()
    # lrn_curve_scores = learning_curve(
    #     estimator=model.model, X=xdata, y=ydata,
    #     train_sizes=train_sizes_frac, cv=cv, groups=groups,
    #     scoring=metric_name,
    #     n_jobs=n_jobs, exploit_incremental_learning=False,
    #     random_state=SEED, verbose=1, shuffle=False)
    # lg.logger.info('Runtime: {:.3f} mins'.format((time.time()-t0)/60))

    # # Dump results
    # # lrn_curve_scores = utils.cv_scores_to_df(lrn_curve_scores, decimals=3, calc_stats=False) # this func won't work
    # # lrn_curve_scores.to_csv(os.path.join(run_outdir, 'lrn_curve_scores_auto.csv'), index=False)
    
    # # Plot learning curves
    # lrn_curve.plt_learning_curve(rslt=lrn_curve_scores, metric_name=metric_name,
    #     title='Learning curve (target: {}, data: {})'.format(target_name, train_sources_name),
    #     path=os.path.join(run_outdir, 'auto_learning_curve_' + target_name + '_' + metric_name + '.png'))

    return lrn_curve_scores
   

def main(args):
    config_fname = os.path.join(file_path, CONFIGFILENAME)
    args = argparser.get_args(args=args, config_fname=config_fname)
    pprint(vars(args))
    args = vars(args)
    #if 'outdir' not in args:
    if args['outdir'] is None:
        args['outdir'] = OUTDIR
    lrn_curve_scores = run(args)
    return lrn_curve_scores, args
    

if __name__ == '__main__':
    # python -m pdb src/models/trn_lrn_curves.py -te ccle gcsi -tr gcsi
    """ __name__ == '__main__' explained:
    https://www.youtube.com/watch?v=sugvnHA7ElY
    """
    """
    https://stackoverflow.com/questions/14500183/in-python-can-i-call-the-main-of-an-imported-module
    How to run code with input args from another code?
    This will be used with multiple train and test sources.
    For example: in launch_model_transfer.py
        import train_from_combined.py
        train_from_combined.main([tr_src, tst_src])
    """
    # python -m pdb src/models/trn_lrn_curves.py -te ccle gcsi -tr gcsi
    main(sys.argv[1:])
