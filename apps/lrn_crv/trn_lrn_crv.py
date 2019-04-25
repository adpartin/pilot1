from __future__ import print_function, division

import warnings
warnings.filterwarnings('ignore')

from comet_ml import Experiment
import os

import sys
from pathlib import Path 
import psutil
import datetime
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

from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import learning_curve
from sklearn.model_selection import cross_val_score, cross_validate, learning_curve

from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit, KFold
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

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
import utils_tidy
import argparser
from classlogger import Logger
import lrn_crv
import ml_models
from cvsplitter import GroupSplit, SimpleSplit, plot_ytr_yvl_dist


# Path
PRJ_NAME = file_path.name
DATADIR = file_path / '../../data/processed/from_combined/tidy_drop_fibro'
OUTDIR = file_path / '../../out/' / PRJ_NAME
DATAFILENAME = 'tidy_data.parquet'
CONFIGFILENAME = 'config_prms.txt'
COMET_PRJ_NAME = 'trn_lrn_curves'
os.makedirs(OUTDIR, exist_ok=True)


# Feature prefix (some already present in the tidy dataframe)
fea_prfx_dict = {'rna': 'cell_rna.', 'cnv': 'cell_cnv.',
                 'dsc': 'drug_dsc.', 'fng': 'drug_fng.',
                 'clb': 'cell_lbl.', 'dlb': 'drug_lbl.'}

np.set_printoptions(precision=3)


def run(args):
    outdir = args['outdir']
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
    model_name = args['model_name']
    cv_method = args['cv_method']
    cv_folds = args['cv_folds']
    lr_curve_ticks = args['lc_ticks']
    n_jobs = args['n_jobs']

    epochs = args['epochs']
    batch_size = args['batch_size']
    dr_rate = args['dr_rate']
    attn = args['attn']

    # Feature list
    feature_list = cell_features + drug_features + other_features

    # Define names
    train_sources_name = '_'.join(train_sources)
    
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
    run_outdir = utils.create_outdir(outdir=outdir, args=args)
    logfilename = run_outdir/'logfile.log'
    lg = Logger(logfilename=logfilename)

    lg.logger.info(f'File path: {file_path}')
    lg.logger.info(f'System CPUs: {psutil.cpu_count(logical=True)}')
    lg.logger.info(f'n_jobs: {n_jobs}')

    # Dump args to file
    utils.dump_args(args, outdir=run_outdir)

    # Create outdir for figs
    # figpath = os.path.join(run_outdir, 'figs')
    # os.makedirs(figpath, exist_ok=True)


    # ========================================================================
    #       Comet
    # ========================================================================
    # www.comet.ml/docs/python-sdk/Experiment
    COMET_API_KEY = os.environ.get('COMET_API_KEY')
    if COMET_API_KEY is not None:
        # args['comet_prj_name'] = COMET_PRJ_NAME
        # args['comet_set_name'] = train_sources_name # model_name
        comet_dict = {'comet_prj_name': COMET_PRJ_NAME,
                      'comet_set_name': train_sources_name}
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
    datapath = DATADIR / DATAFILENAME
    data, te_data = utils_tidy.load_data(datapath=datapath, fea_prfx_dict=fea_prfx_dict,
                                         args=args, logger=lg.logger, random_state=SEED)


    # ========================================================================
    #       Keep a subset of training features
    # ========================================================================
    data = utils_tidy.extract_subset_features(
            data=data,
            feature_list=feature_list,
            fea_prfx_dict=fea_prfx_dict)


    # ========================================================================
    #       Define CV split
    # ========================================================================
    test_size = 0.2
    if mltype == 'cls':
        # Classification
        if cv_method == 'simple':
            if cv_folds == 1:
                cv = ShuffleSplit(n_splits=cv_folds, test_size=test_size, random_state=SEED)
            else:
                cv = KFold(n_splits=cv_folds, shuffle=True, random_state=SEED)
            groups = None

        elif cv_method == 'stratify':
            if cv_folds == 1:
                cv = StratifiedShuffleSplit(n_splits=cv_folds, test_size=test_size, random_state=SEED)
            else:
                cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=SEED)
            groups = None

    elif mltype == 'reg':
        # Regression
        if cv_method == 'group':
            if cv_folds == 1:
                cv = GroupShuffleSplit(n_splits=cv_folds, random_state=SEED)
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
    lg.logger.info(f'Learning curves ... {train_sources}')
    lg.logger.info('='*50)

    # Get data
    xdata, _ = utils_tidy.split_features_and_other_cols(data, fea_prfx_dict=fea_prfx_dict)
    ydata = utils_tidy.extract_target(data=data, target_name=target_name)
    utils_tidy.print_feature_shapes(df=xdata, logger=lg.logger)

    # ML model params
    if model_name == 'lgb_reg':
        init_prms = {'n_jobs': n_jobs, 'random_state': SEED, 'logger': lg.logger}
        fit_prms = {'verbose': False}  # 'early_stopping_rounds': 10,
    elif model_name == 'nn_reg':
        init_prms = {'input_dim': xdata.shape[1], 'dr_rate': dr_rate, 'attn': attn, 'logger': lg.logger}
        fit_prms = {'batch_size': batch_size, 'epochs': epochs, 'verbose': 1, 'validation_split': 0.2} 


    # -----------------------------------------------
    # Generate learning curve - semi automatic method
    # (*) uses cross_validate from sklearn.
    # -----------------------------------------------
    lg.logger.info('\nStart learning curve (my method) ...')

    # Run learning curve
    t0 = time()
    lrn_crv_scores = lrn_crv.my_learning_curve(
        X=xdata,
        Y=ydata,
        mltype=mltype,
        model_name=model_name,
        fit_params=fit_prms,
        init_params=init_prms,
        args=args,
        lr_curve_ticks=lr_curve_ticks,
        data_sizes_frac=None,
        metrics=metrics,
        cv=cv,
        groups=groups,
        n_jobs=n_jobs, random_state=SEED, logger=lg.logger, outdir=run_outdir)
    lg.logger.info('Runtime: {:.3f} mins'.format((time()-t0)/60))

    # Dump results
    lrn_crv_scores.to_csv(run_outdir/'lrn_crv_scores.csv', index=False) 


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
    # (*) Problem: can't generate multiple metrics.
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

    # Kill logger
    lg.kill_logger()
    del data, xdata, ydata
    return lrn_crv_scores
   

def main(args):
    config_fname = file_path / CONFIGFILENAME
    args = argparser.get_args(args=args, config_fname=config_fname)
    pprint(vars(args))
    args = vars(args)
    if args['outdir'] is None:
        args['outdir'] = OUTDIR
    lrn_crv_scores = run(args)
    return lrn_crv_scores, args
    

if __name__ == '__main__':
    # python -m pdb apps/lrn_crv/trn_lrn_crv.py -te ccle gcsi -tr gcsi
    """ __name__ == '__main__' explained:
    www.youtube.com/watch?v=sugvnHA7ElY
    """
    """
    https://stackoverflow.com/questions/14500183/in-python-can-i-call-the-main-of-an-imported-module
    How to run code with input args from another code?
    This will be used with multiple train and test sources.
    For example: in launch_model_transfer.py
        import trn_from_combined.py
        train_from_combined.main([tr_src, tst_src])
    """
    # python -m pdb apps/lrn_crv/trn_lrn_crv.py -te ccle gcsi -tr gcsi
    main(sys.argv[1:])
