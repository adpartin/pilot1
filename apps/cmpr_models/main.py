from __future__ import print_function, division

import warnings
warnings.filterwarnings('ignore')

# from comet_ml import Experiment
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
np.set_printoptions(precision=3)

from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import learning_curve
from sklearn.model_selection import cross_val_score, cross_validate, learning_curve

from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit, KFold
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

from pandas.api.types import is_string_dtype
from sklearn.preprocessing import LabelEncoder

from sklearn.externals import joblib

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
from cvrun import my_cross_validate


# Path
PRJ_NAME = file_path.name
DATADIR = file_path / '../../data/processed/from_combined/tidy_drop_fibro'
OUTDIR = file_path / '../../out/' / PRJ_NAME
DATAFILENAME = 'tidy_data.parquet'
CONFIGFILENAME = 'config_prms.txt'
os.makedirs(OUTDIR, exist_ok=True)


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
    mltype = args['model_name'].split('_')[-1]
    #assert mltype in ['reg', 'cls'], "mltype should be either 'reg' or 'cls'."    
    if mltype not in ['reg', 'cls']:
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
    run_outdir = utils.create_outdir(outdir, args=args)
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
    datapath = DATADIR / DATAFILENAME

    dataset = load_tidy_combined(
            datapath, fea_list=fea_list, logger=lg.logger, random_state=SEED)

    data = get_data_by_src(
            dataset, src_names=tr_sources, logger=lg.logger)
    
    #xdata, ydata, meta, scaler = break_src_data(
    #        data, target=args['target_name'],
    #        scaler_method=args['scaler'], logger=lg.logger)


    # ========================================================================
    #       Train-test split
    # ========================================================================
    te_split_method = 'group'
    te_size = 0.2
    te_splitter = cv_splitter(cv_method=te_split_method, cv_folds=1, test_size=te_size,
                     mltype=mltype, shuffle=True, random_state=SEED)
    if te_split_method=='simple':
        te_groups = None
    elif te_split_method=='group':
        te_groups = data['CELL'].copy()
    
    if is_string_dtype(te_groups):
        grp_enc = LabelEncoder()
        te_groups = grp_enc.fit_transform(te_groups)
    
    tr_id, te_id = next(te_splitter.split(data, groups=te_groups))
    tr_data = data.iloc[tr_id, :]  
    te_data = data.iloc[te_id, :] 

    xdata, ydata, meta, scaler = break_src_data(
            tr_data, target=args['target_name'],
            scaler=args['scaler'], logger=lg.logger)
    joblib.dump(scaler, run_outdir/'scaler.pkl')


    # ========================================================================
    #       Define CV split
    # ========================================================================
    cv = cv_splitter(cv_method=cv_method, cv_folds=cv_folds, test_size=0.2,
                     mltype=mltype, shuffle=True, random_state=SEED)
    if cv_method=='simple':
        cv_groups = None
    elif cv_method=='group':
        cv_groups = tr_data['CELL'].copy()


    # ========================================================================
    #       CV training
    # ========================================================================
    lg.logger.info('\n{}'.format('='*50))
    lg.logger.info(f'CV training ... {tr_sources}')
    lg.logger.info('='*50)

    # ML model params
    if model_name == 'lgb_reg':
        init_prms = {'n_jobs': n_jobs, 'random_state': SEED, 'logger': lg.logger}
        fit_prms = {'verbose': False}  # 'early_stopping_rounds': 10, 'sample_weight': sample_weight
    elif model_name == 'nn_reg':
        init_prms = {'input_dim': xdata.shape[1], 'dr_rate': dr_rate, 'opt_name': opt_name, 'attn': attn, 'logger': lg.logger}
        fit_prms = {'batch_size': batch_size, 'epochs': epochs, 'verbose': 1}  # 'validation_split': 0.1
    elif model_name == 'nn_model1' or 'nn_model2':
        init_prms = {'input_dim': xdata.shape[1], 'dr_rate': dr_rate, 'opt_name': opt_name, 'logger': lg.logger}
        fit_prms = {'batch_size': batch_size, 'epochs': epochs, 'verbose': 1}  # 'validation_split': 0.1

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
        groups=cv_groups,
        n_jobs=n_jobs, random_state=SEED, logger=lg.logger, outdir=run_outdir)
    lg.logger.info('Runtime: {:.1f} mins'.format( (time()-t0)/60) )
    
    # Dump results
    # cv_scores = cv_scores.round(3)
    cv_scores.to_csv( run_outdir/('cv_scores_' + tr_sources_name + '.csv'), index=False )
    lg.logger.info(f'\ncv_scores\n{cv_scores}')


    # ========================================================================
    #       Predict on test set
    # ========================================================================
    xte, yte, meta, _ = break_src_data(
            te_data, target=args['target_name'],
            scaler=scaler, logger=lg.logger)

    yte_pred = best_model.model.predict(xte)
    res = np.hstack( (yte.values.reshape(-1,1), yte_pred.reshape(-1,1)) )
    res = pd.DataFrame(res, columns=['true', 'pred'])
    res.to_csv(run_outdir/'preds.csv', index=False)


    # ========================================================================
    #       Create correlation plot
    # ========================================================================


    # Kill logger
    lg.kill_logger()
    del tr_data, te_data, xdata, ydata, xte, yte
    return csv_all


def main(args):
    config_fname = file_path / CONFIGFILENAME
    args = argparser.get_args(args=args, config_fname=config_fname)
    pprint(vars(args))
    args = vars(args)
    if args['outdir'] is None:
        args['outdir'] = OUTDIR
    csv_scores_all = run(args)
    return csv_scores_all, args
    

if __name__ == '__main__':
    # python -m pdb apps/csv/trn_from_combined.py -te cclediggcsi -tr gcsi
    main(sys.argv[1:])
