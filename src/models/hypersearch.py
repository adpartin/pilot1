"""
hyperopt:
https://towardsdatascience.com/automated-machine-learning-hyperparameter-tuning-in-python-dfda59b72f8a

Comparison:
https://medium.com/@ramrajchandradevan/comparison-among-hyper-parameter-optimizers-cd37483cd47
"""
from __future__ import print_function
from __future__ import division

import warnings
warnings.filterwarnings('ignore')

import os
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
from sklearn.model_selection import learning_curve
from sklearn.model_selection import cross_val_score, cross_validate, learning_curve
from sklearn.model_selection import ShuffleSplit, GroupShuffleSplit, KFold, GroupKFold

# Utils
# file_path = os.getcwd()
# file_path = os.path.join(file_path, 'src/models')
# os.chdir(file_path)

file_path = os.path.dirname(os.path.realpath(__file__))  # os.path.dirname(os.path.abspath(__file__))
# utils_path = os.path.abspath(os.path.join(file_path, 'utils'))
# sys.path.append(utils_path)
import utils
import utils_tidy
import argparser
import classlogger
import lrn_curve
from cvsplitter import GroupSplit, SimpleSplit, plot_ytr_yvl_dist

DATADIR = os.path.join(file_path, '../../data/processed/from_combined')
OUTDIR = os.path.join(file_path, '../../models/from_combined/hyperprms')
DATAFILENAME = 'tidy_data_no_fibro.parquet'
# DATAFILENAME = 'tidy_data.parquet'
CONFIGFILENAME = 'hypersearch.txt'
os.makedirs(OUTDIR, exist_ok=True)

SEED = 0

# Feature prefix (some already present in the tidy dataframe)
fea_prfx_dict = {'rna': 'cell_rna.',
                 'cnv': 'cell_cnv.',
                 'dsc': 'drug_dsc.',
                 'fng': 'drug_fng.',
                 'clb': 'cell_lbl.',
                 'dlb': 'drug_lbl.'}

np.set_printoptions(precision=3)


def run(args):
    target_name = args['target_name']
    target_trasform = args['target_trasform']    
    train_sources = args['train_sources']
    test_sources = args['test_sources']
    row_sample = args['row_sample']
    col_sample = args['col_sample']
    tissue_type = args['tissue_type']
    cell_features = args['cell_features']
    drug_features = args['drug_features']
    other_features = args['other_features']
    model_name = args['ml_models']
    cv_method = args['cv_method']
    cv_folds = args['cv_folds']
    lr_curve_ticks = args['lc_ticks']
    verbose = args['verbose']
    n_jobs = args['n_jobs']

    # Feature list
    feature_list = cell_features + drug_features + other_features


    # # Save best model
    # best_score = 0
    # best_params = None
    # best_model = None
    # os.makedirs(OUTDIR, 'best_model')


    # datasets = [ ['gcsi'], ['ccle'], ['ctrp'], ['gdsc'] ]
    # for dname in datasets:
    for dname in args['train_sources']:
        args['train_sources'] = [dname]
        args['test_sources'] = [dname]


        # ========================================================================
        #       Logger
        # ========================================================================
        t = datetime.datetime.now()
        t = [t.year, '-', t.month, '-', t.day, '_', 'h', t.hour, '-', 'm', t.minute]
        t = ''.join([str(i) for i in t])
        name_sufix = '.'.join(train_sources + [model_name] + [cv_method] + cell_features + drug_features + [target_name])
        run_outdir = os.path.join(OUTDIR, name_sufix + '~' + t)
        os.makedirs(run_outdir)
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
        #       Prepare data
        # ========================================================================
        # Prepare data
        xdata, _ = utils_tidy.split_features_and_other_cols(data, fea_prfx_dict=fea_prfx_dict)
        ydata = utils_tidy.extract_target(data=data, target_name=target_name)
        utils_tidy.print_feature_shapes(df=xdata, logger=lg.logger)

        # Split tr/vl data
        if cv_method=='simple':
            cv = KFold(n_splits=cv_folds, shuffle=False, random_state=SEED)
            groups = None
        elif cv_method=='group':
            cv = GroupKFold(n_splits=cv_folds)
            groups = data['CELL']
        elif cv_method=='stratify':
            pass
        else:
            raise ValueError('This cv_method ({}) is not supported'.format(cv_method))

        """
        # ========================================================================
        #       Hyper-param search
        # ========================================================================
        # https://www.kaggle.com/mlisovyi/lightgbm-hyperparameter-optimisation-lb-0-761
        # https://www.kaggle.com/garethjns/microsoft-lightgbm-with-parameter-tuning-0-823
        from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
        import lightgbm as lgb
        # from ml_models import LGBM_REGRESSOR, RF_REGRESSOR

        lg.logger.info('\n\n{}'.format('-'*50))
        lg.logger.info('Hyper-parameter search ... {}'.format('_'.join(args['train_sources'])))
        lg.logger.info('-'*50)

        n_estimators = 100

        from scipy.stats import randint, uniform
        lgb_reg_params = {'objective': 'regression',
                        'n_jobs': n_jobs,
                        'n_estimators': n_estimators,
                        'random_state': SEED,
                        }

        fit_params = {'early_stopping_rounds': 10,
                    'verbose': False,
                    'sample_weight': None}

        # param_test = {'num_leaves': randint(6, 50),  # (31) alias: max_leaves
        #               'max_depth': [-1], # (-1 -> not limit)
        #               'min_child_samples': randint(100, 500),  # (20) alias: min_data_in_leaf
        #               'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4], # (1e-3)
        #               'subsample': uniform(loc=0.2, scale=0.8), # (1.0) alias: bagging_fraction, 
        #               'colsample_bytree': uniform(loc=0.4, scale=0.6), # (1.0) alias: feature_fraction, 
        #               'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100], # (0.0) alias: lambda_l2
        #               'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100], # (0.0) alias: lambda_l1
        #               'learning_rate': [0.1]} # (0.1) alias: shrinkage_rate, eta

        # prm_grid_srch = {'num_leaves': [31, 45],  # (31) alias: max_leaves
        #               'max_depth': [-1, 7], # (-1 -> not limit)
        #               'learning_rate': [0.1, 0.01], # (0.1) alias: shrinkage_rate, eta
        #               'subsample': [1.0, 0.5], # (1.0) alias: bagging_fraction, 
        #               'colsample_bytree': [1.0, 0.5], # (1.0) alias: feature_fraction, 
        #               #'min_child_weight': [0.001, 0.1], # (0.001)
        #               'min_child_samples': [20, 0, 40], # (20) alias: min_data_in_leaf  -->  very important (note from Kaggle course)!!
        #               #'reg_alpha': [0], # (0.0) alias: lambda_l2
        #               #'reg_lambda': [0], # (0.0) alias: lambda_l1
        # }

        # prm_rndm_srch = {

        # }

        prm_grid_srch = {'learning_rate': [0.1, 0.01], # (0.1) alias: shrinkage_rate, eta
                         'min_child_samples': [20, 40], # (20) alias: min_data_in_leaf  -->  very important (note from Kaggle course)!!
        }

        # model = LGBM_REGRESSOR(n_jobs=n_jobs, random_state=SEED, logger=lg.ogger)
        model = lgb.LGBMModel(**lgb_reg_params)
        print(model)

        gs = GridSearchCV(estimator=model,
                        param_grid=prm_grid_srch,
                        scoring='neg_mean_absolute_error',  # r2 # sorted(sklearn.metrics.SCORERS.keys())
                        n_jobs=n_jobs,
                        cv=cv,
                        refit=True,
                        verbose=1,
                        #fit_params=fit_params,
                        )

        lg.logger.info('\nStart grid search ...')
        t0 = time.time()
        gs.fit(X=xdata, y=ydata,
            #fit_params=fit_params,  # why this doesn't work??
            groups=groups)
        lg.logger.info('Runtime: {:.3f} mins'.format((time.time()-t0)/60))

        # Save resutls
        rf_reg_hypsearch = pd.DataFrame(gs.cv_results_).T
        rf_reg_hypsearch.to_csv(os.path.join(run_outdir, model_name+'_hps_summary.csv'))  # save hyperparam search results

        lg.logger.info(f'\n{model_name} best score (random search): {gs.best_score_:.3f}')
        lg.logger.info('{} best params (random search): \n{}\n'.format(model_name, gs.best_params_))

        # Dump best params into file
        fname = 'best_params_' + '_'.join(dname) + '_' + cv_method + '.txt'
        fname = os.path.join(run_outdir, fname)
        with open(fname, 'w') as file:
            file.write(f'[{model_name}]\n')
            for k, v in gs.best_params_.items():
                file.write('{} = {}\n'.format(k, v))

        lg.kill_logger()
        del data, xdata, ydata, model

        # # Update best score
        # if gs.best_score_ > best_score:
        #     best_score = gs.best_score_
        #     best_params = gs.gs.best_params_
        #     best_model = gs.best_mo
        
        """

        # ====================================================================================
        import lightgbm as lgb
        from hyperopt import hp
        from hyperopt import tpe
        from hyperopt import fmin
        from hyperopt import Trials
        from hyperopt import STATUS_OK

        MAX_EVALS = 10
        N_FOLDS = 10

        # Convert response to binary
        ydata = np.where(ydata < 0.5, 1, 0)

        # Create the dataset
        train_features = xdata
        train_labels = ydata
        train_set = lgb.Dataset(data=train_features, label=train_labels)

        def objective(params, n_folds = N_FOLDS):
            """ Objective function for Gradient Boosting Machine Hyperparameter Tuning. """
            
            # Perform n_fold cross validation with hyperparameters
            # Use early stopping and evalute based on ROC AUC
            cv_results = lgb.cv(params, train_set, nfold=n_folds, num_boost_round=10000, 
                                early_stopping_rounds=100, metrics='auc', seed=SEED)
        
            # Extract the best score
            best_score = max(cv_results['auc-mean'])
            
            # Loss must be minimized
            loss = 1 - best_score
            
            # Dictionary with information for evaluation
            ret = {'loss': loss, 'params': params, 'status': STATUS_OK}
            return ret

        # Define the search space
        # space = {
        #     'class_weight': hp.choice('class_weight', [None, 'balanced']),
        #     'boosting_type': hp.choice('boosting_type', 
        #                             [{'boosting_type': 'gbdt', 
        #                                'subsample': hp.uniform('gdbt_subsample', 0.5, 1)}, 
        #                               {'boosting_type': 'dart', 
        #                                'subsample': hp.uniform('dart_subsample', 0.5, 1)},
        #                               {'boosting_type': 'goss', 'subsample': 1.0}]),
        #     'num_leaves': hp.quniform('num_leaves', 30, 150, 1),
        #     'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),
        #     'subsample_for_bin': hp.quniform('subsample_for_bin', 20000, 300000, 20000),
        #     'min_child_samples': hp.quniform('min_child_samples', 20, 500, 5),
        #     'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
        #     'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
        #     'colsample_bytree': hp.uniform('colsample_by_tree', 0.6, 1.0)
        # }
        space = {
            'class_weight': hp.choice('class_weight', [None, 'balanced']),
            'num_leaves': hp.quniform(label='num_leaves', low=30, high=150, q=1),
            'learning_rate': hp.loguniform(label='learning_rate', low=np.log(0.01), high=np.log(0.2)),
            'subsample_for_bin': hp.quniform(label='subsample_for_bin', low=20000, high=300000, q=20000),
            'min_child_samples': hp.quniform(label='min_child_samples', low=20, high=500, q=5),
            'reg_alpha': hp.uniform(label='reg_alpha', low=0.0, high=1.0),
            'reg_lambda': hp.uniform(label='reg_lambda', low=0.0, high=1.0),
            'colsample_bytree': hp.uniform(     'colsample_by_tree', 0.6, 1.0)
        }
        

        # Algorithm
        tpe_algorithm = tpe.suggest

        # Trials object to track progress
        bayes_trials = Trials()

        # File to save first results
        import csv
        out_file = 'gbm_trials.csv'
        of_connection = open(out_file, 'w')
        writer = csv.writer(of_connection)

        # Write the headers to the file
        writer.writerow(['loss', 'params', 'iteration', 'estimators', 'train_time'])
        of_connection.close()

        # Optimize
        best = fmin(fn = objective, space = space, algo = tpe.suggest, 
                    max_evals = MAX_EVALS, trials = bayes_trials)
        # ====================================================================================

    print('Done')


def main(args):
    # config_fname = os.path.join(file_path, CONFIGFILENAME)
    config_fname = None
    args = argparser.get_args(args=args, config_fname=config_fname)
    pprint(vars(args))
    args = vars(args)
    run(args)


if __name__ == '__main__':
    main(sys.argv[1:])