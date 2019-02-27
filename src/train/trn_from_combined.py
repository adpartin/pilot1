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
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import learning_curve
from sklearn.model_selection import cross_val_score, cross_validate, learning_curve

from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit, KFold
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

# Get file path
# ... manutally run ...
# file_path = os.getcwd()
# file_path = os.path.join(file_path, 'src/models')
# os.chdir(file_path)
# ... auto ...
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
from cvrun import my_cross_validate
from cvsplitter import GroupSplit, SimpleSplit, plot_ytr_yvl_dist

DATADIR = os.path.join(file_path, '../../data/processed/from_combined')
OUTDIR = os.path.join(file_path, '../../models/from_combined')
DATAFILENAME = 'tidy_data_no_fibro.parquet'
# DATAFILENAME = 'tidy_data.parquet'
CONFIGFILENAME = 'config_prms.txt'
os.makedirs(OUTDIR, exist_ok=True)

SEED = None

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
    retrain = args['retrain']
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
    # https://scikit-learn.org/stable/modules/model_evaluation.html#scoring
    def reg_auroc(y_true, y_pred):
        y_true = np.where(y_true < 0.5, 1, 0)
        y_score = np.where(y_pred < 0.5, 1, 0)
        auroc = sklearn.metrics.roc_auc_score(y_true, y_score)
        return auroc
    reg_auroc_score = sklearn.metrics.make_scorer(score_func=reg_auroc, greater_is_better=True)

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

    # Dump args to file
    utils.dump_args(args, outdir=run_outdir)

    # Create outdir for figs
    figpath = os.path.join(run_outdir, 'figs')
    os.makedirs(figpath, exist_ok=True)


    # ========================================================================
    #       Load data and pre-proc
    # ========================================================================
    datapath = os.path.join(DATADIR, DATAFILENAME)
    data, te_data = utils_tidy.load_data(datapath=datapath, fea_prfx_dict=fea_prfx_dict,
                                         args=args, logger=lg.logger, random_state=SEED)


    # ========================================================================
    #       Save plots
    # ========================================================================
    utils.boxplot_rsp_per_drug(df=data, target_name=target_name,
        path=os.path.join(figpath, f'{target_name}_per_drug_boxplot.png'))

    utils.plot_hist(x=data[target_name], var_name=target_name,
        path=os.path.join(figpath, target_name+'_hist.png'))
    
    utils.plot_qq(x=data[target_name], var_name=target_name,
        path=os.path.join(figpath, target_name+'_qqplot.png'))
    
    utils.plot_hist_drugs(x=data['DRUG'], path=os.path.join(figpath, 'drugs_hist.png'))


    # ========================================================================
    #       TODO: outlier removal (move this to data preparation step)
    # ========================================================================
    pass


    # ========================================================================
    #       Keep a subset of training features
    # ========================================================================
    data = utils_tidy.extract_subset_features(data=data, feature_list=feature_list, fea_prfx_dict=fea_prfx_dict)
    # data = utils_tidy.impute_values(data=data, fea_prfx_dict=fea_prfx_dict, logger=lg.logger)


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
    #       ML Model
    # ========================================================================
    # # ML model params
    # if model_name == 'lgb_reg':
    #     init_prms = {'n_jobs': n_jobs, 'random_state': SEED, 'logger': lg.logger}
    #     fit_prms = {'verbose': False}  # 'early_stopping_rounds': 10, 'sample_weight': sample_weight
    # elif model_name == 'nn_reg':
    #     xdata, _ = utils_tidy.split_features_and_other_cols(data, fea_prfx_dict=fea_prfx_dict)
    #     init_prms = {'input_dim': xdata.shape[1], 'dr_rate': dr_rate, 'attn': attn, 'logger': lg.logger}
    #     fit_prms = {'batch_size': batch_size, 'epochs': epochs, 'verbose': 1}  # 'validation_split': 0.1


    # ========================================================================
    #       CV training
    # ========================================================================
    lg.logger.info('\n{}'.format('='*50))
    lg.logger.info(f'CV training ... {train_sources}')
    lg.logger.info('='*50)

    # Get data
    xdata, _ = utils_tidy.split_features_and_other_cols(data, fea_prfx_dict=fea_prfx_dict)
    ydata = utils_tidy.extract_target(data=data, target_name=target_name)

    # ML model params
    if model_name == 'lgb_reg':
        init_prms = {'n_jobs': n_jobs, 'random_state': SEED, 'logger': lg.logger}
        fit_prms = {'verbose': False}  # 'early_stopping_rounds': 10, 'sample_weight': sample_weight
    elif model_name == 'nn_reg':
        init_prms = {'input_dim': xdata.shape[1], 'dr_rate': dr_rate, 'attn': attn, 'logger': lg.logger}
        fit_prms = {'batch_size': batch_size, 'epochs': epochs, 'verbose': 1}  # 'validation_split': 0.1


    # # -----------------
    # # sklearn CV method - (doesn't work with keras models)
    # # ----------------- 
    # # Define ML model
    # model = ml_models.get_model(model_name=model_name, init_params=init_prms)  

    # # Run CV
    # t0 = time.time()
    # cv_scores = cross_validate(
    #     estimator=model.model,
    #     X=xdata, y=ydata,
    #     scoring=metrics,
    #     cv=cv, groups=groups,
    #     n_jobs=n_jobs, fit_params=fit_prms)
    # lg.logger.info('Runtime: {:.3f} mins'.format((time.time()-t0)/60))

    # # Dump results
    # cv_scores = utils.update_cross_validate_scores( cv_scores )
    # cv_scores = cv_scores.reset_index(drop=True)
    # cv_scores.insert( loc=cv_scores.shape[1]-cv_folds, column='mean', value=cv_scores.iloc[:, -cv_folds:].values.mean(axis=1) )
    # cv_scores.insert( loc=cv_scores.shape[1]-cv_folds, column='std',  value=cv_scores.iloc[:, -cv_folds:].values.std(axis=1) )
    # cv_scores = cv_scores.round(3)
    # cv_scores.to_csv(os.path.join(run_outdir, 'cv_scores_' + train_sources_name + '.csv'), index=False)
    # lg.logger.info(f'cv_scores\n{cv_scores}')


    # ------------
    # My CV method - (works with keras models)
    # ------------
    t0 = time.time()
    cv_scores, best_model = my_cross_validate(
        X=xdata, Y=ydata,
        mltype=mltype,
        model_name=model_name,
        fit_params=fit_prms,
        init_params=init_prms,
        args=args,
        cv=cv,
        groups=groups,
        n_jobs=n_jobs, random_state=SEED, logger=lg.logger, outdir=run_outdir)
    lg.logger.info('Runtime: {:.3f} mins'.format((time.time()-t0)/60))
    
    # Dump results
    # cv_scores = cv_scores.round(3)
    cv_scores.to_csv(os.path.join(run_outdir, 'cv_scores_' + train_sources_name + '.csv'), index=False)
    lg.logger.info(f'cv_scores\n{cv_scores}')


    # ========================================================================
    #       Train final model (entire dataset)
    # ========================================================================
    if retrain:
        lg.logger.info('\n{}'.format('='*50))
        lg.logger.info(f'Train final model (use entire dataset) ... {train_sources}')
        lg.logger.info('='*50)

        # Get the data
        xdata, _ = utils_tidy.split_features_and_other_cols(data, fea_prfx_dict=fea_prfx_dict)
        ydata = utils_tidy.extract_target(data=data, target_name=target_name)
        utils_tidy.print_feature_shapes(df=xdata, logger=lg.logger)

        # Define sample weight
        # From lightgbm docs: n_samples / (n_classes * np.bincount(y))
        # thres_target = 0.5
        # a = np.where(ydata.values < thres_target, 0, 1)
        # wgt = len(a) / (2 * np.bincount(a))
        # sample_weight = np.array([wgt[0] if v < 0.5 else wgt[1] for v in a])

        # ML model params
        if model_name == 'lgb_reg':
            # init_prms = {'n_jobs': n_jobs, 'random_state': SEED, 'logger': lg.logger}
            # fit_prms = {'verbose': False}  # 'early_stopping_rounds': 10, 'sample_weight': sample_weight

            # Use early stopping
            init_prms = {'n_jobs': n_jobs, 'random_state': SEED, 'n_estimators': 2000, 'logger': lg.logger}
            X_train, X_test, y_train, y_test = train_test_split(xdata, ydata, test_size=0.2, random_state=SEED)
            xdata, ydata = X_train, y_train
            eval_set = (X_test, y_test)
            fit_prms = {'verbose': False, 'eval_set': eval_set, 'early_stopping_rounds': 10}  # 'sample_weight': sample_weight
            
        elif model_name == 'nn_reg':
            val_split = 0.1
            xdata, _ = utils_tidy.split_features_and_other_cols(data, fea_prfx_dict=fea_prfx_dict)
            init_prms = {'input_dim': xdata.shape[1], 'dr_rate': dr_rate, 'attn': attn, 'logger': lg.logger}
            fit_prms = {'batch_size': batch_size, 'epochs': epochs, 'verbose': 1, 'validation_split': val_split}
            args['final_model_val_split'] = val_split

        # Define ML model
        model_final = ml_models.get_model(model_name=model_name, init_params=init_prms)   

        # Train
        t0 = time.time()
        model_final.model.fit(xdata, ydata, **fit_prms) # TODO: there is a problem here with fit_prms (it seems like it reuses them from my_cross_validate)
        lg.logger.info('Runtime: {:.3f} mins'.format((time.time()-t0)/60))

        # # Save model
        # model_final.save_model(outdir=run_outdir)
    
    else:
        model_final = best_model

    # Dump model
    model_final.dump_model(outdir=run_outdir)

    # Save network figure
    if 'nn' in model_name:
        from keras.utils import plot_model
        plot_model(model_final.model, to_file=os.path.join(figpath, 'nn_model.png'))



    # ========================================================================
    #       Infer
    # ========================================================================
    lg.logger.info('\n{}'.format('='*50))
    lg.logger.info(f'Inference ... {test_sources}')
    lg.logger.info('='*50)

    csv = []  # cross-study-validation scores
    for i, te_src in enumerate(test_sources):
        lg.logger.info(f'\nTest source {i+1}:  _____ {te_src} _____')
        t0 = time.time()

        if train_sources == [te_src]:
            lg.logger.info("That's the train set (take preds preds from cv run).")
            continue

        te_src_data = te_data[te_data['SOURCE'].isin([te_src])].reset_index(drop=True)
        if te_src_data.shape[0] == 0:
            continue  # continue if there are no data samples available for this source

        # Plot dist of response
        utils.plot_hist(x=te_src_data[target_name], var_name=target_name,
                        path=os.path.join(figpath, target_name + '_hist_' + te_src + '.png'))

        # Prep test data for preds
        xte, _ = utils_tidy.split_features_and_other_cols(te_src_data, fea_prfx_dict=fea_prfx_dict)
        yte = utils_tidy.extract_target(data=te_src_data, target_name=target_name)
        utils_tidy.print_feature_shapes(df=xte, logger=lg.logger)

        # Calc scores
        #scores = model_final.calc_scores(xdata=xte, ydata=yte, to_print=True)
        y_preds, y_true = utils.calc_preds(estimator=model_final.model, xdata=xte, ydata=yte, mltype=mltype)
        scores = utils.calc_scores(y_true=y_true, y_preds=y_preds, mltype=mltype)
        csv.append( pd.DataFrame([scores], index=[te_src]).T )
        
        # Dump preds --> TODO: error when use keras
        # preds_fname = 'preds_' + src + '_' + model_name + '.csv'
        # model_final.dump_preds(df_data=te_src_data, xdata=xte, target_name=target_name,
        #                        outpath=os.path.join(run_outdir, preds_fname))                 

        lg.logger.info('\nRuntime: {:.3f}'.format((time.time()-t0)/60))

    # Combine test set preds
    if len(csv) > 0:
        csv = pd.concat(csv, axis=1)

    # Adjust cv_scores in order to combine with test set preds
    # (take the mean cv score for val set)
    cv_scores = cv_scores[cv_scores['tr_set']==False].drop(columns='tr_set')
    cv_scores[train_sources_name] = cv_scores.iloc[:, -cv_folds:].mean(axis=1)
    cv_scores = cv_scores[['metric', train_sources_name]]
    cv_scores = cv_scores.set_index('metric')
    cv_scores.index.name = None

    # Combine scores from val set cross-validation and test set
    csv_all = pd.concat([cv_scores, csv], axis=1)
    csv_all.insert(loc=0, column='train_src', value=train_sources_name)
    csv_all = csv_all.reset_index().rename(columns={'index': 'metric'})
    # csv_all = csv_all.round(decimals=3)

    lg.logger.info('\ncsv_scores\n{}'.format(csv_all))
    csv_all.to_csv(os.path.join(run_outdir, 'csv_scores_' + train_sources_name + '.csv'), index=False)

    # Kill logger
    lg.kill_logger()
    del data, xdata, ydata, model_final
    return csv_all


def main(args):
    config_fname = os.path.join(file_path, CONFIGFILENAME)
    args = argparser.get_args(args=args, config_fname=config_fname)
    pprint(vars(args))
    args = vars(args)
    if args['outdir'] is None:
        args['outdir'] = OUTDIR
    csv_scores_all = run(args)
    return csv_scores_all, args
    

if __name__ == '__main__':
    # python -m pdb src/models/train_from_combined.py -te ccle gcsi -tr gcsi
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
    # python -m pdb src/models/train_from_combined.py -te ccle gcsi -tr gcsi
    main(sys.argv[1:])
