from __future__ import print_function
from __future__ import division

import warnings
warnings.filterwarnings('ignore')

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = 2

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
from sklearn.model_selection import ShuffleSplit, GroupShuffleSplit, KFold, GroupKFold, StratifiedKFold

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
    target_trasform = args['target_trasform']    
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

    # Plots
    utils.boxplot_rsp_per_drug(df=data, target_name=target_name,
        path=os.path.join(run_outdir, f'{target_name}_per_drug_boxplot.png'))
    utils.plot_hist(x=data[target_name], var_name=target_name,
        path=os.path.join(run_outdir, target_name+'_hist.png'))
    utils.plot_qq(x=data[target_name], var_name=target_name,
        path=os.path.join(run_outdir, target_name+'_qqplot.png'))
    utils.plot_hist_drugs(x=data['DRUG'], path=os.path.join(run_outdir, 'drugs_hist.png'))



    # ========================================================================
    #       TODO: outlier removal
    # ========================================================================
    pass



    # ========================================================================
    #       Keep a subset of training features
    # ========================================================================
    data = utils_tidy.extract_subset_features(data=data, feature_list=feature_list, fea_prfx_dict=fea_prfx_dict)
    # data = utils_tidy.impute_values(data=data, fea_prfx_dict=fea_prfx_dict, logger=lg.logger)



    # ========================================================================
    #       Initialize ML model
    # ========================================================================
    # from ml_models import LGBM_REGRESSOR, RF_REGRESSOR
    # def init_model(mlmodel, logger, verbose=False):
    #     if 'lgb_reg' in mlmodel:
    #         if verbose:
    #             logger.info('ML Model: lgb regressor')
    #         model = LGBM_REGRESSOR(n_jobs=n_jobs, random_state=SEED, logger=logger)
    #         fit_params = {'verbose': False,
    #                       #'early_stopping_rounds': 10,
    #         }
    #     elif 'rf_reg' in mlmodel:
    #         if verbose:
    #             logger.info('ML Model: rf regressor')
    #         model = RF_REGRESSOR(n_jobs=4, random_state=SEED, logger=logger)
    #         fit_params = {'verbose': False, 'n_jobs': n_jobs, 'random_state': SEED}
    #     return model, fit_params



    # ========================================================================
    #       Define CV split
    # ========================================================================
    # Split tr/vl data
    if cv_method=='simple':
        cv = KFold(n_splits=cv_folds, shuffle=False, random_state=SEED)
        groups = None
    elif cv_method=='group':
        cv = GroupKFold(n_splits=cv_folds)
        groups = data['CELL'].copy()
    elif cv_method=='stratify':
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=SEED)
        groups = None
    else:
        raise ValueError(f'This cv_method ({cv_method}) is not supported')



    # ========================================================================
    #       CV training
    # ========================================================================
    lg.logger.info('\n\n{}'.format('='*50))
    lg.logger.info('CV training ...')
    lg.logger.info('='*50)

    # -----------------
    # sklearn CV method
    # -----------------
    # Get data
    xdata, _ = utils_tidy.split_features_and_other_cols(data, fea_prfx_dict=fea_prfx_dict)
    ydata = utils_tidy.extract_target(data=data, target_name=target_name)

    # ML model params
    if mlmodel == 'lgb_reg':
        init_params = {'n_jobs': n_jobs, 'random_state': SEED, 'logger': lg.logger}
        fit_params = {'verbose': False}  # 'early_stopping_rounds': 10,
    elif mlmodel == 'nn_reg':
        init_params = {'input_dim': xdata.shape[1], 'dr_rate': dr_rate, 'attn': attn, 'logger': lg.logger}
        fit_params = {'batch_size': batch_size, 'epochs': epochs, 'verbose': 2, 'validation_split': 0.2}

    # Define ML model
    model = ml_models.get_model(mlmodel=mlmodel, init_params=init_params)  

    # Run CV
    t0 = time.time()
    cv_scores = cross_validate(
        estimator=model.model, # sklearn.base.clone(model.model),
        X=xdata, y=ydata,
        scoring=metrics,
        cv=cv, groups=groups,
        n_jobs=n_jobs, fit_params=fit_params)
    lg.logger.info('Runtime: {:.3f} mins'.format((time.time()-t0)/60))

    # Dump results
    cv_scores = utils.update_cross_validate_scores(cv_scores)
    cv_scores = cv_scores.reset_index(drop=True)
    cv_scores.insert( loc=cv_scores.shape[1]-cv_folds, column='mean', value=cv_scores.iloc[:, -cv_folds:].values.mean(axis=1) )
    cv_scores.insert( loc=cv_scores.shape[1]-cv_folds, column='std',  value=cv_scores.iloc[:, -cv_folds:].values.std(axis=1) )
    cv_scores = cv_scores.round(3)
    cv_scores.to_csv(os.path.join(run_outdir, 'cv_scores_' + train_sources_name + '.csv'), index=False)
    lg.logger.info(f'cv_scores\n{cv_scores}')


    # # ------------
    # # My CV method
    # # ------------
    # # from cvrun import my_cv_run
    # # model, _ = init_model(mlmodel, logger=lg.logger)
    # # tr_cv_scores, vl_cv_scores = my_cv_run(
    # #     data=data,
    # #     target_name=target_name,
    # #     model=model.model,
    # #     #metrics=metrics,  # TODO: implement this option
    # #     fea_prfx_dict=fea_prfx_dict,
    # #     cv_method=cv_method, cv_folds=cv_folds,
    # #     logger=lg.logger, verbose=True, random_state=SEED, outdir=run_outdir)



    # ========================================================================
    #       Train final model (entire dataset)
    # ========================================================================
    lg.logger.info('\n{}'.format('='*50))
    lg.logger.info(f'Train final model (use entire dataset) ... {train_sources}')
    lg.logger.info('='*50)

    # Get the data
    xdata, _ = utils_tidy.split_features_and_other_cols(data, fea_prfx_dict=fea_prfx_dict)
    ydata = utils_tidy.extract_target(data=data, target_name=target_name)
    utils_tidy.print_feature_shapes(df=xdata, logger=lg.logger)

    # Define sample weight
    # From lightgbm docs: n_samples / (n_classes * np.bincount(y))
    th_target = 0.5
    a = np.where(ydata.values < th_target, 0, 1)
    wgt = len(a) / (2 * np.bincount(a))
    sample_weight = np.array([wgt[0] if v < 0.5 else wgt[1] for v in a])

    # Define and train ML model
    # model_final, _ = init_model(mlmodel, lg.logger)
    # t0 = time.time()
    # if 'lgb_reg' in mlmodel:
    #     # model_final.fit(xdata, ydata, eval_set=[(xdata, ydata)])  # use my class fit method
    #     params = {'verbose': False, 'sample_weight': sample_weight}
    #     model_final.fit(xdata, ydata, **params)  # use my class fit method
    #     # model_final.model.fit(xdata, ydata, **params)
    #     # model_final.model.fit(xdata, ydata, eval_set=[(xdata, ydata)]) # use lightgbm fit method
    # else:
    #     model_final.fit(xdata, ydata)
    # lg.logger.info('Runtime: {:.3f} mins'.format((time.time()-t0)/60))

    # ML model params
    if mlmodel == 'lgb_reg':
        init_params = {'n_jobs': n_jobs, 'random_state': SEED, 'logger': lg.logger}
        fit_params = {'verbose': False}  # 'early_stopping_rounds': 10,
    elif mlmodel == 'nn_reg':
        init_params = {'input_dim': xdata.shape[1], 'dr_rate': dr_rate, 'attn': attn, 'logger': lg.logger}
        fit_params = {'batch_size': batch_size, 'epochs': epochs, 'verbose': 2, 'validation_split': 0.2}

    # Define ML model
    model_final = ml_models.get_model(mlmodel=mlmodel, init_params=init_params)  

    t0 = time.time()
    #fit_params = {'verbose': False, 'sample_weight': sample_weight}  # 'early_stopping_rounds': 10,
    model_final.model.fit(xdata, ydata, **fit_params)
    lg.logger.info('Runtime: {:.3f} mins'.format((time.time()-t0)/60))

    # # Compute scores
    # scores = model_final.calc_scores(xdata=xdata, ydata=ydata, to_print=True)

    # # Dump preds
    # model_final.dump_preds(df_data=data, xdata=xdata, target_name=target_name,
    #                        outpath=os.path.join(run_outdir, 'preds.csv'))

    # # Save model
    # model_final.save_model(outdir=run_outdir)



    # ========================================================================
    #       Infer
    # ========================================================================
    lg.logger.info('\n{}'.format('='*50))
    lg.logger.info(f'Inference ... {test_sources}')
    lg.logger.info('='*50)

    csv_scores = []  # cross-study-validation scores
    for i, src in enumerate(test_sources):
        lg.logger.info(f'\nTest source {i+1}:  _____ {src} _____')
        t0 = time.time()

        if train_sources == [src]:
            lg.logger.info("That's the taining set (so no preds).")
            continue

        te_src_data = te_data[te_data['SOURCE'].isin([src])].reset_index(drop=True)
        lg.logger.info(f'src_data.shape {te_src_data.shape}')
        if te_src_data.shape[0] == 0:
            continue

        # Prepare test data for predictions
        # te_src_data = utils_tidy.impute_values(data=te_src_data, fea_prfx_dict=fea_prfx_dict, logger=lg.logger)
        xte, _ = utils_tidy.split_features_and_other_cols(te_src_data, fea_prfx_dict=fea_prfx_dict)
        yte = utils_tidy.extract_target(data=te_src_data, target_name=target_name)

        # Plot dist of response
        utils.plot_hist(x=te_src_data[target_name], var_name=target_name,
                        path=os.path.join(run_outdir, target_name + '_hist_' + src + '.png'))

        # Print feature shapes
        lg.logger.info(f'\nxte_'+src)
        utils_tidy.print_feature_shapes(df=xte, logger=lg.logger)

        # Calc and save scores
        lg.logger.info('\nscores:')
        scores = model_final.calc_scores(xdata=xte, ydata=yte, to_print=True)
        #csv_scores.append(scores)
        csv_scores.append( pd.DataFrame([scores], index=[src]).T )
        
        # Dump preds
        preds_fname = 'preds_' + src + '_' + mlmodel + '.csv'
        model_final.dump_preds(df_data=te_src_data, xdata=xte, target_name=target_name,
                               outpath=os.path.join(run_outdir, preds_fname))                 

        lg.logger.info('\nRuntime: {:.3f}'.format((time.time()-t0)/60))

    # Combine test set preds
    if len(csv_scores) > 0:
        csv_scores = pd.concat(csv_scores, axis=1)

    # (New) Adjust cv_scores in order to combine with test set preds
    # (take the cv score for val set)
    cv_scores = cv_scores[cv_scores['tr_set']==False].drop(columns='tr_set')
    cv_scores[train_sources_name] = cv_scores.iloc[:, -cv_folds:].mean(axis=1)
    cv_scores = cv_scores[['metric', train_sources_name]]
    cv_scores = cv_scores.set_index('metric')

    # Combine scores from val set cross-validation and test set
    csv_scores_all = pd.concat([cv_scores, csv_scores], axis=1)

    csv_scores_all = csv_scores_all.round(decimals=3)
    csv_scores_all.insert(loc=0, column='train_src', value=train_sources_name)
    csv_scores_all = csv_scores_all.reset_index().rename(columns={'index': 'metric'})
    lg.logger.info('\ncsv_scores\n{}'.format(csv_scores_all))
    csv_scores_all.to_csv(os.path.join(run_outdir, 'csv_scores_' + train_sources_name + '.csv'), index=False)


    # Kill logger
    lg.kill_logger()

    del data, xdata, ydata, model, model_final
    return csv_scores_all


def main(args):
    config_fname = os.path.join(file_path, CONFIGFILENAME)
    args = argparser.get_args(args=args, config_fname=config_fname)
    pprint(vars(args))
    args = vars(args)
    #if 'outdir' not in args:
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