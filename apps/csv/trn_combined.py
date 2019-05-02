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
    assert mltype in ['reg', 'cls'], "mltype should be either 'reg' or 'cls'."    
    
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

    tr_data = get_data_by_src(
            dataset, src_names=tr_sources, logger=lg.logger)
    
    xdata, ydata, meta, tr_scaler = break_src_data(
            tr_data, target=args['target_name'],
            scaler_method=args['scaler'], logger=lg.logger)


    # ========================================================================
    #       Plots
    # ========================================================================
    figpath = run_outdir/'figs'
    os.makedirs(figpath, exist_ok=True)
   
    utils.boxplot_rsp_per_drug(tr_data, target_name=target_name,
        path=figpath / f'{target_name}_per_drug_boxplot.png')

    utils.plot_hist(ydata, var_name=target_name,
        path=figpath / (target_name+'_hist.png') )
    
    utils.plot_qq(ydata, var_name=target_name,
        path=figpath / (target_name+'_qqplot.png') )
    
    utils.plot_hist_drugs(tr_data['DRUG'], path=figpath/'drugs_hist.png')


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
    if model_name == 'lgb_reg':
        init_prms = {'n_jobs': n_jobs, 'random_state': SEED, 'logger': lg.logger}
        fit_prms = {'verbose': False}  # 'early_stopping_rounds': 10, 'sample_weight': sample_weight
    elif model_name == 'nn_reg':
        init_prms = {'input_dim': xdata.shape[1], 'dr_rate': dr_rate, 'opt_name': opt_name, 'attn': attn, 'logger': lg.logger}
        fit_prms = {'batch_size': batch_size, 'epochs': epochs, 'verbose': 1}  # 'validation_split': 0.1

    # -----------------
    # sklearn CV method - (doesn't work with keras)
    # ----------------- 
    """
    # Define ML model
    model = ml_models.get_model(model_name=model_name, init_params=init_prms)  

    # Run CV
    t0 = time.time()
    cv_scores = cross_validate(
        estimator=model.model,
        X=xdata, y=ydata,
        scoring=metrics,
        cv=cv, groups=groups,
        n_jobs=n_jobs, fit_params=fit_prms)
    lg.logger.info('Runtime: {:.3f} mins'.format((time.time()-t0)/60))

    # Dump results
    cv_scores = utils.update_cross_validate_scores( cv_scores )
    cv_scores = cv_scores.reset_index(drop=True)
    cv_scores.insert( loc=cv_scores.shape[1]-cv_folds, column='mean', value=cv_scores.iloc[:, -cv_folds:].values.mean(axis=1) )
    cv_scores.insert( loc=cv_scores.shape[1]-cv_folds, column='std',  value=cv_scores.iloc[:, -cv_folds:].values.std(axis=1) )
    cv_scores = cv_scores.round(3)
    cv_scores.to_csv(os.path.join(run_outdir, 'cv_scores_' + train_sources_name + '.csv'), index=False)
    lg.logger.info(f'cv_scores\n{cv_scores}')
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


    # ========================================================================
    #       Train final model (entire dataset) TODO: test this!
    # ========================================================================
    if retrain:
        lg.logger.info('\n{}'.format('='*50))
        lg.logger.info(f'Train final model (use entire dataset) ... {tr_sources}')
        lg.logger.info('='*50)

        # Get the data
        xdata, ydata, _, _ = break_src_data(
                tr_data, target=args['target_name'],
                scaler_method=args['scaler'], logger=lg.logger)

        # Define sample weight
        # From lightgbm docs: n_samples / (n_classes * np.bincount(y))
        # thres_target = 0.5
        # a = np.where(ydata.values < thres_target, 0, 1)
        # wgt = len(a) / (2 * np.bincount(a))
        # sample_weight = np.array([wgt[0] if v < 0.5 else wgt[1] for v in a])

        # ML model params
        if model_name == 'lgb_reg':
            # Use early stopping
            init_prms = {'n_jobs': n_jobs, 'random_state': SEED, 'n_estimators': 2000, 'logger': lg.logger}
            X_train, X_test, y_train, y_test = train_test_split(xdata, ydata, test_size=0.1, random_state=SEED)
            xdata, ydata = X_train, y_train
            eval_set = (X_test, y_test)
            fit_prms = {'verbose': False, 'eval_set': eval_set, 'early_stopping_rounds': 10}  # 'sample_weight': sample_weight
            
        elif model_name == 'nn_reg':
            val_split = 0.1
            init_prms = {'input_dim': xdata.shape[1], 'dr_rate': dr_rate, 'opt_name': opt_name, 'attn': attn, 'logger': lg.logger}
            fit_prms = {'batch_size': batch_size, 'epochs': epochs, 'verbose': 1, 'validation_split': val_split}
            args['final_model_val_split'] = val_split

        # Define ML model
        model_final = ml_models.get_model(model_name, init_params=init_prms)   

        # Train
        t0 = time()
        model_final.model.fit(xdata, ydata, **fit_prms) # TODO: there is a problem here with fit_prms (it seems like it reuses them from my_cross_validate)
        lg.logger.info('Runtime: {:.1f} mins'.format( (time()-t0)/60) )

        # # Save model
        # model_final.save_model(outdir=run_outdir)
    
    else:
        model_final = best_model

    # Dump model
    model_final.dump_model(outdir=run_outdir)

    # Save network figure
    if 'nn' in model_name:
        from keras.utils import plot_model
        plot_model(model_final.model, to_file=figpath/'nn_model.png')


    # ========================================================================
    #       Infer
    # ========================================================================
    lg.logger.info('\n{}'.format('='*50))
    lg.logger.info(f'Inference ... {te_sources}')
    lg.logger.info('='*50)
    
    csv = []  # cross-study-validation scores
    for i, te_src in enumerate(te_sources):
        lg.logger.info(f'\nTest source {i+1}:  _____ {te_src} _____')
        t0 = time()

        if tr_sources == [te_src]:  # te_src in tr_sources: 
            lg.logger.info("That is the train set (take preds from cv run).")
            continue
        
        # Extract test data
        te_src_data = get_data_by_src(
            dataset, src_names=[te_src], logger=lg.logger)
        
        if te_src_data.shape[0] == 0:
            continue  # continue if there are no data samples available for this source
        
        # Extract features and target
        xte, yte, _, _ = break_src_data(
            te_src_data, target=args['target_name'],
            scaler_method=None, logger=lg.logger)

        # Scale test data
        colnames = xte.columns
        xte = pd.DataFrame( tr_scaler.transform(xte), columns=colnames ).astype(np.float32)
        
        # Plot dist of response
        utils.plot_hist(yte, var_name=target_name,
                        path=figpath / (target_name + '_hist_' + te_src + '.png') )

        # Calc scores
        # scores = model_final.calc_scores(xdata=xte, ydata=yte, to_print=True)
        y_preds, y_true = utils.calc_preds(estimator=model_final.model, x=xte, y=yte, mltype=mltype)
        scores = utils.calc_scores(y_true=y_true, y_preds=y_preds, mltype=mltype)
        csv.append( pd.DataFrame([scores], index=[te_src]).T )
        
        # Dump preds --> TODO: error when use keras
        # preds_fname = 'preds_' + src + '_' + model_name + '.csv'
        # model_final.dump_preds(df_data=te_src_data, xdata=xte, target_name=target_name,
        #                        outpath=os.path.join(run_outdir, preds_fname))                 

        lg.logger.info('Runtime: {:.1f} mins'.format( (time()-t0)/60) )

    # Combine test set preds
    if len(csv) > 0:
        csv = pd.concat(csv, axis=1)

    # Adjust cv_scores in order to combine with test set preds
    # (take the mean cv score for val set)
    cv_scores = cv_scores[cv_scores['tr_set']==False].drop(columns='tr_set')
    cv_scores[tr_sources_name] = cv_scores.iloc[:, -cv_folds:].mean(axis=1)
    cv_scores = cv_scores[['metric', tr_sources_name]]
    cv_scores = cv_scores.set_index('metric')
    cv_scores.index.name = None

    # Combine scores from val set cross-validation and test set
    csv_all = pd.concat([cv_scores, csv], axis=1)
    csv_all.insert(loc=0, column='train_src', value=tr_sources_name)
    csv_all = csv_all.reset_index().rename(columns={'index': 'metric'})
    # csv_all = csv_all.round(decimals=3)

    lg.logger.info(f'\ncsv_scores\n{csv_all}')
    csv_all.to_csv( run_outdir/('csv_scores_' + tr_sources_name + '.csv'), index=False )

    # Kill logger
    lg.kill_logger()
    del tr_data, te_src_data, xdata, ydata, xte, yte, model_final
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
    # python -m pdb apps/csv/trn_from_combined.py -te ccle gcsi -tr gcsi
    """ __name__ == '__main__' explained:
    www.youtube.com/watch?v=sugvnHA7ElY
    """
    """
    stackoverflow.com/questions/14500183/in-python-can-i-call-the-main-of-an-imported-module
    How to run code with input args from another code?
    This will be used with multiple train and test sources.
    For example: in launch_csv.py
        import train_combined.py
        train_combined.main([tr_src, tst_src])
    """
    # python -m pdb apps/csv/trn_from_combined.py -te ccle gcsi -tr gcsi
    main(sys.argv[1:])
