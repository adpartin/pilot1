""" 
This script generates multiple learning curves for different training sets.
It launches a script (e.g. trn_lrn_crv.py) that train ML model(s) on various training set sizes.

TODO: update the function trn_lrn_crv.py so that it will accept a df instead of the source names!
"""
# python -m pdb apps/lrn_crv/launch_lrn_crv.py
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

SEED = None


# File path
# file_path = os.path.dirname(os.path.realpath(__file__))
file_path = Path(__file__).resolve().parent


# Utils
utils_path = file_path / '../../utils'
sys.path.append(str(utils_path))
# import trn_lrn_crv
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
CONFIGFILENAME = 'config_prms.txt'



def plot_agg_lrn_crv(df, outdir='.'):
    """ Generates learning curve plots for each metric across all cell line sources. """
    # Get the number of cv_folds
    cvf = len([c for c in df.columns.tolist() if c[0]=='f'])

    rs = ['b', 'r', 'k', 'c', 'm']
    title = None

    for i, met_name in enumerate(df['metric'].unique()):
        dfm = df[df['metric']==met_name].reset_index(drop=True)

        y_values = dfm.iloc[:, -cvf:].values
        y_ = y_values.min() * 0.05
        ymin = y_values.min()
        ymax = y_values.max()
        ylim = [ymin - y_, ymax + y_]

        fig = plt.figure(figsize=(14, 7))
        for j, s in enumerate(dfm['src'].unique()):

            dfs = dfm[dfm['src']==s].reset_index(drop=True)
            tr_sizes  = dfs['tr_size'].unique()
            tr_scores = dfs.loc[dfs['tr_set']==True, dfs.columns[-cvf:]]
            te_scores = dfs.loc[dfs['tr_set']==False, dfs.columns[-cvf:]]

            tr_scores_mean = np.mean(tr_scores, axis=1)
            tr_scores_std  = np.std(tr_scores, axis=1)
            te_scores_mean = np.mean(te_scores, axis=1)
            te_scores_std  = np.std(te_scores, axis=1)

            plt.plot(tr_sizes, tr_scores_mean, '.-', color=colors[j], label=s+'_tr')
            plt.plot(tr_sizes, te_scores_mean, '.--', color=colors[j], label=s+'_val')

            plt.fill_between(tr_sizes, tr_scores_mean - tr_scores_std, tr_scores_mean + tr_scores_std, alpha=0.1, color=colors[j])
            plt.fill_between(tr_sizes, te_scores_mean - te_scores_std, te_scores_mean + te_scores_std, alpha=0.1, color=colors[j])

            if title is not None:
                plt.title(title)
            else:
                plt.title('Learning curve (' + met_name + ')')
            plt.xlabel('Train set size')
            plt.ylabel(met_name)
            plt.legend(bbox_to_anchor=(1.1, 1), loc='upper right', ncol=1)
            # plt.legend(loc='best')
            plt.grid(True)
            plt.tight_layout()
        
        plt.ylim(ylim) 
        plt.savefig( Path(outdir) / ('lrn_crv_' + met_name + '.png') )


        
def create_outdir(outdir, args, src):
    t = datetime.now()
    t = [t.year, '-', t.month, '-', t.day, '_', 'h', t.hour, '-', 'm', t.minute]
    t = ''.join([str(i) for i in t])
    
    if ('nn' in args['model_name']) and (args['attn'] is True): 
        name_sffx = '.'.join( [src] + [args['model_name']] + ['attn'] + \
                             args['opt'] + [args['cv_method']] + [('cvf'+str(args['cv_folds']))] + args['cell_features'] + \
                             args['drug_features'] + [args['target_name']] )
            
    elif ('nn' in args['model_name']) and (args['attn'] is False): 
        name_sffx = '.'.join( [src] + [args['model_name']] + ['fc'] + \
                             args['opt'] + [args['cv_method']] + [('cvf'+str(args['cv_folds']))] + args['cell_features'] + \
                             args['drug_features'] + [args['target_name']] )
        
    else:
        name_sffx = '.'.join( [src] + [args['model_name']] + \
                             args['opt'] + [args['cv_method']] + [('cvf'+str(args['cv_folds']))] + args['cell_features'] + \
                             args['drug_features'] + [args['target_name']] )

    outdir = Path(outdir) / (name_sffx + '_' + t)
    os.makedirs(outdir)
    return outdir
    
        
        
def run(args):
    t0 = time()

    # Create folder to store results
    #t = datetime.now()
    #t = [t.year, '-', t.month, '-', t.day, '_', 'h', t.hour, '-', 'm', t.minute]
    #t = ''.join([str(i) for i in t])
    #dirname = 'lrn_crv_' + t
    #outdir = OUTDIR/dirname
    #os.makedirs(outdir, exist_ok=True)
    
    
    # Full set
#     csv_sets = [
#         {'tr_src': ['gcsi']},
#         {'tr_src': ['ccle']},
#         {'tr_src': ['gdsc']},
#         {'tr_src': ['ctrp']},
#         {'tr_src': ['nci60']}
#     ]
    dname = args['dname']

    cell_fea = args['cell_features']
    drug_fea = args['drug_features']
    other_fea = args['other_features']
    tr_sources = args['train_sources']
    
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
    
    # Print args
    pprint(args)

    # Define names
    #tr_sources_name = '_'.join(tr_sources)    
    
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
    #       Load data and pre-proc
    # ========================================================================
    if dname == 'combined':
        DATADIR = file_path / '../../data/processed/from_combined/tidy_drop_fibro'
        DATAFILENAME = 'tidy_data.parquet'
        datapath = DATADIR / DATAFILENAME
    
        dataset = load_tidy_combined(
                datapath, fea_list=fea_list, random_state=SEED) # logger=lg.logger

        dfs = {}
        for src in tr_sources:
            print(f'\n{src} ...')
            tr_data = get_data_by_src(
                    dataset, src_names=[src]) # logger=lg.logger

            xdata, ydata, _, _ = break_src_data(
                    tr_data, target=args['target_name'],
                    scaler_method=args['scaler']) # logger=lg.logger
            
            dfs[src] = (ydata, xdata)
            
        del tr_data, xdata, ydata

    elif dname == 'top6':
        DATADIR = file_path / '../../data/raw/'
        DATAFILENAME = 'uniq.top6.reg.parquet'
        datapath = DATADIR / DATAFILENAME
        
        df = pd.read_parquet(datapath, engine='auto', columns=None)
        df = df.sample(frac=1.0, axis=0, random_state=SEED).reset_index(drop=True)

        scaler_method = args['scaler']
        if  scaler_method is not None:
            if scaler_method == 'stnd':
                scaler = StandardScaler()
            elif scaler_method == 'minmax':
                scaler = MinMaxScaler()
            elif scaler_method == 'rbst':
                scaler = RobustScaler()

        xdata = df.iloc[:, 1:]
        ydata = df.iloc[:, 0]
        xdata = scaler.fit_transform(xdata).astype(np.float32)
        
    
    
    for src, data in dfs.items():
        #ourdir = create_outdir(OUTDIR, args, src)
        ydata, xdata = data[0], data[1]
        #trn_lrn_crv_new(xdata, ydata, *args)
        
        # -----------------------------------------------
        #       Logger
        # -----------------------------------------------
        #run_outdir = utils.create_outdir(outdir, args=args)
        run_outdir = create_outdir(OUTDIR, args, src)
        logfilename = run_outdir/'logfile.log'
        lg = Logger(logfilename)

        lg.logger.info(f'File path: {file_path}')
        lg.logger.info(f'System CPUs: {psutil.cpu_count(logical=True)}')
        lg.logger.info(f'n_jobs: {n_jobs}')

        # Dump args to file
        utils.dump_args(args, run_outdir)        
        
        
        # -----------------------------------------------
        #       Define CV split
        # -----------------------------------------------
        cv = cv_splitter(cv_method=cv_method, cv_folds=cv_folds, test_size=0.2,
                         mltype=mltype, shuffle=True, random_state=SEED)
        if cv_method=='simple':
            groups = None
        elif cv_method=='group':
            groups = tr_data['CELL'].copy()
        
        # -----------------------------------------------
        #      ML model configurations
        # -----------------------------------------------
        lg.logger.info('\n\n{}'.format('='*50))
        if dname == 'combined':
            lg.logger.info(f'Learning curves ... {src}')
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
        #      Generate learning curve 
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
        lrn_crv_scores.to_csv( run_outdir/('lrn_crv_scores_' + src + '.csv'), index=False) 
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
        
    print('Done.')


#     # Multiple runs
#     dfs = []
#     for run_id in range(len(csv_sets)):
#         print('{} Run {} {}'.format('-'*40, run_id+1, '-'*40))
#         lrn_crv_scores, prms = trn_lrn_crv.main(
#             ['-tr', *csv_sets[run_id]['tr_src'],
#              '--outdir', str(outdir),
#              *args])
#         src_name = '_'.join( csv_sets[run_id]['tr_src'] )
#         lrn_crv_scores.insert(loc=0, column='src', value=src_name)
#         dfs.append(lrn_crv_scores)

#     df = pd.concat(dfs, axis=0, sort=False)
#     df.to_csv(outdir/'lrn_crv_all.csv', index=False)

#     plot_agg_lrn_crv(df, outdir)

#     print('Total runtime {:.1f}\n'.format( (time()-t0)/60) )


def main(args):
    config_fname = file_path / CONFIGFILENAME
    args = argparser.get_args(args=args, config_fname=config_fname)
    # pprint(vars(args))
    args = vars(args)
    if args['outdir'] is None:
        args['outdir'] = OUTDIR
    lrn_crv_scores = run(args)
    # return lrn_crv_scores, args    
    
    
if __name__ == '__main__':
    """ __name__ == '__main__' explained: www.youtube.com/watch?v=sugvnHA7ElY """
    main(sys.argv[1:])

