""" 
This code generates cv and/or train/test splits of a dataset.
TODO: Add plots of the splits (e.g. drug, cell line, reponse distributions).
"""
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

from pandas.api.types import is_string_dtype
from sklearn.preprocessing import LabelEncoder

SEED = 42


# File path
# file_path = os.path.dirname(os.path.realpath(__file__))
file_path = Path(__file__).resolve().parent


# Utils
utils_path = file_path / '../../utils'
sys.path.append(str(utils_path))
import utils
from utils_tidy import load_tidy_combined, get_data_by_src, break_src_data 
from classlogger import Logger
from cv_splitter import cv_splitter, plot_ytr_yvl_dist


# Path
PRJ_NAME = file_path.name 
OUTDIR = file_path / '../../data/processed' / PRJ_NAME

        
def run(args):
    t0 = time()

    dpath = args['dpath'] 
    dname = args['dname'] # TODO: remove this functionality
    src_names = args['src_names']
    
    # Target
    target_name = args['target_name']

    # Data splits
    te_method = args['te_method']
    te_size = args['te_size']
    cv_method = args['cv_method']
    # cv_folds = args['cv_folds']
    
    # Features 
    cell_fea = args['cell_features']
    drug_fea = args['drug_features']
    other_fea = args['other_features']
    fea_list = cell_fea + drug_fea + other_fea    
    
    # Other params
    n_jobs = args['n_jobs']

    # this is for `combined` TODO: probably need to pass this with meta
    if dname == 'combined':
        grp_by_col = 'CELL' 
    else:
        # some datasets may not have columns to group by
        grp_by_col = None
        cv_method = 'simple'

    # Print args
    pprint(args)

    # TODO: this need to be improved
    mltype = 'reg'  # required for the splits (stratify in case of classification)
    
    
    # -----------------------------------------------
    #       Logger
    # -----------------------------------------------
    if dpath is not None: 
        prffx = Path(dpath).name.split('.')[:-1]
        prffx = '_'.join(prffx)
        prffx = 'top6' if 'top6' in prffx else prffx

    elif dname == 'combined':
        prffx = '_'.join(src_names)

    if te_method is not None:
        fname = prffx + '_te_' + te_method + '_cv_' + cv_method
    else:
        fname = prffx + '_cv_' + cv_method

    run_outdir = OUTDIR/fname
    os.makedirs(run_outdir, exist_ok=True)
    logfilename = run_outdir/'logfile.log'
    lg = Logger(logfilename)

    lg.logger.info(f'File path: {file_path}')
    lg.logger.info(f'System CPUs: {psutil.cpu_count(logical=True)}')
    lg.logger.info(f'n_jobs: {n_jobs}')

    # Dump args to file
    utils.dump_args(args, run_outdir)        
        
        
    # -----------------------------------------------
    #       Load data and pre-proc
    # -----------------------------------------------
    if dpath is not None: 
        if dpath.split('.')[-1] == 'parquet': 
            data = pd.read_parquet(dpath, engine='auto', columns=None)
        else:
            data = pd.read_csv(dpath)
        
        # Split features and traget
        ydata = data.loc[:, [target_name]]
        xdata = data.drop(columns=target_name)

    elif dname == 'combined':
        DATADIR = file_path / '../../data/processed/from_combined/tidy_drop_fibro'
        DATAFILENAME = 'tidy_data.parquet'
        datapath = DATADIR / DATAFILENAME
    
        data = load_tidy_combined( datapath, fea_list=fea_list, shuffle=False, random_state=SEED ) # logger=lg.logger
        data = get_data_by_src( data, src_names=src_names, logger=lg.logger )
        xdata, ydata, meta, tr_scaler = break_src_data( data, target=target_name, scaler=None, logger=lg.logger)
        ydata = pd.DataFrame(ydata)
    
        meta.to_parquet(run_outdir/'meta.parquet', engine='auto', compression='snappy')

    # Dump data
    xdata = xdata.reset_index(drop=True)
    ydata = ydata.reset_index(drop=True)
    xdata.to_parquet(run_outdir/'xdata.parquet', engine='auto', compression='snappy')
    ydata.to_parquet(run_outdir/'ydata.parquet', engine='auto', compression='snappy')


    # -----------------------------------------------
    #       Train-test split
    # -----------------------------------------------
    idx_vec = np.random.permutation(xdata.shape[0])

    if te_method is not None:
        lg.logger.info('\nSplit train/test.')
        te_splitter = cv_splitter(cv_method=te_method, cv_folds=1, test_size=te_size,
                                  mltype=mltype, shuffle=False, random_state=SEED)
        if te_method=='simple':
            te_grp = None
        elif te_method=='group':
            te_grp = meta[grp_by_col].copy()
            te_grp = te_grp.values[idx_vec]

        if is_string_dtype(te_grp):
            grp_enc = LabelEncoder()
            te_grp = grp_enc.fit_transform(te_grp)
   
        # Split train/test
        tr_id, te_id = next(te_splitter.split(idx_vec, groups=te_grp))
        tr_id = idx_vec[tr_id] # adjust the indices!
        te_id = idx_vec[te_id] # adjust the indices!

        pd.Series(tr_id).to_csv(run_outdir/f'tr_id.csv', index=False, header=False)
        pd.Series(te_id).to_csv(run_outdir/f'te_id.csv', index=False, header=False)

        #te_xdata = xdata.iloc[te_id, :]
        te_ydata = ydata.iloc[te_id, :] 
        #tr_xdata = xdata.iloc[tr_id, :]  
        tr_ydata = ydata.iloc[tr_id, :]  
        
        # Update the master idx vector for the CV splits
        idx_vec = tr_id

        # Plot dist of responses
        plot_ytr_yvl_dist(ytr=tr_ydata.values, yvl=te_ydata.values,
                title='tr and te', outpath=run_outdir/'tr_te_resp_dist.png')

        # Confirm that group splits are correct
        if te_method=='group' and grp_by_col is not None:
            tr_grps_unq = set(meta.loc[tr_id, grp_by_col])
            vl_grps_unq = set(meta.loc[te_id, grp_by_col])
            lg.logger.info(f'  Total group ({grp_by_col}) intersections btw tr and te: {len(tr_grps_unq.intersection(te_grps_unq))}.')
            lg.logger.info(f'  A few intersections : {list(tr_grps_unq.intersection(te_grps_unq))[:3]}.')
    
        # Save train/test
        #tr_xdata.to_parquet(run_outdir/'xdata.parquet', engine='auto', compression='snappy')
        #tr_ydata.to_parquet(run_outdir/'ydata.parquet', engine='auto', compression='snappy')
        #te_xdata.to_parquet(run_outdir/'te_xdata.parquet', engine='auto', compression='snappy')
        #te_ydata.to_parquet(run_outdir/'te_ydata.parquet', engine='auto', compression='snappy')

        del te_xdata, tr_xdata, te_ydata, tr_ydata, te_meta, tr_meta, tr_id, te_id


    # -----------------------------------------------
    #       Generate CV splits
    # -----------------------------------------------
    cv_folds_list = [1, 5, 7, 10, 15, 20]
    lg.logger.info(f'\nStart CV splits ...')

    for cv_folds in cv_folds_list:
        lg.logger.info(f'\nCV folds: {cv_folds}')

        vl_size = 0.2
        cv = cv_splitter(cv_method=cv_method, cv_folds=cv_folds, test_size=vl_size,
                         mltype=mltype, shuffle=False, random_state=SEED)

        if cv_method=='simple':
            cv_grp = None
        elif cv_method=='group':
            cv_grp = meta[grp_by_col].copy()
            cv_grp = cv_grp.values[idx_vec]

        if is_string_dtype(cv_grp):
            grp_enc = LabelEncoder()
            cv_grp = grp_enc.fit_transform(cv_grp)
    
        tr_folds = {} 
        vl_folds = {} 

        # Start CV iters
        for fold, (tr_id, vl_id) in enumerate(cv.split(idx_vec, groups=cv_grp)):
            tr_id = idx_vec[tr_id] # adjust the indices!
            vl_id = idx_vec[vl_id] # adjust the indices!
            # if lg.logger is not None:
            #     lg.logger.info(f'Fold {fold+1}/{cv_folds}')

            tr_folds[fold] = tr_id.tolist()
            vl_folds[fold] = vl_id.tolist()

            # Confirm that group splits are correct
            if cv_method=='group' and grp_by_col is not None:
                tr_grps_unq = set(meta.loc[tr_id, grp_by_col])
                vl_grps_unq = set(meta.loc[vl_id, grp_by_col])
                lg.logger.info(f'  Total group ({grp_by_col}) intersections btw tr and vl: {len(tr_grps_unq.intersection(vl_grps_unq))}.')
                lg.logger.info(f'   Unique cell lines in tr: {len(tr_grps_unq)}.')
                lg.logger.info(f'   Unique cell lines in vl: {len(vl_grps_unq)}.')
        
        # Convet to df
        # from_dict takes too long  -->  stackoverflow.com/questions/19736080/
        # tr_folds = pd.DataFrame.from_dict(tr_folds, orient='index').T 
        # vl_folds = pd.DataFrame.from_dict(vl_folds, orient='index').T
        tr_folds = pd.DataFrame(dict([ (k, pd.Series(v)) for k, v in tr_folds.items() ]))
        vl_folds = pd.DataFrame(dict([ (k, pd.Series(v)) for k, v in vl_folds.items() ]))

        # Dump
        tr_folds.to_csv(run_outdir/f'{cv_folds}fold_tr_id.csv', index=False)
        vl_folds.to_csv(run_outdir/f'{cv_folds}fold_vl_id.csv', index=False)

    lg.kill_logger()
    print('Done.')


def main(args):
    parser = argparse.ArgumentParser(description="Generate and save dataset splits.")

    # Data path
    parser.add_argument('--dpath',
            default=None, type=str,
            help='Full data path (default: None).')

    # Data name
    parser.add_argument('--dname',
        default=None, choices=['combined'],
        help='Data name (default: None).')

    # Cell line sources 
    parser.add_argument('-src', '--src_names', nargs='+',
        default=None, choices=['ccle', 'gcsi', 'gdsc', 'ctrp', 'nci60'],
        help='Data sources to use (default: None).')

    # Target to predict
    parser.add_argument('-t', '--target_name',
        default='AUC', choices=['AUC', 'AUC1', 'IC50'],
        help='Column name of the target variable (default: `AUC`).')

    # Feature types
    parser.add_argument('-cf', '--cell_features', nargs='+',
        default=['rna'], choices=['rna', 'cnv', 'clb'],
        help='Cell line features (default: `rna`).') # ['rna_latent']
    parser.add_argument('-df', '--drug_features', nargs='+',
        default=['dsc'], choices=['dsc', 'fng', 'dlb'],
        help='Drug features (default: `dsc`).') # ['fng', 'dsc_latent', 'fng_latent']
    parser.add_argument('-of', '--other_features',
        default=[], choices=[],
        help='Other feature types (derived from cell lines and drugs). E.g.: cancer type, etc).') # ['cell_labels', 'drug_labels', 'ctype', 'csite', 'rna_clusters']

    # Data split methods
    parser.add_argument('-tem', '--te_method',
        default=None, choices=['simple', 'group'],
        help='Test split method (default: None).')
    parser.add_argument('--te_size', type=float,
        default=0.1, 
        help='Test size split ratio (default: 0.1).')
    parser.add_argument('-cvm', '--cv_method',
        default='simple', choices=['simple', 'group'],
        help='Cross-val split method (default: `simple`).')

    # Define n_jobs
    parser.add_argument('--n_jobs', default=4,  type=int, help='Default: 4.')

    # Parse args and run
    args = parser.parse_args(args)
    args = vars(args)
    ret = run(args)
    
    
if __name__ == '__main__':
    """ __name__ == '__main__' explained: www.youtube.com/watch?v=sugvnHA7ElY """
    main(sys.argv[1:])

