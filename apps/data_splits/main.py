""" 
This code generates cv and/or train/test splits of a dataset.
TODO: Add plots of the splits (e.g. drug, cell line, reponse distributions).
"""
from __future__ import print_function, division

import warnings
warnings.filterwarnings('ignore')

import os
import sys
import platform
from pathlib import Path
import psutil
import argparse
from datetime import datetime
from time import time
from pprint import pprint, pformat
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

        
def parse_args(args):
    parser = argparse.ArgumentParser(description='Generate and save dataset splits.')

    # Input data
    parser.add_argument('--dirpath', default=None, type=str, help='Full path to data (default: None).')

    # Combined related params
    # Data name
    parser.add_argument('--dname', default=None, choices=['combined'], help='Data name (default: None).')
    parser.add_argument('--rna_norm', default='raw', choices=['raw', 'combat'], help='RNA normalization (default: raw).')
    parser.add_argument('--no_fibro', action='store_true', default=False, help='Default: False')
    parser.add_argument('--src', nargs='+', default=None, choices=['ccle', 'gcsi', 'gdsc', 'ctrp', 'nci60'],
        help='Data sources to use (default: None).')

    # Target to predict
    # parser.add_argument('-t', '--target_name', default='AUC', choices=['AUC', 'AUC1', 'IC50'],
    #     help='Column name of the target variable (default: `AUC`).')

    # Feature types
    parser.add_argument('-cf', '--cell_fea', nargs='+', default=['rna'], choices=['rna', 'cnv', 'clb'],
        help='Cell line features (default: rna).') # ['rna_latent']
    parser.add_argument('-df', '--drug_fea', nargs='+', default=['dsc'], choices=['dsc', 'fng', 'dlb'],
        help='Drug features (default: dsc).') # ['fng', 'dsc_latent', 'fng_latent']
    parser.add_argument('-of', '--other_fea', default=[], choices=[],
        help='Other feature types (derived from cell lines and drugs). E.g.: cancer type, etc).') # ['cell_labels', 'drug_labels', 'ctype', 'csite', 'rna_clusters']

    # Data split methods
    parser.add_argument('--te_method', default=None, choices=['simple', 'group'], help='Test split method (default: None).')
    parser.add_argument('--te_size', type=float, default=0.1, help='Test size split ratio (default: 0.1).')
    parser.add_argument('--cv_method', default='simple', choices=['simple', 'group'], help='Cross-val split method (default: simple).')
    parser.add_argument('--vl_size', type=float, default=0.1, help='Val size split ratio for single split (default: 0.1).')

    # Define n_jobs
    parser.add_argument('--n_jobs', default=4,  type=int, help='Default: 4.')

    # Parse args and run
    args = parser.parse_args(args)
    return args


def create_outdir(outdir, args, src):
    l = [('cvf'+str(args['cv_folds']))] + args['cell_fea'] + args['drug_fea'] + [args['target_name']] 
    if args['clr_mode'] is not None: l = [args['clr_mode']] + l
    if 'nn' in args['model_name']: l = [args['opt']] + l
                
    name_sffx = '.'.join( [src] + [args['model_name']] + l )
    outdir = Path(outdir) / name_sffx
    # os.makedirs(outdir)
    os.makedirs(outdir, exist_ok=True)
    return outdir


def split_size(x):
    """ Split size can be float (0, 1) or int.
    This function casts this value as needed. 
    """
    assert x > 0, 'Split size must be greater than 0.'
    return int(x) if x > 1.0 else x


def run(args):
    if args['dirpath'] is not None: dirpath = Path(args['dirpath'])

    # Combined
    dname = args['dname']
    rna_norm = args['rna_norm']
    no_fibro = args['no_fibro']
    src = args['src']
    
    # Target
    # target_name = args['target_name']

    # Data splits
    te_method = args['te_method']
    cv_method = args['cv_method']
    te_size = split_size(args['te_size'])
    vl_size = split_size(args['vl_size'])

    te_size = split_size(args['te_size'])
    vl_size = split_size(args['vl_size'])

    # Features 
    cell_fea = args['cell_fea']
    drug_fea = args['drug_fea']
    other_fea = args['other_fea']
    fea_list = cell_fea + drug_fea + other_fea    
    
    # Other params
    n_jobs = args['n_jobs']

    # This is for `combined` TODO: probably need to pass this with meta
    if dname == 'combined':
        grp_by_col = 'CELL' 
    else:
        # some datasets may not have columns to group by
        grp_by_col = None
        cv_method = 'simple'

    # TODO: this need to be improved
    mltype = 'reg'  # required for the splits (stratify in case of classification)
    
    
    # -----------------------------------------------
    #       Ourdir and Logger
    # -----------------------------------------------
    if dirpath is not None:
        prffx = dirpath.name
    elif dname == 'combined':
        prffx = '_'.join(src)

    # fname = prffx + '_' + rna_norm + '_cv_' + cv_method
    fname = prffx + '_cv_' + cv_method
    if te_method is not None: fname = fname + '_te_' + te_method

    if dname == 'combined':
        fname = fname + '_' + rna_norm
        if no_fibro: fname = fname + '_no_fibro'

    # Outdir
    run_outdir = OUTDIR / fname
    os.makedirs(run_outdir, exist_ok=True)

    # Logger
    lg = Logger(run_outdir/'logfile.log')
    lg.logger.info(f'File path: {file_path}')
    lg.logger.info(f'\n{pformat(args)}')

    # Dump args to file
    utils.dump_args(args, run_outdir)        
        
        
    # -----------------------------------------------
    #       Load data and pre-proc
    # -----------------------------------------------
    if dirpath is not None:
        if (dirpath/'xdata.parquet').is_file():
            xdata = pd.read_parquet( dirpath/'xdata.parquet', engine='auto', columns=None )
            meta = pd.read_parquet( dirpath/'meta.parquet', engine='auto', columns=None )

    elif dname == 'combined':
        DATADIR = file_path / '../../data/processed/from_combined'
        file_format = '.parquet'
        fname = 'tidy_' + rna_norm
        if no_fibro: fname = fname + '_no_fibro'
            
        DATAFILENAME = fname + file_format
        datapath = DATADIR / DATAFILENAME

        data = load_tidy_combined( datapath, fea_list=fea_list, shuffle=False, random_state=SEED ) # logger=lg.logger
        data = get_data_by_src( data, src_names=src, logger=lg.logger )
        xdata, _, meta, tr_scaler = break_src_data( data, target=None, scaler=None, logger=lg.logger)
        # ydata = pd.DataFrame(ydata)
    
        #meta.to_parquet(run_outdir/'meta.parquet', engine='auto', compression='snappy')

    # Dump data
    xdata = xdata.reset_index(drop=True)
    meta = meta.reset_index(drop=True)
    xdata.to_parquet( run_outdir/'xdata.parquet', engine='auto', compression='snappy' )
    meta.to_parquet( run_outdir/'meta.parquet', engine='auto', compression='snappy' )


    # -----------------------------------------------
    #       Train-test split
    # -----------------------------------------------
    np.random.seed(SEED)
    idx_vec = np.random.permutation(xdata.shape[0])

    if te_method is not None:
        lg.logger.info('\nSplit train/test.')
        te_splitter = cv_splitter(cv_method=te_method, cv_folds=1, test_size=te_size,
                                  mltype=mltype, shuffle=False, random_state=SEED)

        te_grp = meta[grp_by_col].values[idx_vec] if te_method=='group' else None
        if is_string_dtype(te_grp): te_grp = LabelEncoder().fit_transform(te_grp)
   
        # Split train/test
        tr_id, te_id = next(te_splitter.split(idx_vec, groups=te_grp))
        tr_id = idx_vec[tr_id] # adjust the indices!
        te_id = idx_vec[te_id] # adjust the indices!

        pd.Series(tr_id).to_csv(run_outdir/f'tr_id.csv', index=False, header=[0])
        pd.Series(te_id).to_csv(run_outdir/f'te_id.csv', index=False, header=[0])
        
        lg.logger.info('Train: {:.1f}'.format( len(tr_id)/xdata.shape[0] ))
        lg.logger.info('Test:  {:.1f}'.format( len(te_id)/xdata.shape[0] ))
        
        # Update the master idx vector for the CV splits
        idx_vec = tr_id

        # Plot dist of responses (TODO: this can be done to all response metrics)
        # plot_ytr_yvl_dist(ytr=tr_ydata.values, yvl=te_ydata.values,
        #         title='tr and te', outpath=run_outdir/'tr_te_resp_dist.png')

        # Confirm that group splits are correct
        if te_method=='group' and grp_by_col is not None:
            tr_grp_unq = set(meta.loc[tr_id, grp_by_col])
            te_grp_unq = set(meta.loc[te_id, grp_by_col])
            lg.logger.info(f'\tTotal group ({grp_by_col}) intersections btw tr and te: {len(tr_grp_unq.intersection(te_grp_unq))}.')
            lg.logger.info(f'\tA few intersections : {list(tr_grp_unq.intersection(te_grp_unq))[:3]}.')

        # Update vl_size to effective vl_size
        vl_size = vl_size * xdata.shape[0]/len(tr_id)

        del tr_id, te_id


    # -----------------------------------------------
    #       Generate CV splits
    # -----------------------------------------------
    cv_folds_list = [1, 5, 7, 10, 15, 20]
    lg.logger.info(f'\nStart CV splits ...')
    
    for cv_folds in cv_folds_list:
        lg.logger.info(f'\nCV folds: {cv_folds}')

        cv = cv_splitter(cv_method=cv_method, cv_folds=cv_folds, test_size=vl_size,
                         mltype=mltype, shuffle=False, random_state=SEED)

        cv_grp = meta[grp_by_col].values[idx_vec] if cv_method=='group' else None
        if is_string_dtype(cv_grp): cv_grp = LabelEncoder().fit_transform(cv_grp)
    
        tr_folds = {}
        vl_folds = {}

        # Start CV iters
        for fold, (tr_id, vl_id) in enumerate(cv.split(idx_vec, groups=cv_grp)):
            tr_id = idx_vec[tr_id] # adjust the indices!
            vl_id = idx_vec[vl_id] # adjust the indices!

            tr_folds[fold] = tr_id.tolist()
            vl_folds[fold] = vl_id.tolist()

            # Confirm that group splits are correct
            if cv_method=='group' and grp_by_col is not None:
                tr_grp_unq = set(meta.loc[tr_id, grp_by_col])
                vl_grp_unq = set(meta.loc[vl_id, grp_by_col])
                lg.logger.info(f'\tTotal group ({grp_by_col}) intersections btw tr and vl: {len(tr_grp_unq.intersection(vl_grp_unq))}.')
                lg.logger.info(f'\tUnique cell lines in tr: {len(tr_grp_unq)}.')
                lg.logger.info(f'\tUnique cell lines in vl: {len(vl_grp_unq)}.')
        
        # Convet to df
        # from_dict takes too long  -->  faster described here: stackoverflow.com/questions/19736080/
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
    args = parse_args(args)
    args = vars(args)
    ret = run(args)
    
    
if __name__ == '__main__':
    """ __name__ == '__main__' explained: www.youtube.com/watch?v=sugvnHA7ElY """
    main(sys.argv[1:])