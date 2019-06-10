""" 
This code generates cv and/or train/test splits of a dataset.
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

SEED = None


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

    dname = args['dname']
    src_names = args['src_names']
    target_name = args['target_name']
    n_jobs = args['n_jobs']
    
    # Data splits
    te_method = args['te_method']
    te_size = args['te_size']
    cv_method = args['cv_method']
    cv_folds = args['cv_folds']
    
    # Features 
    cell_fea = args['cell_features']
    drug_fea = args['drug_features']
    other_fea = args['other_features']
    fea_list = cell_fea + drug_fea + other_fea    
    
    # Define names
    src_name_join = '_'.join(src_names)

    # Print args
    pprint(args)

    
    # -----------------------------------------------
    #       Logger
    # -----------------------------------------------
    #run_outdir = utils.create_outdir(outdir, args=args)
    #run_outdir = create_outdir(OUTDIR, args, src)
    fname = src_name_join + '_te_' + te_method + '_cv_' + cv_method
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
    if dname == 'combined':
        DATADIR = file_path / '../../data/processed/from_combined/tidy_drop_fibro'
        DATAFILENAME = 'tidy_data.parquet'
        datapath = DATADIR / DATAFILENAME
    
        data = load_tidy_combined( datapath, fea_list=fea_list, random_state=SEED ) # logger=lg.logger
        data = get_data_by_src( data, src_names=src_names, logger=lg.logger )
    
        #xdata, ydata, meta, tr_scaler = break_src_data(
        #        tr_data, target=args['target_name'],
        #        scaler=args['scaler'], logger=lg.logger)

    elif dname == 'top6':
        DATADIR = file_path / '../../data/raw/'
        DATAFILENAME = 'uniq.top6.reg.parquet'
        datapath = DATADIR / DATAFILENAME
        
        data = pd.read_parquet(datapath, engine='auto', columns=None)
        data = data.sample(frac=1.0, axis=0, random_state=SEED).reset_index(drop=True)


    # -----------------------------------------------
    #       Train-test split
    # -----------------------------------------------
    mltype = 'reg'
    te_splitter = cv_splitter(cv_method=te_method, cv_folds=1, test_size=te_size,
                              mltype=mltype, shuffle=True, random_state=SEED)
    if te_method=='simple':
        te_groups = None
    elif te_method=='group':
        te_groups = data['CELL'].copy()

    if is_string_dtype(te_groups):
        grp_enc = LabelEncoder()
        te_groups = grp_enc.fit_transform(te_groups)
   
    # Split train/test
    tr_id, te_id = next(te_splitter.split(data, groups=te_groups))
    tr_data = data.iloc[tr_id, :]  
    te_data = data.iloc[te_id, :] 

    # # Confirm that group splits are correct ...
    tr_grps_unq = set(tr_data['CELL'])
    te_grps_unq = set(te_data['CELL'])
    lg.logger.info(f'\nTotal group (cell) intersections btw tr and te: {len(tr_grps_unq.intersection(te_grps_unq))}.')
    lg.logger.info(f'A few intersections : {list(tr_grps_unq.intersection(te_grps_unq))[:3]}.')
    
    # Save train/test
    tr_data.to_parquet(run_outdir/'tr_data.parquet', engine='auto', compression='snappy')
    te_data.to_parquet(run_outdir/'te_data.parquet', engine='auto', compression='snappy')


    # -----------------------------------------------
    #       Define CV split
    # -----------------------------------------------
    cv = cv_splitter(cv_method=cv_method, cv_folds=cv_folds, test_size=0.2,
                     mltype=mltype, shuffle=True, random_state=SEED)
    if cv_method=='simple':
        cv_groups = None
    elif cv_method=='group':
        cv_groups = tr_data['CELL'].copy()


    # -----------------------------------------------
    #       Generate splits
    # -----------------------------------------------
    if is_string_dtype(cv_groups):
        grp_enc = LabelEncoder()
        cv_groups = grp_enc.fit_transform(cv_groups)
    
    tr_id_all = {} 
    vl_id_all = {} 

    # Start CV iters
    for fold_id, (tr_id, vl_id) in enumerate(cv.split(tr_data, groups=cv_groups)):
        if lg.logger is not None:
            lg.logger.info(f'\nFold {fold_id+1}/{cv_folds}')

        tr_id_all[fold_id] = tr_id.tolist()
        vl_id_all[fold_id] = vl_id.tolist()

        # Confirm that group splits are correct
        tr_grps_unq = set(cv_groups[tr_id])
        vl_grps_unq = set(cv_groups[vl_id])
        lg.logger.info(f'   Total group (cell) intersections btw tr and vl: {len(tr_grps_unq.intersection(vl_grps_unq))}.')
        # lg.logger.info(f'   A few intersections : {list(tr_grps_unq.intersection(vl_grps_unq))[:3]}.')
        lg.logger.info(f'    Unique cell lines in tr: {len(tr_grps_unq)}.')
        lg.logger.info(f'    Unique cell lines in vl: {len(vl_grps_unq)}.')

    tr_df = pd.DataFrame.from_dict(tr_id_all, orient='index').T
    vl_df = pd.DataFrame.from_dict(vl_id_all, orient='index').T
    tr_df.to_csv(run_outdir/'tr_id.csv', index=False)
    vl_df.to_csv(run_outdir/'vl_id.csv', index=False)

    lg.kill_logger()
    print('Done.')


def main(args):
    parser = argparse.ArgumentParser(description="Generate and save dataset splits.")

    # Select data name
    parser.add_argument('--dname',
        default='combined', choices=['combined', 'top6'],
        help='Data name (combined (default) or top6).')

    # Select (cell line) sources 
    parser.add_argument('-src', '--src_names', nargs='+',
        default=['ccle'], choices=['ccle', 'gcsi', 'gdsc', 'ctrp', 'nci60'],
        help='Data sources to use.')

    # Select target to predict
    parser.add_argument('-t', '--target_name',
        default='AUC', choices=['AUC', 'AUC1', 'IC50'],
        help='Column name of the target variable.')

    # Select feature types
    parser.add_argument('-cf', '--cell_features', nargs='+',
        default=['rna'], choices=['rna', 'cnv', 'clb'],
        help='Cell line features.') # ['rna_latent']
    parser.add_argument('-df', '--drug_features', nargs='+',
        default=['dsc'], choices=['dsc', 'fng', 'dlb'],
        help='Drug features.') # ['fng', 'dsc_latent', 'fng_latent']
    parser.add_argument('-of', '--other_features',
        default=[], choices=[],
        help='Other feature types (derived from cell lines and drugs). E.g.: cancer type, etc).') # ['cell_labels', 'drug_labels', 'ctype', 'csite', 'rna_clusters']

    # Select CV scheme
    parser.add_argument('-tem', '--te_method',
        default='simple', choices=['simple', 'group'],
        help='Test split method.')
    parser.add_argument('--te_size', type=float,
        default=0.1, 
        help='Test size split ratio.')
    parser.add_argument('-cvm', '--cv_method',
        default='simple', choices=['simple', 'group'],
        help='Cross-val split method.')
    parser.add_argument('-cvf', '--cv_folds', type=int,
        default=5, help='Number cross-val folds.')

    # Define n_jobs
    parser.add_argument('--n_jobs', default=4,  type=int)

    # Parse args and run
    args = parser.parse_args(args)
    args = vars(args)
    ret = run(args)
    
    
if __name__ == '__main__':
    """ __name__ == '__main__' explained: www.youtube.com/watch?v=sugvnHA7ElY """
    main(sys.argv[1:])

