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
from glob import glob

import sklearn
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

SEED = 42


# File path
file_path = Path(__file__).resolve().parent


# Utils
utils_path = file_path / '../../utils'
sys.path.append(str(utils_path))
import utils
from classlogger import Logger


# Path
OUTDIR = file_path/'../../data/yitan/Data/tidy'


def parse_args(args):
    # Args
    parser = argparse.ArgumentParser(description="Create tidy df from raw (Yitan) data.")
    parser.add_argument('--src', default='GDSC', type=str, help='Data source (default: GDSC).')
    # parser.add_argument('--fold', default=0, type=int, help='Fold (default: 0).')
    
    parser.add_argument('-cf', '--ccl_fea', nargs='+', default=['geneGE'], choices=['geneGE'],
                        help='Cell line features (default: `geneGE`).')
    parser.add_argument('-df', '--drg_fea', nargs='+', default=['DD'], choices=['DD'],
                        help='Drug features (default: `DD`).')
    
    parser.add_argument('--n_jobs', default=4, type=int, help='Number of cpu workers (default: 4).')
    
    args = parser.parse_args(args)
    args = vars(args)
    return args


def create_outdir(outdir, args):
    # t = datetime.now()
    # t = [t.year, '-', t.month, '-', t.day, '_', 'h', t.hour, '-', 'm', t.minute]
    # t = ''.join([str(i) for i in t])
    
    l = [args['src']] + args['ccl_fea'] + args['drg_fea']
                
    name_sffx = '.'.join( l )
    # outdir = Path(outdir) / (name_sffx + '_' + t)
    outdir = Path(outdir)/name_sffx
    os.makedirs(outdir)
    return outdir


def cnt_fea(df, fea_sep='_', verbose=True, logger=None):
    """ Count the number of features per feature type. """
    dct = {}
    unq_prfx = df.columns.map(lambda x: x.split(fea_sep)[0]).unique() # unique feature prefixes
    for prfx in unq_prfx:
        fea_type_cols = [c for c in df.columns if (c.split(fea_sep)[0]) in prfx] # all fea names of specific type
        dct[prfx] = len(fea_type_cols)
    
    if verbose and logger is not None:
        logger.info(pformat(dct))
    elif verbose:
        pprint(dct)
        
    return dct


def extract_subset_fea(df, fea_list, fea_sep='_'):
    """ Extract features based feature prefix name. """
    fea = [c for c in df.columns if (c.split(fea_sep)[0]) in fea_list]
    df = df[fea]
    return df


def run(args):
    src = args['src']
    ccl_fea_list = args['ccl_fea']
    drg_fea_list = args['drg_fea']
    n_jobs = args['n_jobs']
    fea_sep = '_'
    
    
    # =====================================================
    #       Logger
    # =====================================================
    run_outdir = create_outdir(OUTDIR, args)
    logfilename = run_outdir/'logfile.log'
    lg = Logger(logfilename)
    lg.logger.info(datetime.now())
    lg.logger.info(f'\nFile path: {file_path}')
    lg.logger.info(f'Machine: {platform.node()} ({platform.system()}, {psutil.cpu_count()} CPUs)')
    lg.logger.info(f'\n{pformat(args)}')

    # Dump args to file
    utils.dump_args(args, run_outdir)     
    
    
    # =====================================================
    #       Load data
    # =====================================================
    datadir = Path(file_path/'../../data/yitan/Data')
    ccl_folds_dir = Path(file_path/'../../data/yitan/CCL_10Fold_Partition')
    pdm_folds_dir = Path(file_path/'../../data/yitan/PDM_10Fold_Partition')
    fea_data_name = 'CCL_PDM_TransferLearningData_rmFactor_0.0_ddNorm_std.pkl'
    
    # Un-pickle files
    import _pickle as cp
    pkl_file = open(datadir/fea_data_name, 'rb')
    res = cp.load(pkl_file)
    ccl = cp.load(pkl_file)
    drg = cp.load(pkl_file)
    pkl_file.close()

    lg.logger.info('\n{}'.format('=' * 50))
    lg.logger.info('res: {}'.format(res.shape))
    lg.logger.info('ccl: {}'.format(ccl.shape))
    lg.logger.info('drg: {}'.format(drg.shape))
    
    tmp = res.groupby('SOURCE').agg({'ccl_name': 'nunique', 'ctrpDrugID': 'nunique'}).reset_index()
    lg.logger.info(tmp)
    
    # Update resp
    res = res.reset_index()
    res = res.rename(columns={'index': 'idx', 'SOURCE': 'src', 'area_under_curve': 'auc'})

    
    # =====================================================
    #       Prepare dfs for merging
    # =====================================================
    # Retain specific source
    lg.logger.info('\nFull dataset: {}'.format(res.shape))
    res = res[ res['src'].isin([src]) ]
    lg.logger.info('Only {}: {}'.format(src, res.shape))

    # Extract specific types of features
    lg.logger.info('\nExtract fea types ...')
    cnt_fea(ccl, logger=lg.logger);
    ccl = extract_subset_fea(df=ccl, fea_list=ccl_fea_list, fea_sep=fea_sep)
    cnt_fea(ccl, logger=lg.logger);

    cnt_fea(drg, logger=lg.logger);
    drg = extract_subset_fea(df=drg, fea_list=drg_fea_list, fea_sep=fea_sep)
    cnt_fea(drg, logger=lg.logger);

    # Bring the labels in, in order to merge on
    ccl = ccl.reset_index().rename(columns={'index': 'ccl_name'})
    drg = drg.reset_index().rename(columns={'index': 'ctrpDrugID'})


    # =====================================================
    #       Merge
    # =====================================================
    def merge_dfs(res_df, ccl_df, drg_df):
        """ Merge the following dfs: response, ccl fea, drug fea """
        mrg_df = pd.merge(res_df, ccl_df, on='ccl_name', how='inner')
        mrg_df = pd.merge(mrg_df, drg_df, on='ctrpDrugID', how='inner')
        return mrg_df

    lg.logger.info('\nMerge ...')
    mrg = merge_dfs(res, ccl, drg)
    lg.logger.info('mrg.shape: {}'.format(mrg.shape))


    # =====================================================
    #       Extract xdata, ydata, meta and  dump to file
    # =====================================================
    xdata = extract_subset_fea(df=mrg, fea_list=ccl_fea_list+drg_fea_list, fea_sep=fea_sep)
    ydata = mrg[['auc']]
    meta = mrg.drop(columns=xdata.columns)
    meta = meta.drop(columns=['auc'])
    
    prfx = '.'.join( [src.lower()] + ccl_fea_list + drg_fea_list )
    xdata.to_parquet(run_outdir/(prfx+'_xdata.parquet'), index=False)
    ydata.to_parquet(run_outdir/(prfx+'_ydata.parquet'), index=False)
    meta.to_parquet( run_outdir/(prfx+'_meta.parquet'),  index=False)
    
    lg.logger.info('xdata memory usage: {:.2f} GB\n'.format(sys.getsizeof(xdata)/1e9))
    lg.kill_logger()


def main(args):
    args = parse_args(args)
    ret = run(args)
    

if __name__ == '__main__':
    """ __name__ == '__main__' explained: www.youtube.com/watch?v=sugvnHA7ElY """
    main(sys.argv[1:])
