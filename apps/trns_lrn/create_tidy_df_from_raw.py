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

from sklearn.decomposition import PCA
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
    
    parser.add_argument('-dk', '--drg_pca_k', default=None, type=int, help='Number of drug PCA components (default: None).')
    parser.add_argument('-ck', '--ccl_pca_k', default=None, type=int, help='Number of cell PCA components (default: None).')
    
    parser.add_argument('--n_jobs', default=4, type=int, help='Number of cpu workers (default: 4).')
    
    args = parser.parse_args(args)
    args = vars(args)
    return args


def create_outdir(outdir, args):
    # t = datetime.now()
    # t = [t.year, '-', t.month, '-', t.day, '_', 'h', t.hour, '-', 'm', t.minute]
    # t = ''.join([str(i) for i in t])
    if any([True for i in [args['ccl_pca_k'], args['drg_pca_k']] if i is not None]): 
        l = [args['src']] + args['ccl_fea'] + args['drg_fea'] + [f'drg_pca{drg_pca_k}'] + [f'ccl_pca{ccl_pca_k}']
    else:
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
    drg_pca_k = args['drg_pca_k']
    ccl_pca_k = args['ccl_pca_k']    
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
    
    # Update resp
    res = res.reset_index()
    res = res.rename(columns={'index': 'idx', 'ccl_name': 'cclname', 'SOURCE': 'src', 'area_under_curve': 'auc'})    
    
    tmp = res.groupby('src').agg({'cclname': 'nunique', 'ctrpDrugID': 'nunique'}).reset_index()
    lg.logger.info(tmp)

    
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
    
    def pca_exp_var(pca, k):
        return np.sum(pca.explained_variance_ratio_[:k])
    
    def dump_pca_data(datadir, pca_x, k, fea_name):
        df = pd.DataFrame(pca_x.iloc[:, :k])
        df.to_csv(datadir/f'{fea_name}_pca{k}.csv', index=False)    
    
    def pca_on_fea(df, fea_name, k, datadir):
        lg.logger.info(f'Calc PCA for {fea_name} features.')
        index = df.index
        n = 1400
        pca = PCA(n_components=n, random_state=SEED, svd_solver='auto')
        pca_x = pca.fit_transform(df)
        pca_x = StandardScaler().fit_transform(pca_x)
        pca_x = pd.DataFrame(pca_x, index=index,
                             columns=[f'{fea_name}_PC'+str(i+1) for i in range(n)])
        exp_var = pca_exp_var(pca, k)
        # dump_pca_data(datadir, pca_x=pca_x, k=k, fea_name=fea_name);
        lg.logger.info(f'{fea_name} PCA variance explained {exp_var:.4f} (k={k}).')
        return pca_x.iloc[:, :k]
        
    if drg_pca_k is not None:
        ccl = pca_on_fea(df=ccl, fea_name='ccl', k=ccl_pca_k, datadir=datadir)

    if ccl_pca_k is not None:
        drg = pca_on_fea(df=drg, fea_name='drg', k=drg_pca_k, datadir=datadir)
            
    # Bring the labels in, in order to merge on
    ccl = ccl.reset_index().rename(columns={'index': 'cclname'})
    drg = drg.reset_index().rename(columns={'index': 'ctrpDrugID'})


    # =====================================================
    #       Merge
    # =====================================================
    def merge_dfs(res_df, ccl_df, drg_df):
        """ Merge the following dfs: response, ccl fea, drug fea """
        mrg_df = pd.merge(res_df, ccl_df, on='cclname', how='inner')
        mrg_df = pd.merge(mrg_df, drg_df, on='ctrpDrugID', how='inner')
        return mrg_df

    lg.logger.info('\nMerge ...')
    mrg_df = merge_dfs(res, ccl, drg)
    lg.logger.info('mrg_df.shape: {}'.format(mrg_df.shape))


    # =====================================================
    #       Extract xdata, ydata, meta and  dump to file
    # =====================================================
    """
    xdata = extract_subset_fea(df=mrg_df, fea_list=['cclname']+ccl_fea_list+drg_fea_list, fea_sep=fea_sep)
    ydata = mrg_df[['auc']]
    meta = mrg_df.drop(columns=xdata.columns)
    meta = meta.drop(columns=['auc'])
    
    prfx = '.'.join( [src.lower()] + ccl_fea_list + drg_fea_list )
    xdata.to_parquet(run_outdir/(prfx+'_xdata.parquet'), index=False)
    ydata.to_parquet(run_outdir/(prfx+'_ydata.parquet'), index=False)
    meta.to_parquet( run_outdir/(prfx+'_meta.parquet'),  index=False)

    lg.logger.info('xdata memory usage: {:.2f} GB\n'.format(sys.getsizeof(xdata)/1e9))
    """
    if any([True for i in [ccl_pca_k, drg_pca_k] if i is not None]): 
        prfx = '.'.join( [src.lower()] + ccl_fea_list + drg_fea_list + [f'drg_pca{drg_pca_k}'] + [f'ccl_pca{ccl_pca_k}'] )
    else:
        prfx = '.'.join( [src.lower()] + ccl_fea_list + drg_fea_list )
    
    mrg_df.to_parquet(run_outdir/(prfx+'_data.parquet'), index=False)
    lg.logger.info('mrg_df usage: {:.2f} GB\n'.format(sys.getsizeof(mrg_df)/1e9))
    
    lg.kill_logger()


def main(args):
    args = parse_args(args)
    ret = run(args)
    

if __name__ == '__main__':
    """ __name__ == '__main__' explained: www.youtube.com/watch?v=sugvnHA7ElY """
    main(sys.argv[1:])
