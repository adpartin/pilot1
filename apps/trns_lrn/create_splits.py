"""
Create tidy df from raw dfs of cell features, drug features, and drug responses.
Yitan provided the raw dfs and the data splits.
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
from glob import glob

import sklearn
import numpy as np
import pandas as pd

from pandas.api.types import is_string_dtype
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

SEED = 42


# File path
file_path = Path(__file__).resolve().parent


# Utils
#utils_path = file_path / '../../utils'
#sys.path.append(str(utils_path))
#import utils
#from classlogger import Logger
import utils

utils_path = file_path / '../../utils'
sys.path.append(str(utils_path))
#import utils
from classlogger import Logger
from cv_splitter import cv_splitter, plot_ytr_yvl_dist


# Path
#OUTDIR = file_path/'../../data/yitan/Data/tidy'
OUTDIR = file_path/'../../data/yitan/CCL_10Fold_Partition'


def parse_args(args):
    parser = argparse.ArgumentParser(description="Create tidy df from raw (Yitan's) data.")
    parser.add_argument('--src', default='GDSC', type=str, choices=['CCLE', 'CTRP', 'gCSI', 'GDSC', 'NCI60', 'all'], 
                        help='Data source (default: GDSC).')
    # parser.add_argument('--fold', default=0, type=int, help='Fold (default: 0).')
    
    parser.add_argument('--drg_subset', default='all', choices=['pdm', 'common', 'all'],
                        help='Drug subset to use for training (default: all).')

    parser.add_argument('-cf', '--ccl_fea', nargs='+', default=['geneGE'], choices=['geneGE'],
                        help='Cell line features (default: `geneGE`).')
    parser.add_argument('-df', '--drg_fea', nargs='+', default=['DD'], choices=['DD'],
                        help='Drug features (default: `DD`).')
    
    parser.add_argument('-dk', '--drg_pca_k', default=None, type=int, help='Number of drug PCA components (default: None).')
    parser.add_argument('-ck', '--ccl_pca_k', default=None, type=int, help='Number of cell PCA components (default: None).')
    
    parser.add_argument('--n_jobs', default=4, type=int, help='Number of cpu workers (default: 4).')
    
    args = parser.parse_args(args)
    return args


def create_outdir(outdir, args):
    #if any([True for i in [args['ccl_pca_k'], args['drg_pca_k']] if i is not None]): 
    #    l = [args['src']] + ['drg_'+args['drg_subset']] + args['ccl_fea'] + args['drg_fea'] + [f'drg_pca{drg_pca_k}'] + [f'ccl_pca{ccl_pca_k}']
    #else:
    #    l = [args['src']] + ['drg_'+args['drg_subset']] + args['ccl_fea'] + args['drg_fea']

    l =  [args['src']] 

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
    drg_subset = args['drg_subset']

    n_jobs = args['n_jobs']
    drg_pca_k = args['drg_pca_k']
    ccl_pca_k = args['ccl_pca_k']    
    fea_sep = '_'
     
    
    # =====================================================
    #       Logger
    # =====================================================
    run_outdir = create_outdir(OUTDIR, args)
    lg = Logger(run_outdir/'logfile.log')
    lg.logger.info(f'File path: {file_path}')
    lg.logger.info(f'\n{pformat(args)}')

    # Dump args to file
    utils.dump_dict(args, run_outdir/'args.txt')      
    
    
    # =====================================================
    #       Load data
    # =====================================================
    data = utils.load_data(file_path, src=src,
            ccl_fea_list=ccl_fea_list, drg_fea_list=drg_fea_list,
            drg_subset=drg_subset,
            fea_sep=fea_sep, logger=lg.logger)

    X, Y, meta = utils.extract_data(data, fea_list=ccl_fea_list+drg_fea_list)

    idx_vec = np.random.permutation(data.shape[0])
    idx_vec_ = idx_vec

    cv_folds_list = [10]
    mltype = 'reg'
    grp_by_col = 'cclname'
    #te_method = 'group'
    cv_method = 'group'
    #te_size = int(X.shape[0]*0.2)  # 0.2
    vl_size = None # int(X.shape[0]*0.2)  # 0.2


    # -----------------------------------------------
    #       Generate CV splits
    # -----------------------------------------------
    # cv_folds_list = [1, 5, 7, 10, 15, 20]
    lg.logger.info(f'\nStart CV splits ...')

    for cv_folds in cv_folds_list:
        lg.logger.info(f'\nCV folds: {cv_folds}')

        # -----------------------------------------------
        #te_splitter = cv_splitter(cv_method=te_method, cv_folds=1, test_size=te_size,
        #                          mltype=mltype, shuffle=False, random_state=SEED)

        #te_grp = meta[grp_by_col].values[idx_vec] if te_method=='group' else None
        #if is_string_dtype(te_grp): te_grp = LabelEncoder().fit_transform(te_grp)

        ## Split train/test
        #tr_id, te_id = next(te_splitter.split(idx_vec, groups=te_grp))
        #tr_id = idx_vec[tr_id] # adjust the indices!
        #te_id = idx_vec[te_id] # adjust the indices!

        ##pd.Series(tr_id).to_csv(run_outdir/f'tr_id.csv', index=False, header=False)
        ##pd.Series(te_id).to_csv(run_outdir/f'te_id.csv', index=False, header=False)

        ## Update the master idx vector for the CV splits
        #idx_vec_ = tr_id

        ## Confirm that group splits are correct
        #if te_method=='group' and grp_by_col is not None:
        #    tr_grp_unq = set(meta.loc[tr_id, grp_by_col])
        #    te_grp_unq = set(meta.loc[te_id, grp_by_col])
        #    lg.logger.info(f'\tTotal group ({grp_by_col}) intersections btw tr and te: {len(tr_grp_unq.intersection(te_grp_unq))}.')
        #    lg.logger.info(f'\tUnique cell lines in tr: {len(tr_grp_unq)}.')
        #    lg.logger.info(f'\tUnique cell lines in vl: {len(te_grp_unq)}.')
        #    
        #del tr_id, te_id, te_grp_unq, vl_grp_unq
        # -----------------------------------------------

        cv = cv_splitter(cv_method=cv_method, cv_folds=cv_folds, test_size=vl_size,
                         mltype=mltype, shuffle=False, random_state=SEED)

        cv_grp = meta[grp_by_col].values[idx_vec_] if cv_method=='group' else None
        if is_string_dtype(cv_grp): cv_grp = LabelEncoder().fit_transform(cv_grp)
    
        tr_folds = {} 
        vl_folds = {} 

        # Start CV iters
        for fold, (tr_id, vl_id) in enumerate(cv.split(idx_vec_, groups=cv_grp)):
            tr_id = idx_vec_[tr_id] # adjust the indices!
            vl_id = idx_vec_[vl_id] # adjust the indices!

            tr_folds[fold] = tr_id.tolist()
            vl_folds[fold] = vl_id.tolist()

            # Confirm that group splits are correct
            if cv_method=='group' and grp_by_col is not None:
                tr_grp_unq = set(meta.loc[tr_id, grp_by_col])
                vl_grp_unq = set(meta.loc[vl_id, grp_by_col])
                lg.logger.info(f'\tTotal group ({grp_by_col}) intersections btw tr and vl: {len(tr_grp_unq.intersection(vl_grp_unq))}.')
                lg.logger.info(f'\tUnique cell lines in tr: {len(tr_grp_unq)}.')
                lg.logger.info(f'\tUnique cell lines in vl: {len(vl_grp_unq)}.')

            # Save ccl lists as per Yitan's scheme
            os.makedirs(run_outdir/f'cv_{fold}')
            pd.Series(list(tr_grp_unq)).to_csv(run_outdir/f'cv_{fold}'/'TrainList.txt', index=False) 
            pd.Series(list(vl_grp_unq)).to_csv(run_outdir/f'cv_{fold}'/'ValList.txt', index=False) 
        
        # Convet to df
        # from_dict takes too long  -->  stackoverflow.com/questions/19736080/
        # tr_folds = pd.DataFrame.from_dict(tr_folds, orient='index').T 
        # vl_folds = pd.DataFrame.from_dict(vl_folds, orient='index').T
        tr_folds = pd.DataFrame(dict([ (k, pd.Series(v)) for k, v in tr_folds.items() ]))
        vl_folds = pd.DataFrame(dict([ (k, pd.Series(v)) for k, v in vl_folds.items() ]))

        # Dump
        tr_folds.to_csv(run_outdir/f'{cv_folds}fold_tr_id.csv', index=False)
        vl_folds.to_csv(run_outdir/f'{cv_folds}fold_vl_id.csv', index=False)
    # ---------------------------------------------------------------------------

    #datadir = Path(file_path/'../../data/yitan/Data')
    #ccl_folds_dir = Path(file_path/'../../data/yitan/CCL_10Fold_Partition')
    #pdm_folds_dir = Path(file_path/'../../data/yitan/PDM_10Fold_Partition')
    #fea_data_name = 'CCL_PDM_TransferLearningData_rmFactor_0.0_ddNorm_std.pkl'
    ##fea_data_name = 'CCL_PDM_TransferLearningData_rmFactor_0.1_ddNorm_quantile.pkl'
    #
    ## Un-pickle files
    #import _pickle as cp
    #pkl_file = open(datadir/fea_data_name, 'rb')
    #res = cp.load(pkl_file)
    #ccl = cp.load(pkl_file)
    #drg = cp.load(pkl_file)
    #pkl_file.close()

    #lg.logger.info('\n{}'.format('=' * 50))
    #lg.logger.info('res: {}'.format(res.shape))
    #lg.logger.info('ccl: {}'.format(ccl.shape))
    #lg.logger.info('drg: {}'.format(drg.shape))
    #
    ## Update resp
    #res = res.reset_index()
    #res = res.rename(columns={'index': 'idx', 'ccl_name': 'cclname', 'SOURCE': 'src', 'area_under_curve': 'auc'})    
    #
    #tmp = res.groupby('src').agg({'cclname': 'nunique', 'ctrpDrugID': 'nunique'}).reset_index()
    #lg.logger.info(tmp)

    #
    ## =====================================================
    ##       Prepare dfs for merging
    ## =====================================================
    ## Retain specific source
    #lg.logger.info('\nFull dataset: {}'.format(res.shape))
    #res = res[ res['src'].isin([src]) ]
    #lg.logger.info('Only {}: {}'.format(src, res.shape))

    ## Extract specific types of features
    #lg.logger.info('\nExtract fea types ...')
    #cnt_fea(ccl, logger=lg.logger);
    #ccl = extract_subset_fea(df=ccl, fea_list=ccl_fea_list, fea_sep=fea_sep)
    #cnt_fea(ccl, logger=lg.logger);

    #cnt_fea(drg, logger=lg.logger);
    #drg = extract_subset_fea(df=drg, fea_list=drg_fea_list, fea_sep=fea_sep)
    #cnt_fea(drg, logger=lg.logger);
    
    # ---------------------------------------------
    #     PCA 
    # ---------------------------------------------
    #def pca_exp_var(pca, k):
    #    return np.sum(pca.explained_variance_ratio_[:k])
    #
    #def dump_pca_data(datadir, pca_x, k, fea_name):
    #    df = pd.DataFrame(pca_x.iloc[:, :k])
    #    df.to_csv(datadir/f'{fea_name}_pca{k}.csv', index=False)    
    #
    #def pca_on_fea(df, fea_name, k, datadir):
    #    lg.logger.info(f'Calc PCA for {fea_name} features.')
    #    index = df.index
    #    n = 1400
    #    pca = PCA(n_components=n, random_state=SEED, svd_solver='auto')
    #    pca_x = pca.fit_transform(df)
    #    pca_x = StandardScaler().fit_transform(pca_x)
    #    pca_x = pd.DataFrame(pca_x, index=index,
    #                         columns=[f'{fea_name}_PC'+str(i+1) for i in range(n)])
    #    exp_var = pca_exp_var(pca, k)
    #    # dump_pca_data(datadir, pca_x=pca_x, k=k, fea_name=fea_name);
    #    lg.logger.info(f'{fea_name} PCA variance explained {exp_var:.4f} (k={k}).')
    #    return pca_x.iloc[:, :k]
    #    
    #if drg_pca_k is not None:
    #    ccl = pca_on_fea(df=ccl, fea_name='ccl', k=ccl_pca_k, datadir=datadir)

    #if ccl_pca_k is not None:
    #    drg = pca_on_fea(df=drg, fea_name='drg', k=drg_pca_k, datadir=datadir)
    #        
    ## Bring the labels in, in order to merge on
    #ccl = ccl.reset_index().rename(columns={'index': 'cclname'})
    #drg = drg.reset_index().rename(columns={'index': 'ctrpDrugID'})


    # =====================================================
    #       Merge
    # =====================================================
    #def merge_dfs(res_df, ccl_df, drg_df):
    #    """ Merge the following dfs: response, ccl fea, drug fea """
    #    mrg_df = pd.merge(res_df, ccl_df, on='cclname', how='inner')
    #    mrg_df = pd.merge(mrg_df, drg_df, on='ctrpDrugID', how='inner')
    #    return mrg_df

    #lg.logger.info('\nMerge ...')
    #mrg_df = merge_dfs(res, ccl, drg)
    #lg.logger.info('mrg_df.shape: {}'.format(mrg_df.shape))


    # =====================================================
    #       Dump to file
    # =====================================================
    #if any([True for i in [ccl_pca_k, drg_pca_k] if i is not None]): 
    #    prfx = '.'.join( [src.lower()] + ccl_fea_list + drg_fea_list + [f'drg_pca{drg_pca_k}'] + [f'ccl_pca{ccl_pca_k}'] )
    #else:
    #    prfx = '.'.join( [src.lower()] + ccl_fea_list + drg_fea_list )
    #
    #mrg_df.to_parquet(run_outdir/(prfx+'_data.parquet'), index=False)
    #lg.logger.info('mrg_df usage: {:.2f} GB\n'.format(sys.getsizeof(mrg_df)/1e9))
    
    lg.kill_logger()


def main(args):
    args = parse_args(args)
    args = vars(args)
    ret = run(args)
    

if __name__ == '__main__':
    """ __name__ == '__main__' explained: www.youtube.com/watch?v=sugvnHA7ElY """
    main(sys.argv[1:])

