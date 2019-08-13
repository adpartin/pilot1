"""
Create data partitions.
"""
from __future__ import print_function, division

import warnings
warnings.filterwarnings('ignore')

import os
import sys
from pathlib import Path
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
import utils

utils_path = file_path / '../../utils'
sys.path.append(str(utils_path))
from classlogger import Logger
from cv_splitter import cv_splitter, plot_ytr_yvl_dist


# Path
OUTDIR = file_path/'../../data/yitan/CCL_10Fold_Partition'


def parse_args(args):
    parser = argparse.ArgumentParser(description="Create tidy df from raw (Yitan's) data.")
    parser.add_argument('--src', default='GDSC', type=str, choices=['CCLE', 'CTRP', 'gCSI', 'GDSC', 'NCI60', 'all'], 
                        help='Data source (default: GDSC).')
    
    parser.add_argument('--drg_subset', default='all', choices=['pdm', 'common', 'all'],
                        help='Drug subset to use for training (default: all).')

    parser.add_argument('-cf', '--ccl_fea', nargs='+', default=['geneGE'], choices=['geneGE'],
                        help='Cell line features (default: geneGE).')
    parser.add_argument('-df', '--drg_fea', nargs='+', default=['DD'], choices=['DD'],
                        help='Drug features (default: DD).')
    
    parser.add_argument('--cv_method', default='group', type=str, help='Split method btw tr and vl (default: group).')
    parser.add_argument('--te_method', default='simple', type=str, help='Split method btw vl and te (default: simple).')

    parser.add_argument('-dk', '--drg_pca_k', default=None, type=int, help='Number of drug PCA components (default: None).')
    parser.add_argument('-ck', '--ccl_pca_k', default=None, type=int, help='Number of cell PCA components (default: None).')
    
    parser.add_argument('--n_jobs', default=4, type=int, help='Number of cpu workers (default: 4).')
    
    args = parser.parse_args(args)
    return args


def create_outdir(outdir, args):
    name_sffx = args['src']
    outdir = Path(outdir)/name_sffx
    os.makedirs(outdir)
    return outdir


def run(args):
    src = args['src']
    ccl_fea_list = args['ccl_fea']
    drg_fea_list = args['drg_fea']
    drg_subset = args['drg_subset']

    cv_method = args['cv_method']
    te_method = args['te_method']

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

    cv_folds_list = [5]
    mltype = 'reg'
    grp_by_col = 'cclname'
    vl_size = None # int(X.shape[0]*0.2)  # 0.2


    # =====================================================
    #       Generate CV splits
    # =====================================================
    # cv_folds_list = [1, 5, 7, 10, 15, 20]
    lg.logger.info(f'\nStart CV splits ...')

    for cv_folds in cv_folds_list:
        lg.logger.info(f'\nCV folds: {cv_folds}')

        cv = cv_splitter(cv_method=cv_method, cv_folds=cv_folds, test_size=vl_size,
                         mltype=mltype, shuffle=False, random_state=SEED)

        cv_grp = meta[grp_by_col].values[idx_vec] if cv_method=='group' else None
        if is_string_dtype(cv_grp): cv_grp = LabelEncoder().fit_transform(cv_grp)
    
        tr_folds = {} 
        vl_folds = {} 
        te_folds = {}
        
        # Start CV iters
        for fold, (tr_id, vl_id) in enumerate(cv.split(idx_vec, groups=cv_grp)):
            lg.logger.info(f'\nFold {fold}')
            tr_id = idx_vec[tr_id] # adjust the indices!
            vl_id = idx_vec[vl_id] # adjust the indices!

            tr_folds[fold] = tr_id.tolist()
            # vl_folds[fold] = vl_id.tolist()

            # Split vl set into vl and te
            te_splitter = cv_splitter(cv_method=te_method, cv_folds=1, test_size=0.5,
                                      mltype=mltype, shuffle=False, random_state=SEED)
            idx_vec_ = vl_id; del vl_id

            te_grp = meta[grp_by_col].values[idx_vec_] if te_method=='group' else None
            if is_string_dtype(te_grp): te_grp = LabelEncoder().fit_transform(te_grp)

            # Split train/test
            vl_id, te_id = next(te_splitter.split(idx_vec_, groups=te_grp))
            vl_id = idx_vec_[vl_id] # adjust the indices!
            te_id = idx_vec_[te_id] # adjust the indices!

            vl_folds[fold] = vl_id.tolist()
            te_folds[fold] = te_id.tolist()

            lg.logger.info(f'tr data size {len(tr_id)}')
            lg.logger.info(f'vl data size {len(vl_id)}')
            lg.logger.info(f'te data size {len(te_id)}')

            # Confirm that group splits are correct
            if cv_method=='group' and grp_by_col is not None:
                tr_grp_unq = set(meta.loc[tr_id, grp_by_col])
                vl_grp_unq = set(meta.loc[vl_id, grp_by_col])
                te_grp_unq = set(meta.loc[te_id, grp_by_col])
                lg.logger.info(f'\tTotal group ({grp_by_col}) intersec btw tr and vl: {len(tr_grp_unq.intersection(vl_grp_unq))}.')
                lg.logger.info(f'\tTotal group ({grp_by_col}) intersec btw tr and te: {len(tr_grp_unq.intersection(te_grp_unq))}.')
                lg.logger.info(f'\tTotal group ({grp_by_col}) intersec btw vl and te: {len(vl_grp_unq.intersection(te_grp_unq))}.')
                lg.logger.info(f'\tUnique cell lines in tr: {len(tr_grp_unq)}.')
                lg.logger.info(f'\tUnique cell lines in vl: {len(vl_grp_unq)}.')
                lg.logger.info(f'\tUnique cell lines in te: {len(te_grp_unq)}.')

            # Save ccl lists as per Yitan's scheme
            os.makedirs(run_outdir/f'cv_{fold}')
            pd.Series(list(tr_grp_unq)).to_csv(run_outdir/f'cv_{fold}'/'TrainList.txt', index=False) 
            pd.Series(list(vl_grp_unq)).to_csv(run_outdir/f'cv_{fold}'/'ValList.txt', index=False) 
            pd.Series(list(te_grp_unq)).to_csv(run_outdir/f'cv_{fold}'/'TestList.txt', index=False) 
        
        # Convet to df
        # from_dict takes too long  -->  stackoverflow.com/questions/19736080/
        # tr_folds = pd.DataFrame.from_dict(tr_folds, orient='index').T 
        # vl_folds = pd.DataFrame.from_dict(vl_folds, orient='index').T
        tr_folds = pd.DataFrame(dict([ (k, pd.Series(v)) for k, v in tr_folds.items() ]))
        vl_folds = pd.DataFrame(dict([ (k, pd.Series(v)) for k, v in vl_folds.items() ]))
        te_folds = pd.DataFrame(dict([ (k, pd.Series(v)) for k, v in te_folds.items() ]))

        # Dump
        tr_folds.to_csv(run_outdir/f'{cv_folds}fold_tr_id.csv', index=False)
        vl_folds.to_csv(run_outdir/f'{cv_folds}fold_vl_id.csv', index=False)
        te_folds.to_csv(run_outdir/f'{cv_folds}fold_te_id.csv', index=False)


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

    
    lg.kill_logger()


def main(args):
    args = parse_args(args)
    args = vars(args)
    ret = run(args)
    

if __name__ == '__main__':
    main(sys.argv[1:])

