import warnings
warnings.filterwarnings('ignore')

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from collections import OrderedDict

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

SEED = 42


# Args
src = 'GDSC'
fold = 0
ccl_fea_list = ['geneGE']
drg_fea_list = ['DD']
fea_sep = '_'
# fea_float_dtype = fea_float_dtype


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


def cnt_fea(df, fea_sep='_', verbose=True):
    """ Count the number of features per feature type. """
    dct = {}
    unq_prfx = df.columns.map(lambda x: x.split(fea_sep)[0]).unique() # unique feature prefixes
    for prfx in unq_prfx:
        fea_type_cols = [c for c in df.columns if (c.split(fea_sep)[0]) in prfx] # all fea names of specific type
        dct[prfx] = len(fea_type_cols)
    if verbose: print(dct)
    return dct


def extract_subset_fea(df, fea_list, fea_sep='_'):
    """ Extract features based feature prefix name. """
    fea = [c for c in df.columns if (c.split(fea_sep)[0]) in fea_list]
    df = df[fea]
    return df


def run(args):
    # Args
    # framework = args['frm'] # 'pytorch'
    # data_name = args['dname'] # 'top6'
    src = args['src']
    # fold = args['fold']
    ccl_fea_list = args['ccl_fea'] # ['geneGE']
    drg_fea_list = args['drg_fea'] # ['DD']
    n_jobs = args['n_jobs']
    
    
    # =====================================================
    # Load data
    # =====================================================
    datadir = Path('../../data/yitan/Data')
    ccl_folds_dir = Path('../../data/yitan/CCL_10Fold_Partition')
    pdm_folds_dir = Path('../../data/yitan/PDM_10Fold_Partition')

    # Un-pickle files
    import _pickle as cp
    pkl_file = open(datadir/'CCL_PDM_TransferLearningData_rmFactor_0.0_ddNorm_std.pkl', 'rb')

    res = cp.load(pkl_file)
    ccl = cp.load(pkl_file)
    drg = cp.load(pkl_file)
    pkl_file.close()

    print('res ', res.shape)
    print('ccl ', ccl.shape)
    print('drg ', drg.shape)
    print(res.groupby('SOURCE').agg({'ccl_name': 'nunique', 'ctrpDrugID': 'nunique'}).reset_index())

    # Update resp
    res = res.reset_index()
    res = res.rename(columns={'index': 'idx', 'SOURCE': 'src', 'area_under_curve': 'auc'})


    # =====================================================
    # Prepare dfs for merging
    # =====================================================
    res_df_ = res.copy()
    ccl_df_ = ccl.copy()
    drg_df_ = drg.copy()

    # Get lists of ccl names based on src and fold
    ids_path = ccl_folds_dir/f'{src}/cv_{fold}' # 'TestList.txt'
    tr_ids_list = pd.read_csv(ids_path/'TrainList.txt', header=None).squeeze().values
    vl_ids_list = pd.read_csv(ids_path/'ValList.txt', header=None).squeeze().values
    te_ids_list = pd.read_csv(ids_path/'TestList.txt', header=None).squeeze().values

    # Show how much is left for train, val, and test
    tr_sz, vl_sz, te_sz = len(tr_ids_list), len(vl_ids_list), len(te_ids_list)
    sz = tr_sz + vl_sz + te_sz
    print(f'\nTrain portion: {tr_sz/sz:.2f}')

    # Retain specific source
    print('Full dataset:', res_df_.shape)
    res_df_ = res_df_[ res_df_['src'].isin([src]) ]
    print(f'Only {src}: ', res_df_.shape)

    # Create res dfs for train and val
    res_tr = res_df_[ res_df_['ccl_name'].isin( tr_ids_list ) ]
    res_vl = res_df_[ res_df_['ccl_name'].isin( vl_ids_list ) ]
    print('\nres_tr.shape:', res_tr.shape)
    print('res_vl.shape:', res_vl.shape)

    # Extract specific types of features
    print('\nExtract fea types ...')
    cnt_fea(ccl_df_);
    ccl_df_ = extract_subset_fea(df=ccl_df_, fea_list=ccl_fea_list, fea_sep=fea_sep)
    cnt_fea(ccl_df_);

    cnt_fea(drg_df_);
    drg_df_ = extract_subset_fea(df=drg_df_, fea_list=drg_fea_list, fea_sep=fea_sep)
    cnt_fea(drg_df_);

    # Bring the labels in, in order to merge on
    ccl_df_ = ccl_df_.reset_index().rename(columns={'index': 'ccl_name'})
    drg_df_ = drg_df_.reset_index().rename(columns={'index': 'ctrpDrugID'})


    # =====================================================
    # Now merge
    # =====================================================
    def merge_dfs(res_df, ccl_df, drg_df):
        """ Merge the following dfs: response, ccl fea, drug fea """
        mrg_df = pd.merge(res_df, ccl_df, on='ccl_name', how='inner')
        mrg_df = pd.merge(mrg_df, drg_df, on='ctrpDrugID', how='inner')
        return mrg_df

    print('\nMerge ...')
    mrg_tr = merge_dfs(res_tr, ccl_df_, drg_df_)
    mrg_vl = merge_dfs(res_vl, ccl_df_, drg_df_)
    print('mrg_tr.shape:', mrg_tr.shape)
    print('mrg_vl.shape:', mrg_vl.shape)


    # =====================================================
    # Extract tr and vl and dump to file
    # =====================================================
    # Extract tr and vl
    xtr = extract_subset_fea(mrg_tr, fea_list = ccl_fea_list + drg_fea_list, fea_sep=fea_sep)
    ytr = mrg_tr[['auc']]

    xvl = extract_subset_fea(mrg_vl, fea_list = ccl_fea_list + drg_fea_list, fea_sep=fea_sep)
    yvl = mrg_vl[['auc']]

    print('\nExtract x and y ...')
    print('xtr.shape:', xtr.shape)
    print('xvl.shape:', xvl.shape)
    print('ytr.shape:', ytr.shape)
    print('yvl.shape:', yvl.shape)

    # Dump train data to file
    xtr.to_parquet(datadir/f'{src.lower()}_xtr.parquet', index=False)
    ytr.to_parquet(datadir/f'{src.lower()}_ytr.parquet', index=False)

    # Dump val data to file
    xvl.to_parquet(datadir/f'{src.lower()}_xvl.parquet', index=False)
    yvl.to_parquet(datadir/f'{src.lower()}_yvl.parquet', index=False)


def main(args):
    args = parse_args(args)
    ret = run(args)
    

if __name__ == '__main__':
    """ __name__ == '__main__' explained: www.youtube.com/watch?v=sugvnHA7ElY """
    main(sys.argv[1:])