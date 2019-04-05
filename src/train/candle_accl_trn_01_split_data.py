from __future__ import print_function, division

import warnings
warnings.filterwarnings('ignore')

import os
import sys
from pathlib import Path
import argparse
import datetime
from time import time
from pprint import pprint

import sklearn
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit, KFold
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

SEED = None
t_start = time()


# Utils
import classlogger
import utils


# File path
# file_path = os.path.dirname(os.path.realpath(__file__))
file_path = Path(__file__).resolve().parent


# Path - create dir to dump results (AP)
PRJ_NAME = 'candle_accl_trn'
OUTDIR = Path(file_path / '../../models' / PRJ_NAME / 'data')
os.makedirs(OUTDIR, exist_ok=True)


# Arg parser
psr = argparse.ArgumentParser(description='input agg csv file')
psr.add_argument('--in', default=None, help='Specify the input dataset.')
# psr.add_argument('--split_method', type=str, choices=['rnd', 'hrd'], default='hrd')
psr.add_argument('--split_by', type=str, choices=['cell', 'drug', 'both', 'none'], default='cell',
                 help='Specify how to disjointly partition the dataset: \
                 `cell` (disjoint on cell), `drug` (disjoint on drug), \
                 `both` (disjoint on cell and drug), `none` (random split).')
psr.add_argument('--ratio', type=int, default=0.5)

args = vars(psr.parse_args())
pprint(args)

# Args
data_path = args['in']
split_by = args['split_by']  # applied with hard partition
wrm_ratio = args['ratio']
test_size = 1 - wrm_ratio

# Outdir
# if split_method == 'rnd':
#     outdir = OUTDIR / 'rnd'
# elif split_method == 'hrd':
#     outdir = OUTDIR / 'hrd'
# os.makedirs(outdir, exist_ok=True)

outdir = OUTDIR / ('split_by_' + split_by)
os.makedirs(outdir, exist_ok=True)

# Dump args
utils.dump_args(args, outdir=outdir)

# Logger
logfilename = outdir / 'logging.log'
lg = classlogger.Logger(logfilename=logfilename)


# ---------
# Load data
# ---------
lg.logger.info(f'Loading data ... {data_path}')
t0 = time()
if 'csv' in data_path:
    df = pd.read_csv(data_path, skiprows=1, dtype='float32', nrows=args['nrows']).values
elif 'parquet' in data_path:
    df = pd.read_parquet(data_path, engine='auto', columns=None)
    df = df.sample(frac=1.0, axis=0, random_state=SEED).reset_index(drop=True)
lg.logger.info('Done ({:.2f} mins)\n'.format((time()-t0)/60))


# Drop constant features (AP)
lg.logger.info('Drop constant features ...')
lg.logger.info('Before: {}'.format( df.shape )) 
col_idx = df.nunique(dropna=True).values == 1  # col indexes to drop
df = df.iloc[:, ~col_idx]
lg.logger.info('After:  {}'.format( df.shape ))


# ----------
# Split data
# ----------
def split_cell_drug(dff):
    """ Split drug and cell features. """
    dff = dff.copy()
    dd_cols = [c for c in df.columns if 'DD_' in c]
    ge_cols = [c for c in df.columns if 'GE_' in c]
    dd = dff[dd_cols]
    ge = dff[ge_cols]
    print('\ndd.shape', dd.shape)
    print('ge.shape', ge.shape)
    return dd, ge


def add_lbl_dup(dff, label_name='label', prffx='_'):
    """ Label unique rows. Add column indicating a unique row (label). """
    # Save the original row indexes in order to re-order rows after processing
    idx_org = dff.index.values
    
    # Sort rows (duplicated rows will be concateneted)
    dff = dff.sort_values(by=dff.columns.tolist())
    # Add boolean col indicating the start of new unique row
    dff = pd.concat([dff.duplicated(keep='first'), dff], axis=1).rename(columns={0: 'd'})

    # Add col indicating a unique row
    c = -1
    v = np.ones((len(dff),))
    for i, x in enumerate(dff['d']):
        # if i % 50000 == 0: print(i)
        if x is False:
            c += 1
            v[i] = int(c)
        else:
            v[i] = c

    dff.insert(loc=1, column=label_name, value=v) 
    dff = dff.reindex(idx_org)  # back to the original row ordering
    dff = dff.drop(columns=['d'])
    
    dff[label_name] = dff[label_name].map(lambda x: prffx + str(int(x)))
    return dff


# Determine split indices based on split method
dd, ge = split_cell_drug(dff=df)
dlb = add_lbl_dup(dd, label_name='dlb', prffx='d')['dlb']
clb = add_lbl_dup(ge, label_name='clb', prffx='c')['clb']
lg.logger.info(f'\nUnique drugs: {len(dlb.unique())}')
lg.logger.info(f'Unique cells: {len(clb.unique())}')
del dd, ge

if split_by == 'none':
    # Random split
    cv = ShuffleSplit(n_splits=2, test_size=test_size, random_state=0)
    id_grp1, id_grp2 = next(cv.split(df))
    
else:
    #dd, ge = split_cell_drug(dff=df)
    #dlb = add_lbl_dup(dd, label_name='dlb', prffx='d')['dlb']
    #clb = add_lbl_dup(ge, label_name='clb', prffx='c')['clb']
    # print('Unique drugs:', len(dlb.unique()))
    # print('Unique cells:', len(clb.unique()))
    #del dd, ge
    
    if split_by == 'cell':  # disjoint split by cell
        cv = GroupShuffleSplit(n_splits=2, test_size=test_size, random_state=0)
        id_grp1, id_grp2 = next(cv.split(df, groups=clb))  # Split indexes
    
    elif split_by == 'drug':  # disjoint split by drug
        cv = GroupShuffleSplit(n_splits=2, test_size=test_size, random_state=0)
        id_grp1, id_grp2 = next(cv.split(df, groups=dlb))  # Split indexes
    
    elif split_by == 'both':  # disjoint split by both, cell and drug
        # TODO: integrate test_size into this type of split!
        
        # Create cross-tab table with cell and drugs
        # (the values indicate the number of data points for each [drug, cell] combination)
        ctb = pd.concat([clb, dlb], axis=1)
        ctb['one'] = 1
        ctb = pd.pivot_table(ctb, index='clb', columns='dlb', values='one', aggfunc='sum', fill_value=0)
        ctb.columns.name = None
        ctb.index.name = None
        lg.logger.info('\nCross-tab table: {}'.format( ctb.shape ))
        lg.logger.info('Total samples in cross-tab table: {}'.format( ctb.values.reshape(-1,).sum() ))
        
        # Shuffle both cells and drugs
        ctb = ctb.sample(frac=1.0, axis=0)
        ctb = ctb.sample(frac=1.0, axis=1)
        
        # Choose range and split data (disjoint sets in the cross-tab table)
        r_bot, r_top = range(0, round(ctb.shape[0]/2)), range(round(ctb.shape[0]/2), ctb.shape[0])
        c_bot, c_top = range(0, round(ctb.shape[1]/2)), range(round(ctb.shape[1]/2), ctb.shape[1])
        t1 = ctb.iloc[r_bot, c_bot]
        t2 = ctb.iloc[r_top, c_top]
        
        # Get cell and drug labels for each data partition
        c1, d1 = t1.index.values, t1.columns.values
        c2, d2 = t2.index.values, t2.columns.values
        
        # Split indexes
        id_grp1 = dlb.isin(d1) & clb.isin(c1)
        id_grp2 = dlb.isin(d2) & clb.isin(c2)
        

# Split
lg.logger.info('\nSplit ...')
df = pd.concat([clb, dlb, df], axis=1)
df1 = df.loc[id_grp1, :]
df2 = df.loc[id_grp2, :]

# Print number of unique cells and drugs
lg.logger.info('Number of unique cells. df1: {}, df2: {}'.format( len(df1['clb'].unique()), len(df2['clb'].unique()) ))
lg.logger.info('Number of unique drugs. df1: {}, df2: {}'.format( len(df1['dlb'].unique()), len(df2['dlb'].unique()) ))

# Test cell and drug intersection between datasets
cell_intrsc = set(df1['clb']).intersection(set(df2['clb']))
drug_intrsc = set(df1['dlb']).intersection(set(df2['dlb']))
lg.logger.info(f'Cell intersection: {len(cell_intrsc)}')
lg.logger.info(f'Drug intersection: {len(drug_intrsc)}')

# Cols to retain
dd_cols = [c for c in df.columns if 'DD_' in c]
ge_cols = [c for c in df.columns if 'GE_' in c]
cols = ['AUC1'] + dd_cols + ge_cols

# Extract only relevant cols (features and target)
df1 = df1[cols].reset_index(drop=True)
df2 = df2[cols].reset_index(drop=True)
lg.logger.info('df1.shape {}'.format( df1.shape ))
lg.logger.info('df2.shape {}'.format( df2.shape ))


# # Define split indices
# if split_method == 'rnd':
#     cv = ShuffleSplit(n_splits=2, test_size=test_size, random_state=0)
#     id_grp1, id_grp2 = next(cv.split(df))
    
# elif split_method == 'hrd':
#     df_dd, df_ge = split_cell_drug(dff=df)
#     # print('df_dd memory_usage {:.3f} GB'.format(df_dd.memory_usage().sum()/1e9))
#     # print('df_ge memory_usage {:.3f} GB'.format(df_ge.memory_usage().sum()/1e9))

#     # Get drug label vector
#     label_name = 'dlb'
#     # df_dd = add_lbl_dup(df_dd, label_name='dlb', prffx='d')
#     dlb = add_lbl_dup(df_dd, label_name=label_name, prffx='d')[label_name]

#     # Get cell label vector
#     label_name = 'clb'
#     # df_ge = add_lbl_dup(df_ge, label_name='clb', prffx='c')
#     clb = add_lbl_dup(df_ge, label_name=label_name, prffx='c')[label_name]

#     del df_dd, df_ge    
    
#     cv = GroupShuffleSplit(n_splits=2, test_size=test_size, random_state=0)
#     if split_by == 'c':
#         # id_grp1, id_grp2 = next(cv.split(df, groups=df_ge[label_name]))  # hard split by cell
#         id_grp1, id_grp2 = next(cv.split(df, groups=clb))  # hard split by cell
#     elif split_by == 'd':
#         # id_grp1, id_grp2 = next(cv.split(df, groups=df_dd[label_name]))  # hard split by drug
#         id_grp1, id_grp2 = next(cv.split(df, groups=dlb))  # hard split by drug


# # Split
# df1 = df.loc[id_grp1, :]
# df2 = df.loc[id_grp2, :]
# del df


# Dump dfs
lg.logger.info('\nDump dfs ...')
df1.to_parquet(outdir/'df_wrm.parquet', engine='auto', compression='snappy')
df2.to_parquet(outdir/'df_ref.parquet', engine='auto', compression='snappy')

lg.logger.info('\nProgram runtime: {:.2f} mins'.format( (time() - t_start)/60 ))
lg.logger.info('Done.')

