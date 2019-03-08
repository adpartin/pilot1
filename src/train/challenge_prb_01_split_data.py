from __future__ import print_function, division

import warnings
warnings.filterwarnings('ignore')

import os
import sys
import pathlib
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
from sklearn.model_selection import ShuffleSplit, KFold # (AP)
from sklearn.model_selection import GroupShuffleSplit, GroupKFold # (AP)
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold # (AP)

SEED = None
t_start = time()


# Utils
import classlogger
import utils


# File path
# file_path = os.path.dirname(os.path.realpath(__file__))
file_path = pathlib.Path(__file__).resolve().parent


# Path - create dir to dump results (AP)
PRJ_NAME = 'candle_challenge_prb'
OUTDIR = pathlib.Path(file_path / '../../models' / PRJ_NAME / 'data')
os.makedirs(OUTDIR, exist_ok=True)


# Arg parser
psr = argparse.ArgumentParser(description='input agg csv file')
psr.add_argument('--in', default=None)
psr.add_argument('--split_method', type=str, choices=['rnd', 'hrd'], default='hrd')
psr.add_argument('--split_by', type=str, choices=['c', 'd'], default='c')
psr.add_argument('--ratio', type=int, default=0.5)

args = vars(psr.parse_args())
pprint(args)

# Args
data_path = args['in']
split_method = args['split_method']
split_by = args['split_by']  # applied with hard partition
wrm_ratio = args['ratio']
test_size = 1 - wrm_ratio

# Outdir
if split_method == 'rnd':
    outdir = OUTDIR / 'rnd'
elif split_method == 'hrd':
    outdir = OUTDIR / 'hrd'
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
    df = pd.read_csv(data_path, skiprows=1, dtype='float32', nrows=args['nrows']).values # (AP)
elif 'parquet' in data_path:
    df = pd.read_parquet(data_path, engine='auto', columns=None) # (AP)
    df = df.sample(frac=1.0, axis=0, random_state=SEED) # shuffle values
lg.logger.info('Done ({:.2f} mins).\n'.format((time()-t0)/60))


# Drop constant features (AP)
lg.logger.info('Drop constant features ...')
lg.logger.info(df.shape)
col_idx = df.nunique(dropna=True).values==1  # col indexes to drop
df = df.iloc[:, ~col_idx]
lg.logger.info(df.shape)


# ----------
# Split data
# ----------
def split_cell_drug(dff):
    """ Split drug and cell. """
    dff = dff.copy()
    dd_cols = [c for c in df.columns if 'DD_' in c]
    ge_cols = [c for c in df.columns if 'GE_' in c]
    df_dd = dff[dd_cols]
    df_ge = dff[ge_cols]
    print('\ndf_dd.shape', df_dd.shape)
    print('df_ge.shape', df_ge.shape)
    return df_dd, df_ge


def add_lbl_dup(dff, label_name='lb', prffx='_'):
    """ Add col indicating with unique row (label). """
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
    dff = dff.reindex(idx_org)
    dff = dff.drop(columns=['d'])
    
    dff[label_name] = dff[label_name].map(lambda x: prffx + str(int(x)))
    return dff


# Define split indices
if split_method == 'rnd':
    cv = ShuffleSplit(n_splits=2, test_size=test_size, random_state=0)
    id_grp1, id_grp2 = next(cv.split(df))
    
elif split_method == 'hrd':
    df_dd, df_ge = split_cell_drug(dff=df)
    # print('df_dd memory_usage {:.3f} GB'.format(df_dd.memory_usage().sum()/1e9))
    # print('df_ge memory_usage {:.3f} GB'.format(df_ge.memory_usage().sum()/1e9))

    # Get drug label vector
    label_name = 'dlb'
    # df_dd = add_lbl_dup(df_dd, label_name='dlb', prffx='d')
    dlb = add_lbl_dup(df_dd, label_name='dlb', prffx='d')[label_name]

    # Get cell label vector
    label_name = 'clb'
    # df_ge = add_lbl_dup(df_ge, label_name='clb', prffx='c')
    clb = add_lbl_dup(df_ge, label_name='clb', prffx='c')[label_name]

    del df_dd, df_ge    
    
    cv = GroupShuffleSplit(n_splits=2, test_size=test_size, random_state=0)
    if split_by == 'c':
        # id_grp1, id_grp2 = next(cv.split(df, groups=df_ge[label_name]))  # hard split by cell
        id_grp1, id_grp2 = next(cv.split(df, groups=clb))  # hard split by cell
    elif split_by == 'd':
        # id_grp1, id_grp2 = next(cv.split(df, groups=df_dd[label_name]))  # hard split by drug
        id_grp1, id_grp2 = next(cv.split(df, groups=dlb))  # hard split by drug


# Split
df1 = df.loc[id_grp1, :]
df2 = df.loc[id_grp2, :]
del df


# Dump dfs
lg.logger.info('\nDump dfs ...')
df1.to_parquet(outdir/'df_wrm.parquet', engine='auto', compression='snappy')
df2.to_parquet(outdir/'df_cnt.parquet', engine='auto', compression='snappy')

lg.logger.info('\nProgram runtime: {:.2f} mins'.format( (time() - t_start)/60 ))
lg.logger.info('Done.')


















# # --------------------------------
# # Train 1st model and dump weights
# # --------------------------------
# # cv = ShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
# # tr_idx, vl_idx = next(cv.split(dfx1, dfy1))
# xtr1, xte1, ytr1, yte1 = train_test_split(dfx1, dfy1)
# print('xtr1.shape:', xtr1.shape)
# print('xte1.shape:', xte1.shape)
# print('ytr1.shape:', ytr1.shape)
# print('yte1.shape:', yte1.shape)

# # Define callbacks and fit_params for phase 1 (warm-up)
# checkpointer = ModelCheckpoint(os.path.join(outdir, 'model1.wrm.{epoch:02d}-{val_loss:.2f}.h5'), verbose=0, save_weights_only=False, save_best_only=True)
# csv_logger = CSVLogger(filename=os.path.join(outdir, 'model1.wrm.log'))
# callback_list = [checkpointer, csv_logger, early_stop, reduce_lr]

# fit_params = {'batch_size': BATCH, 'epochs': EPOCH_WRM, 'verbose': 1}
# fit_params['validation_data'] = (xte1, yte1)
# fit_params['callbacks'] = callback_list

# # Get the estimator
# model1 = ml_models.get_model(model_name=model_name, init_params=init_params)

# # Train model phase 1
# history_wrm = model1.model.fit(xtr1, ytr1, **fit_params)
# score = model1.model.evaluate(xte1, yte1, verbose=0)
# print('val_loss: {:.3f}'.format(score[0]))

# # Print plots
# model1_wrm_plts_path = os.path.join(outdir, 'model1_wrm_plts')
# os.makedirs(model1_wrm_plts_path, exist_ok=True)
# ml_models.plot_prfrm_metrics(history=history_wrm, title=f'Model 1: warm-up training',
#                              outdir=model1_wrm_plts_path)


# # Define path
# model1_path = os.path.join(outdir, 'model1.wrm.json')
# weights1_path = os.path.join(outdir, 'weights1.wrm.h5')

# # wrm model
# model_json = model1.model.to_json()
# with open(model1_path, 'w') as json_file:
#     json_file.write(model_json)

# # wrm weights
# model1.model.save_weights(weights1_path)
    
# # Define callbacks and fit_params for phase 2 (continue training)
# checkpointer = ModelCheckpoint(filepath=os.path.join(outdir, 'model1.cnt.h5'), verbose=0, save_weights_only=False, save_best_only=True)
# csv_logger = CSVLogger(filename=os.path.join(outdir, 'model1.cnt.log'))
# callback_list = [checkpointer, csv_logger, early_stop, reduce_lr]

# fit_params = {'batch_size': BATCH, 'epochs': EPOCH - EPOCH_WRM, 'verbose': 1}
# fit_params['validation_data'] = (xte1, yte1)
# fit_params['callbacks'] = callback_list

# # Train model phase 2
# history_cnt = model1.model.fit(xtr1, ytr1, **fit_params)
# score = model1.model.evaluate(xte1, yte1, verbose=0)
# print('val_loss: {:.3f}'.format(score[0]))

# # Print plots
# model1_cnt_plts_path = os.path.join(outdir, 'model1_cnt_plts')
# os.makedirs(model1_cnt_plts_path, exist_ok=True)
# ml_models.plot_prfrm_metrics(history=history_cnt, title=f'Model 1: continue training',
#                              outdir=model1_cnt_plts_path)



# # --------------------------------
# # Train 2nd model and dump weights
# # --------------------------------
# xtr2, xte2, ytr2, yte2 = train_test_split(dfx2, dfy2)
# print('xtr2.shape:', xtr2.shape)
# print('xte2.shape:', xte2.shape)
# print('ytr2.shape:', ytr2.shape)
# print('yte2.shape:', yte2.shape)

# # Define callbacks and fit_params for phase 1 (warm-up)
# checkpointer = ModelCheckpoint(filepath=os.path.join(outdir, 'model2.scratch.h5'), verbose=0, save_weights_only=False, save_best_only=True)
# csv_logger = CSVLogger(filename=os.path.join(outdir, 'model2.scratch.log'))
# callback_list = [checkpointer, csv_logger, early_stop, reduce_lr]

# fit_params = {'batch_size': BATCH, 'epochs': EPOCH, 'verbose': 1}
# fit_params['validation_data'] = (xte2, yte2)
# fit_params['callbacks'] = callback_list

# # Get the estimator
# model2 = ml_models.get_model(model_name=model_name, init_params=init_params)

# # Train model2 - from scratch
# history_from_scratch = model2.model.fit(xtr2, ytr2, **fit_params)
# score = model2.model.evaluate(xte2, yte2, verbose=0)
# print('val_loss: {:.3f}'.format(score[0]))

# # Print plots
# model2_from_scratch_plts_path = os.path.join(outdir, 'model2_from_scratch_plts')
# os.makedirs(model2_from_scratch_plts_path, exist_ok=True)
# ml_models.plot_prfrm_metrics(history=history_from_scratch, title=f'Model 2: training from scratch',
#                              outdir=model2_from_scratch_plts_path)


# # ------------------
# # Load model1 warmed 
# # ------------------
# # Define callbacks and fit_params for phase 1 (warm-up)
# checkpointer = ModelCheckpoint(filepath=os.path.join(outdir, 'model2.from_wrm.h5'), verbose=0, save_weights_only=False, save_best_only=True)
# csv_logger = CSVLogger(filename=os.path.join(outdir, 'model2.from_wrm.log'))
# callback_list = [checkpointer, csv_logger, early_stop, reduce_lr]

# fit_params = {'batch_size': BATCH, 'epochs': EPOCH - EPOCH_WRM, 'verbose': 1}
# fit_params['validation_data'] = (xte2, yte2)
# fit_params['callbacks'] = callback_list

# # Load wamred model1
# json_file = open(model1_path, 'r')
# model1_wrm_json = json_file.read()
# json_file.close()
# model1_wrm_loaded = model_from_json(model1_wrm_json)

# # Load weights into warmed model                                                                                                                 
# model1_wrm_loaded.load_weights(weights1_path)

# # Compile loaded model
# model1_wrm_loaded.compile(loss='mean_squared_error',
#                           optimizer=SGD(lr=0.0001, momentum=0.9),
#                           metrics=['mae', r2])

# # Train model2 - from wrm
# history_from_wrm = model1_wrm_loaded.fit(xtr2, ytr2, **fit_params)
# score = model1_wrm_loaded.evaluate(xte2, yte2, verbose=0)
# print('val_loss: {:.3f}'.format(score[0]))

# # Print plots
# model2_from_wrm_plts_path = os.path.join(outdir, 'model2_from_wrm_plts')
# os.makedirs(model2_from_wrm_plts_path, exist_ok=True)
# ml_models.plot_prfrm_metrics(history=history_from_wrm, title=f'Model 2: training from warmed model1',
#                              outdir=model2_from_wrm_plts_path)

# print('Done')



