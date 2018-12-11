"""
Not finished!!
TODO:
- Need to make a fair comparison btw vsd and rpkm/tpm. There are two options:
    1. Compare vsd vs rpkm values provided by ccle. In this case, need to load
       rpkm values and create another tidy dataset (probably in R).
    2. Compare vsd vs tpm values from combined. In this case, need to make sure
       cell and drug mappings are correct in order to use the correct response values.
- Add descriptors to the tidy data.
- Add prefix to features.
- Add distribution plots of target values to R code.
"""
from __future__ import division
from __future__ import print_function

import warnings
warnings.filterwarnings('ignore')

import os
import sys
import time
import datetime
import logging
import argparse
import psutil
from collections import OrderedDict
import numpy as np
import pandas as pd

import matplotlib
# matplotlib.use('TkAgg')
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats
from sklearn.preprocessing import Imputer, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split, learning_curve, KFold, StratifiedKFold

# Utils
# file_path = os.getcwd()
# file_path = os.path.join(file_path, 'src/models')
# os.chdir(file_path)

file_path = os.path.dirname(os.path.realpath(__file__))  # os.path.dirname(os.path.abspath(__file__))
import utils_models as utils

DATADIR = os.path.join(file_path, '../../data/processed/from_ccle_org')
OUTDIR = os.path.join(file_path, '../../models/from_ccle_org')
# FILENAME = 'tidy_data_ccle_vsd_lincs.txt'
FILENAME = 'tidy_data_ccle_rpkm_lincs.txt'
os.makedirs(OUTDIR, exist_ok=True)

SEED = 0


# ========================================================================
#       Args TODO: add to argparse
# ========================================================================
# Train and infer data
# train_sources = ['ccle']  # ['ccle', 'gcsi', 'gdsc', 'ctrp']
# infer_sources = ['ccle']

# Traget (response)
# target_name: 'EC50um', 'IC50um', 'Amax', 'ActArea'
rsp_cols = ['EC50um', 'IC50um', 'Amax', 'ActArea']
target_name = 'ActArea'

# Features
cell_features = ['rna'] # ['rnaseq', 'cnv']
drug_features = [] # ['descriptors'] # [] # ['descriptors', 'fingerprints', 'descriptors_latent', 'fingerprints_latent']
other_features = [] # ['dlb']
feature_list = cell_features + drug_features + other_features

# Models
ml_models = ['lgb_reg']

trasform_target = False
verbose = True
n_jobs = 4

# Feature prefix (some already present in the tidy dataframe)
fea_prfx_dict = {'rna': 'cell_rna.',
                 'cnv': 'cell_cnv.',
                 'dsc': 'drug_dsc.',
                 'fng': 'drug_fng.',
                 'clb': 'cell_lbl.',
                 'dlb': 'drug_lbl.'}


# ========================================================================
#       Logger
# ========================================================================
t = datetime.datetime.now()
t = [t.year, '-', t.month, '-', t.day, '_', 'h', t.hour, '-', 'm', t.minute]
t = ''.join([str(i) for i in t])
run_outdir = os.path.join(OUTDIR, 'run_'+t)
os.makedirs(run_outdir)
logfilename = os.path.join(run_outdir, 'logfile.log')
logger = utils.setup_logger(logfilename=logfilename)

logger.info(f'File path: {file_path}')
logger.info(f'Num of system CPUs: {psutil.cpu_count()}')
logger.info(f'n_jobs: {n_jobs}')


# ========================================================================
#       Load data and pre-proc
# ========================================================================
datapath = os.path.join(DATADIR, FILENAME)
logger.info(f'\nLoad tidy data ... {datapath}')
data = pd.read_table(datapath, sep='\t')
data.rename(columns={'CellName': 'CELL', 'Drug': 'DRUG'}, inplace=True)
logger.info(f'data.shape {data.shape}')
logger.info('data memory usage (GB): {:.3f}'.format(sys.getsizeof(data)/1e9))
# print(data.groupby('SOURCE').agg({'CELL': 'nunique', 'DRUG': 'nunique', 'PUBCHEM': 'nunique'}).reset_index())
logger.info(data.groupby('tissuetype').agg({'CELL': 'nunique', 'DRUG': 'nunique'}).reset_index())


# Shuffle data
data = data.sample(frac=1.0, axis=0, random_state=SEED).reset_index(drop=True)


if 'dlb' in other_features:
    # print('\nAdd drug labels to features ...')
    logger.info('\nAdd drug labels to features ...')
    # print(data['DRUG'].value_counts())

    # http://queirozf.com/entries/one-hot-encoding-a-feature-on-a-pandas-dataframe-an-example
    # One-hot encoder
    dlb = pd.get_dummies(data=data[['DRUG']], prefix=fea_prfx_dict['dlb'],
                         dummy_na=False).reset_index(drop=True)

    # Label encoder
    # dlb = data[['DRUG']].astype('category', ordered=False).reset_index(drop=True)
    # print(dlb.dtype)

    # Concat drug labels and other features
    data = pd.concat([dlb, data], axis=1).reset_index(drop=True)
    logger.info(f'dlb.shape {dlb.shape}')
    logger.info(f'data.shape {data.shape}')


# ========================================================================
# ========================================================================
def plot_rsp_dists(rsp, rsp_cols, savepath=None):
    """ Plot distributions of response variables.
    Args:
        rsp : df of response values
        rsp_cols : list of col names
        savepath : full path to save the image
    """
    ncols = 2
    nrows = int(np.ceil(len(rsp_cols)/ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=False, sharey=False)
    for i, ax in enumerate(axes.ravel()):
        if i >= len(rsp_cols):
            fig.delaxes(ax) # delete un-used ax
        else:
            target_name = rsp_cols[i]
            # print(i, target_name)
            x = rsp[target_name].copy()
            x = x[~x.isna()].values
            sns.distplot(x, bins=100, kde=True, ax=ax, label=target_name, # fit=norm, 
                        kde_kws={'color': 'k', 'lw': 0.4, 'alpha': 0.8},
                        hist_kws={'color': 'b', 'lw': 0.4, 'alpha': 0.5})
            ax.tick_params(axis='both', which='major', labelsize=7)
            txt = ax.yaxis.get_offset_text(); txt.set_size(7) # adjust exponent fontsize in xticks
            txt = ax.xaxis.get_offset_text(); txt.set_size(7)
            ax.legend(fontsize=5, loc='best')
            ax.grid(True)

    plt.tight_layout()
    # plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
    if savepath is not None:
        plt.savefig(savepath, bbox_inches='tight', dpi=300)
    else:
        plt.savefig('rsp_dists.png', bbox_inches='tight', dpi=300)


plot_rsp_dists(rsp=data, rsp_cols=rsp_cols, savepath=os.path.join(run_outdir, 'rsp_dists.png'))
# ========================================================================
# ========================================================================


# ========================================================================
#       Impute missing values
# ========================================================================
# data = utils.impute_values(data, fea_prefix=fea_prefix, logger=logger)
data = utils.impute_values(data=data, fea_prfx_dict=fea_prfx_dict, logger=logger)


# ========================================================================
#       Split train and val (test)
# ========================================================================
tr_data, vl_data = utils.split_tr_vl(data=data, test_size=0.2, random_state=SEED, logger=logger)


# Extract target and features
##xtr = utils.extract_features(data=tr_data, feature_list=feature_list, fea_prefix=fea_prefix)
##xvl = utils.extract_features(data=vl_data, feature_list=feature_list, fea_prefix=fea_prefix)
xtr, _ = utils.split_features_and_other_cols(tr_data, fea_prfx_dict=fea_prfx_dict)
xvl, _ = utils.split_features_and_other_cols(vl_data, fea_prfx_dict=fea_prfx_dict)

ytr = utils.extract_target(data=tr_data, target_name=target_name)
yvl = utils.extract_target(data=vl_data, target_name=target_name)


# Print features shapes
def print_feature_shapes(df, name):
    logger.info(f'\n{name}')
    for prefx in np.unique(list(map(lambda x: x.split('.')[0], df.columns.tolist()))):
        cols = df.columns[[True if prefx in c else False for c in df.columns.tolist()]]
        logger.info('{}: {}'.format(prefx, df[cols].shape))

print_feature_shapes(df=xtr, name='xtr')
print_feature_shapes(df=xvl, name='xvl')

# Print target
# TODO: update this to something more informative (min, max, quantiles, etc.)
logger.info(f'\nTarget: {target_name}')


# Plot response dist for tr and vl
fig, ax = plt.subplots()
plt.hist(ytr, bins=100, label='ytr', color='b', alpha=0.5)
plt.hist(yvl, bins=100, label='yvl', color='r', alpha=0.5)
plt.title(target_name+' hist')
plt.tight_layout()
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(run_outdir, target_name+'_ytr_yvl_hist.png'), bbox_inches='tight')


# ========================================================================
#       Train models
# ========================================================================
from sklearn.metrics import r2_score, mean_absolute_error, median_absolute_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, ParameterGrid
from sklearn.externals import joblib

train_runtime = OrderedDict() # {}
preds_filename_prefix = 'dev'


# -------------
# LGBMRegressor
# -------------
if 'lgb_reg' in ml_models:
    model_name = 'lgb_reg'
    try:
        import lightgbm as lgb
    except ImportError:  # install??
        logger.error('Module not found (lightgbm)')

    # https://lightgbm.readthedocs.io/en/latest/Python-API.html
    # TODO: use config file to set default parameters
    logger.info('\nTrain LGBMRegressor ...')
    ml_objective = 'regression'
    eval_metric = ['l1', # aliases: regression_l1, mean_absolute_error, mae
                   'l2', # aliases: regression, regression_l2, mean_squared_error, mse, and more
                   ]

    # ----- lightgbm "Training API" - start
    # lgb_tr = lgb.Dataset(data=xtr, label=ytr, categorical_feature='auto')
    # lgb_vl = lgb.Dataset(data=xvl, label=yvl, categorical_feature='auto')
    # # https://lightgbm.readthedocs.io/en/latest/Parameters.html
    # params = {'task': 'train', # default='train'
    #         'objective': ml_objective, # default='regression' which alias for 'rmse' and 'mse' (but these are different??)
    #         'boosting': 'gbdt', # default='gbdt'
    #         'num_iterations': 100, # default=100 (num of boosting iterations)
    #         'learning_rate': 0.1, # default=0.1
    #         'num_leaves': 31, # default=31 (num of leaves in 1 tree)
    #         'seed': SEED,
    #         'num_threads': n_jobs, # default=0 (set to the num of real CPU cores)
    #         'device_type': 'cpu', # default='cpu'
    #         'metric': eval_metric # metric(s) to be evaluated on the evaluation set(s)
    #         }
    # t0 = time.time()
    # lgb_reg = lgb.train(params=params, train_set=lgb_tr, valid_sets=lgb_vl, verbose_eval=False)
    # # lgb_cv = lgb.train(params=params, train_set=lgb_tr, nfolds=5)
    # train_runtime['lgb_reg'] = time.time() - t0
    # logger.info('Runtime: {:.2f} mins'.format(train_runtime['lgb_reg']/60))
    # ----- lightgbm "Training API" - end 

    # ----- lightgbm "sklearn API" appraoch 1 - start
    lgb_reg = lgb.LGBMModel(objective=ml_objective,
                            n_jobs=n_jobs,
                            random_state=SEED)
    # lgb_reg = lgb.LGBMRegressor()
    t0 = time.time()
    lgb_reg.fit(xtr, ytr, eval_metric=eval_metric, eval_set=[(xtr, ytr), (xvl, yvl)],
                early_stopping_rounds=10, verbose=False, callbacks=None)
    train_runtime[model_name] = time.time() - t0
    logger.info('Runtime: {:.2f} mins'.format(train_runtime[model_name]/60))
    # ----- lightgbm "sklearn API" appraoch 1 - end

    # Save model
    # lgb_reg.save_model(os.path.join(run_outdir, 'lgb_'+ml_type+'_model.txt'))
    joblib.dump(lgb_reg, filename=os.path.join(run_outdir, model_name+'_model.pkl'))
    # lgb_reg_ = joblib.load(filename=os.path.join(run_outdir, 'lgb_reg_model.pkl'))

    # Print preds
    # utils.print_scores(model=lgb_reg, xdata=xtr, ydata=ytr)
    # utils.print_scores(model=lgb_reg, xdata=xvl, ydata=yvl)
    utils.print_scores(model=lgb_reg, xdata=xvl, ydata=yvl, logger=logger)

    # Dump preds
    utils.dump_preds(model=lgb_reg, df_data=vl_data, xdata=xvl, target_name=target_name,
                     path=os.path.join(run_outdir, preds_filename_prefix+'_'+model_name+'_preds.csv'))

    # Plot feature importance
    lgb.plot_importance(booster=lgb_reg, max_num_features=20, grid=True, title='LGBMRegressor')
    plt.tight_layout()
    plt.savefig(os.path.join(run_outdir, model_name+'_importances.png'))

    # Plot learning curves
    # TODO: note, plot_metric didn't accept 'mae' although it's alias for 'l1' 
    # TODO: plot_metric requires dict from train(), but train returns 'lightgbm.basic.Booster'??
    for m in eval_metric:
        ax = lgb.plot_metric(booster=lgb_reg, metric=m, grid=True)
        plt.savefig(os.path.join(run_outdir, model_name+'_learning_curve_'+m+'.png'))