"""
TODO:
1. Multiple ML models
- train multiple models
- rank models based on performance
- optimize each model using hyperparam search
- ensemble/stack models
  http://blog.kaggle.com/2017/06/15/stacking-made-easy-an-introduction-to-stacknet-by-competitions-grandmaster-marios-michailidis-kazanova/

Auto-ML models:
- tpot
- auto-sklearn

ML models:
- NN (consider various normalization methods) - https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html
- xgboost (gbtree or gblinear)
- lightgbm
- catboost
- random forest
- naive bayes
- svm
- knn
- elastic net

Hyperparam schemes:
- CANDLE
- https://medium.com/@mikkokotila/a-comprehensive-list-of-hyperparameter-optimization-tuning-solutions-88e067f19d9

2. Learning curve (performance vs training set size)
https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html

3. Outliers and transformations
https://www.analyticsvidhya.com/blog/2015/11/8-ways-deal-continuous-variables-predictive-modeling/
- unskew the data; drop outliers based on boxplot (stratified by drug and tissue type)

4. Another feature to add would be cluster-based:
http://blog.kaggle.com/2015/07/27/taxi-trajectory-winners-interview-1st-place-team-%F0%9F%9A%95/
- Apply clustering to rna-seq. The clusters vector will become a categorical variable. In this case
  we avoid using tissue type labels but rather use proximity in the actual feature space.
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
matplotlib.use('TkAgg')
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

import seaborn as sns
from pprint import pprint

from scipy import stats
from sklearn.preprocessing import Imputer, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split, learning_curve, KFold, StratifiedKFold

# Utils
file_path = os.getcwd()
file_path = os.path.join(file_path, 'src/models')
os.chdir(file_path)

# DATADIR = './tidy_data_from_combined'
# FILENAME = 'tidy_data.parquet'
# OUTDIR = os.path.join(file_path, 'ml_tidy_combined')
# os.makedirs(OUTDIR, exist_ok=True)

# file_path = os.path.dirname(os.path.realpath(__file__))  # os.path.dirname(os.path.abspath(__file__))
import utils_models as utils

DATADIR = os.path.join(file_path, '../../data/processed/from_combined')
OUTDIR = os.path.join(file_path, '../../models/from_combined')
os.makedirs(OUTDIR, exist_ok=True)

SEED = 0


# ========================================================================
#       Args TODO: add to argparse
# ========================================================================
FILENAME = 'tidy_data_no_fibro.parquet'

train_sources = ['ccle']  # ['ccle', 'gcsi', 'gdsc', 'ctrp']
infer_sources = ['ccle']

target_name = 'AUC'  # response

cell_features = ['rnaseq'] # ['rnaseq', cnv', 'rnaseq_latent']
drug_features = ['descriptors'] # [] # ['descriptors', 'fingerprints', 'descriptors_latent', 'fingerprints_latent']
other_features = [] # ['drug_labels'] # ['cell_labels', 'drug_labels', 'ctype', 'csite', 'rna_clusters']

verbose = True

fea_prefix = {'rnaseq': 'cell_rna',
              'cnv': 'cell_cnv',
              'descriptors': 'drug_dsc',
              'fingerprints': 'drug_fng'}


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


# ========================================================================
#       Load data and pre-proc
# ========================================================================
datapath = os.path.join(DATADIR, FILENAME)
logger.info(f'\nLoad tidy data ... {datapath}')
data = pd.read_parquet(datapath, engine='auto', columns=None)
logger.info(f'data.shape {data.shape}')
logger.info('data memory usage (GB): {:.3f}'.format(sys.getsizeof(data)/1e9))
# print(data.groupby('SOURCE').agg({'CELL': 'nunique', 'DRUG': 'nunique', 'PUBCHEM': 'nunique'}).reset_index())


# Extract sources
logger.info('\nExtract train sources ... {}'.format(train_sources))
data = data[data['SOURCE'].isin(train_sources)].reset_index(drop=True)
logger.info(f'data.shape {data.shape}')
logger.info('data memory usage (GB): {:.3f}'.format(sys.getsizeof(data)/1e9))
# print(data.groupby('SOURCE').agg({'CELL': 'nunique', 'DRUG': 'nunique'}).reset_index())


# Replace characters that are illegal for xgboost feature names
# xdata.columns = list(map(lambda s: s.replace('[','_').replace(']','_'), xdata.columns.tolist())) # required by xgboost
import re
regex = re.compile(r'\[|\]|<', re.IGNORECASE)
data.columns = [regex.sub('_', c) if any(x in str(c) for x in set(('[', ']', '<'))) else c for c in data.columns.values]


# Assign type to categoricals
# cat_cols = data.select_dtypes(include='object').columns.tolist()
# data[cat_cols] = data[cat_cols].astype('category', ordered=False)


# Shuffle data
data = data.sample(frac=1.0, axis=0, random_state=SEED).reset_index(drop=True)


# Filter out AUC>1
# print('\nFilter some AUC outliers (>1)')
# print('data.shape', data.shape)
# data = data[[False if x>1.0 else True for x in data[target_name]]].reset_index(drop=True)
# print('data.shape', data.shape)


# Plots
utils.boxplot_rsp_per_drug(df=data, target_name=target_name,
                           path=os.path.join(run_outdir, f'{target_name}_per_drug_boxplot.png'))
utils.plot_hist(x=data[target_name], var_name=target_name,
                path=os.path.join(run_outdir, target_name+'_hist.png'))
utils.plot_qq(x=data[target_name], var_name=target_name, path=os.path.join(run_outdir, target_name+'_qqplot.png'))
utils.plot_hist_drugs(x=data['DRUG'], path=os.path.join(run_outdir, 'drugs_hist.png'))


# Transform the target
# y = np.log1p(ydata); plot_hist(x=y, var_name=target_name+'_log1p')
# # y = np.log(ydata+1); plot_hist(x=y, var_name=target_name+'_log+1')
# y = np.log10(ydata+1); plot_hist(x=y, var_name=target_name+'_log10')
# y = np.log2(ydata+1); plot_hist(x=y, var_name=target_name+'_log2')
# y = ydata**2; plot_hist(x=ydata, var_name=target_name+'_x^2')
# y, lmbda = stats.boxcox(ydata+1); # utils.plot_hist(x=y, var_name=target_name+'_boxcox', path=)
# ydata = pd.DataFrame(y)


if 'drug_labels' in other_features:
    # print('\nAdd drug labels to features ...')
    logger.info('\nAdd drug labels to features ...')
    # print(data['DRUG'].value_counts())

    # http://queirozf.com/entries/one-hot-encoding-a-feature-on-a-pandas-dataframe-an-example
    # One-hot encoder
    drug_labels = pd.get_dummies(data=data[['DRUG']], prefix='drug_label',
                                 prefix_sep='.', dummy_na=False).reset_index(drop=True)

    # Label encoder
    # drug_labels = data[['DRUG']].astype('category', ordered=False).reset_index(drop=True)
    # print(drug_labels.dtype)

    # Concat drug labels and other features
    data = pd.concat([drug_labels, data], axis=1).reset_index(drop=True)
    logger.info(f'drug_labels.shape {drug_labels.shape}')
    logger.info(f'data.shape {data.shape}')


if 'rna_clusters' in other_features:
    # TODO
    pass



# ========================================================================
#       Impute missing values
# ========================================================================
from sklearn.impute import SimpleImputer, MissingIndicator

logger.info('\nImpute missing features ...')
tmp_data = data.copy()
df_list = []
for prefx in fea_prefix.values():
    cols = data.columns[[True if prefx in c else False for c in data.columns.tolist()]]
    if len(cols) > 0:        
        df = data[cols].copy()
        tmp_data.drop(columns=cols, inplace=True)
        df_list.append(df)

xdata_to_impute = pd.DataFrame(pd.concat(df_list, axis=1))

# TODO: try regressor (impute continuous features) or classifier (impute discrete features)
# https://scikit-learn.org/stable/auto_examples/plot_missing_values.html
logger.info('Num features with missing values: {}'.format(sum(xdata_to_impute.isna().sum() > 1)))
cols = xdata_to_impute.columns
imputer = SimpleImputer(missing_values=np.nan, strategy='mean', verbose=1)
xdata_imputed = imputer.fit_transform(xdata_to_impute)
xdata_imputed = pd.DataFrame(xdata_imputed, columns=cols)
logger.info('Num features with missing values (after impute): {}'.format(sum(xdata_imputed.isna().sum() > 1)))

data = pd.concat([tmp_data, xdata_imputed], axis=1)
del xdata_to_impute, xdata_imputed



# ========================================================================
#       Split train and val (test)
# ========================================================================
logger.info('\nSplit to train and val (test) ...')
tr_data, vl_data = train_test_split(data, test_size=0.2, random_state=SEED)
tr_data.reset_index(drop=True, inplace=True)
vl_data.reset_index(drop=True, inplace=True)
logger.info(f'tr_data.shape {tr_data.shape}')
logger.info(f'vl_data.shape {vl_data.shape}')

# Extract target and features
fea_prefix_list = [fea_prefix[fea] for fea in (cell_features + drug_features) if fea in fea_prefix.keys()]
xtr = tr_data[[c for c in tr_data.columns if c.split('.')[0] in fea_prefix_list]].reset_index(drop=True).copy()
xvl = vl_data[[c for c in vl_data.columns if c.split('.')[0] in fea_prefix_list]].reset_index(drop=True).copy()
ytr = tr_data[target_name].copy()
yvl = vl_data[target_name].copy()

# Print features shapes
def print_feature_shapes(df, name):
    logger.info(f'\n{name}')
    for prefx in np.unique(list(map(lambda x: x.split('.')[0], df.columns.tolist()))):
        cols = df.columns[[True if prefx in c else False for c in df.columns.tolist()]]
        logger.info('{}: {}'.format(prefx, df[cols].shape))

print_feature_shapes(df=xtr, name='xtr')
print_feature_shapes(df=xvl, name='xvl')

# Plots
# TODO: overlap tr and vl plots for both response and drug counts
# utils.plot_hist(x=ytr, var_name=target_name+'_ytr', path=os.path.join(OUTDIR, target_name+'_ytr_hist.png'))
# utils.plot_hist(x=yvl, var_name=target_name+'_yvl', path=os.path.join(OUTDIR, target_name+'_yvl_hist.png'))
# utils.plot_hist_drugs(x=tr_data['DRUG'], name='drug_hist_train', path=os.path.join(OUTDIR, target_name+'_drug_hist_train.png'))
# utils.plot_hist_drugs(x=vl_data['DRUG'], name='drug_hist_val', path=os.path.join(OUTDIR, target_name+'_drug_hist_train.png'))

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
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor
import xgboost as xgb
import lightgbm as lgb
import tpot

from sklearn.metrics import r2_score, mean_absolute_error, median_absolute_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, ParameterGrid
from sklearn.externals import joblib


ml_runtime = OrderedDict() # {}


# ---------------------
# RandomForestRegressor
# ---------------------
logger.info('\nTrain RandomForestRegressor ...')
rf_reg = RandomForestRegressor(max_features='sqrt', bootstrap=True, oob_score=True,
                               verbose=1, random_state=SEED, n_jobs=-1)
# grid_search_params = {'n_estimators': [100, 250], # [100, 250, 500]
#                      'max_depth': [None, 9],  # [None, 5, 9, 15]
#                      # 'min_samples_split': [2, 6, 10],
#                      # 'min_samples_leaf': [1, ]
# }
# rf_reg_gridsearch = GridSearchCV(
#     estimator=rf_reg,
#     param_grid=grid_search_params,
#     scoring=None,
#     n_jobs=-1,
#     cv=5,
#     refit=True,
#     verbose=0,
# )
random_search_params = {'n_estimators': [100, 200],
                        'max_depth': [None, 10]}
# random_search_params = {'n_estimators': [int(x) for x in np.linspace(start=200, stop=2000, num=10)],
#                         'max_depth': [None] + [int(x) for x in np.linspace(10, 110, num = 11)],
#                         'min_samples_split': [2, 5, 9],
#                         'min_samples_leaf': [1, 5, 9]}                        
rf_reg_gridsearch = RandomizedSearchCV(
    estimator=rf_reg,
    param_distributions=random_search_params,
    n_iter=3,  # num of parameter settings that are sampled and used for training (num of models trained)
    scoring=None, # string or callable used to evaluate the predictions on the test set
    n_jobs=8,
    cv=5,
    refit=True,  # Refit an estimator using the best found parameters on the whole dataset
    verbose=0)
t0 = time.time()
rf_reg_gridsearch.fit(xtr, ytr)
ml_runtime['rf_reg_gridsearch'] = time.time() - t0
logger.info('Runtime: {:.2f} mins'.format(ml_runtime['rf_reg_gridsearch']/60))

# Save best model
rf_reg = rf_reg_gridsearch.best_estimator_
joblib.dump(rf_reg, filename=os.path.join(run_outdir, 'rf_reg_hypsearch_best_model.pkl'))

# Save resutls
rf_reg_hypsearch = pd.DataFrame(rf_reg_gridsearch.cv_results_)
rf_reg_hypsearch.to_csv(os.path.join(run_outdir, 'rf_reg_hypsearch_summary.csv'))  # save hyperparam search results

logger.info(f'rf_reg best score (random search): {rf_reg_gridsearch.best_score_:.3f}')
logger.info('rf_reg best params (random search): \n{}'.format(rf_reg_gridsearch.best_params_))

# Dump preds
utils.dump_preds(model=rf_reg, df_data=vl_data, xdata=xvl, target_name=target_name,
                 path=os.path.join(run_outdir, 'rf_pred_vl_preds.csv'))


# ------------
# XGBRegressor
# ------------
# https://xgboost.readthedocs.io/en/latest/python/python_api.html
# xgboost does not support categorical features!
# Rules of thumb
# 1. learning_rate should be 0.1 or lower (smaller values will require more trees).
# 2. tree_depth should be between 2 and 8 (where not much benefit is seen with deeper trees).
# 3. subsample should be between 30% and 80% of the training dataset, and compared to a value of 100% for no sampling.
logger.info('\nTrain XGBRegressor ...')
# dtr = xgb.DMatrix(data=xtr, label=ytr, nthread=-1)
# dvl = xgb.DMatrix(data=xvl, label=yvl, nthread=-1)
xgb_reg = xgb.XGBRegressor(objective='reg:linear', # default: 'reg:linear', TODO: docs recommend funcs for different distributions (??)
                           booster='gbtree', # default: gbtree (others: gblinear, dart)
                           max_depth=3, # default: 3
                           learning_rate=0.1, # default: 0.1
                           n_estimators=100, # default: 100
                           n_jobs=-1, # default: 1
                           reg_alpha=0, # default=0, L1 regularization
                           reg_lambda=1, # default=1, L2 regularization
                           random_state=SEED
)
eval_metric = ['mae', 'rmse']
t0 = time.time()
xgb_reg.fit(xtr, ytr, eval_metric=eval_metric, eval_set=[(xtr, ytr), (xvl, yvl)],
            early_stopping_rounds=10, verbose=False, callbacks=None)
ml_runtime['xgb_reg'] = time.time() - t0
logger.info('Runtime: {:.2f} mins'.format(ml_runtime['xgb_reg']/60))

# Save model
# xgb_reg.save_model(os.path.join(run_outdir, 'xgb_reg.model'))
joblib.dump(xgb_reg, filename=os.path.join(run_outdir, 'xgb_reg_model.pkl'))
# xgb_reg_ = joblib.load(filename=os.path.join(run_outdir, 'xgb_reg_model.pkl'))

# Print preds
# utils.print_scores(model=xgb_reg, xdata=xtr, ydata=ytr)
# utils.print_scores(model=xgb_reg, xdata=xvl, ydata=yvl)
utils.print_scores(model=xgb_reg, xdata=xvl, ydata=yvl, logger=logger)

# Dump preds
utils.dump_preds(model=xgb_reg, df_data=vl_data, xdata=xvl, target_name=target_name,
                 path=os.path.join(run_outdir, 'xgb_pred_vl_preds.csv'))
    
# Plot feature importance
xgb.plot_importance(booster=xgb_reg, max_num_features=20, grid=True, title='XGBRegressor')
plt.tight_layout()
plt.savefig(os.path.join(run_outdir, 'xgb_reg_importances.png'))

# Plot learning curves
xgb_results = xgb_reg.evals_result()
epoch_vec = np.arange(1, len(xgb_results['validation_0'][eval_metric[0]])+1)
for m in eval_metric:
    fig, ax = plt.subplots()
    for i, s in enumerate(xgb_results):
        label = 'Train' if i==0 else 'Val'
        ax.plot(epoch_vec, xgb_results[s][m], label=label)
    plt.xlabel('Epochs')
    plt.title(m)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(run_outdir, 'xgb_reg_leraning_curve_'+m+'.png'))


# -------------
# LGBMRegressor
# -------------
# https://lightgbm.readthedocs.io/en/latest/Python-API.html
# TODO: use config file to set default parameters
logger.info('\nTrain LGBMRegressor ...')
# ----- lightgbm "Training API" start
ml_type = 'reg'
lgb_tr = lgb.Dataset(data=xtr, label=ytr, categorical_feature='auto')
lgb_vl = lgb.Dataset(data=xvl, label=yvl, categorical_feature='auto')
# https://lightgbm.readthedocs.io/en/latest/Parameters.html
eval_metric = ['l1', # aliases: regression_l1, mean_absolute_error, mae
               'l2', # aliases: regression, regression_l2, mean_squared_error, mse, and more
]
params = {'task': 'train', # default='train'
          'objective': 'regression', # default='regression' which alias for 'rmse' and 'mse' (but these are different??)
          'boosting': 'gbdt', # default='gbdt'
          'num_iterations': 100, # default=100 (num of boosting iterations)
          'learning_rate': 0.1, # default=0.1
          'num_leaves': 31, # default=31 (num of leaves in 1 tree)
          'seed': SEED,
          'num_threads': 0, # default=0 (set to the num of real CPU cores)
          'device_type': 'cpu', # default='cpu'
          'metric': eval_metric # metric(s) to be evaluated on the evaluation set(s)
}
t0 = time.time()
lgb_reg = lgb.train(params=params, train_set=lgb_tr, valid_sets=lgb_vl)
# lgb_cv = lgb.train(params=params, train_set=lgb_tr, nfolds=5)
ml_runtime['lgb_reg'] = time.time() - t0
logger.info('Runtime: {:.2f} mins'.format(ml_runtime['lgb_reg']/60))
# ----- lightgbm "Training API" end 
# ----- lightgbm "sklearn API" start
# t0 = time.time()
# lgb_reg = lgb.LGBMRegressor()
# lgb_reg.fit(xtr, ytr, eval_metric=eval_metric, eval_set=[(xtr, ytr), (xvl, yvl)],
#             early_stopping_rounds=10, verbose=False, callbacks=None)
# ml_runtime['lgb_reg'] = time.time() - t0
# logger.info('Runtime: {:.2f} mins'.format(ml_runtime['lgb_reg']/60))

# Save model
# lgb_reg.save_model(os.path.join(run_outdir, 'lgb_'+ml_type+'_model.txt'))
joblib.dump(lgb_reg, filename=os.path.join(run_outdir, 'lgb_reg_model.pkl'))
# lgb_reg_ = joblib.load(filename=os.path.join(run_outdir, 'lgb_reg_model.pkl'))

# Print preds
# utils.print_scores(model=lgb_reg, xdata=xtr, ydata=ytr)
# utils.print_scores(model=lgb_reg, xdata=xvl, ydata=yvl)
utils.print_scores(model=lgb_reg, xdata=xvl, ydata=yvl, logger=logger)

# Dump preds
utils.dump_preds(model=lgb_reg, df_data=vl_data, xdata=xvl, target_name=target_name,
                 path=os.path.join(run_outdir, 'lgb_pred_vl_preds.csv'))

# Plot feature importance
lgb.plot_importance(booster=lgb_reg, max_num_features=20, grid=True, title='LGBMRegressor')
plt.tight_layout()
plt.savefig(os.path.join(run_outdir, 'lgb_reg_importances.png'))

# Plot learning curves
# TODO: note, plot_metric didn't accept 'mae' although it's alias for 'l1' 
# TODO: plot_metric requires dict from train(), but train returns 'lightgbm.basic.Booster'??
# for m in ['l1', 'l2']:
#     ax = lgb.plot_metric(booster=lgb_reg, metric=m, grid=True)
#     plt.savefig(os.path.join(run_outdir, 'lgb_reg_leraning_curve_'+m+'.png'))


# -------------
# TPOTRegressor
# -------------
tpot_reg_checkpnt_dir = os.path.join(run_outdir, 'tpot_reg_checkpoints')
os.makedirs(os.path.join(run_outdir, tpot_reg_checkpnt_dir))
logger.info('\nTrain TPOTRegressor ...')
tpot_reg = tpot.TPOTRegressor(generations=5, population_size=50, verbosity=2)
t0 = time.time()
tpot_reg.fit(xtr, ytr)
ml_runtime['tpot_reg'] = time.time() - t0
logger.info('Runtime: {:.2f} mins'.format(ml_runtime['tpot_reg']/60))
tpot_reg.export(os.path.join(run_outdir, 'tpot_reg_pipeline.py'))
logger.info(tpot_reg.score(xvl, yvl))

# Dump preds
utils.dump_preds(model=tpot_reg, df_data=vl_data, xdata=xvl, target_name=target_name,
                 path=os.path.join(run_outdir, 'tpot_reg_pred_vl_preds.csv'))


# ========================================================================
# From competition
# ========================================================================
# # https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard
# from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge, LassoLarsIC
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# from sklearn.kernel_ridge import KernelRidge
# from sklearn.pipeline import Pipeline, make_pipeline
# from sklearn.preprocessing import RobustScaler
# from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
# from sklearn.model_selection import KFold, cross_val_score, train_test_split
# from sklearn.metrics import mean_squared_error
# import xgboost as xgb
# import lightgbm as lgb


# # Validation function
# def rmsle_cv(model, kfolds=5):
#     # https://scikit-learn.org/stable/modules/model_evaluation.html
#     kf = KFold(kfolds, shuffle=False, random_state=SEED).get_n_splits(xtr.values)
#     rmse= np.sqrt(-cross_val_score(estimator=model, X=xdata, y=ydata,
#                                    scoring='r2', cv=kf, n_jobs=-1))
#     return(rmse)


# ml_runtime = OrderedDict() # {}
# print('\nTrain Lasso ...')
# t0 = time.time()
# # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html
# lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=SEED))
# # lasso = Pipeline(steps=[('RobustScaler', RobustScaler()), ('Lasso', Lasso(alpha=0.0005, random_state=SEED))])
# # scaler = RobustScaler()
# # scaler.fit_transform(xdata)
# print(lasso.score(xvl, yvl))   
# ml_runtime['lasso'] = time.time() - t0
# print('lasso: {:.2f} mins'.format(ml_runtime['lasso']/60))

# print('\nTrain elastic net ...')
# # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html
# elnet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=0.9, random_state=SEED))
# print(elnet.score(xvl, yvl))   
# ml_runtime['elnet'] = time.time() - t0
# print('elnet: {:.2f} mins'.format(ml_runtime['elnet']/60))

# print('\nTrain kernel ridge ...')
# # https://scikit-learn.org/stable/modules/generated/sklearn.kernel_ridge.KernelRidge.html
# krr = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
# print(krr.score(xvl, yvl))   
# ml_runtime['krr'] = time.time() - t0
# print('krr: {:.2f} mins'.format(ml_runtime['krr']/60))

# print('\nTrain GBR ...')
# # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html
# gbr = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
#                                    max_depth=4, max_features='sqrt',
#                                    min_samples_leaf=15, min_samples_split=10, 
#                                    loss='huber', random_state =5)
# print(gbr.score(xvl, yvl))   
# ml_runtime['gbr'] = time.time() - t0
# print('gbr: {:.2f} mins'.format(ml_runtime['gbr']/60))



# =====================================================================================
# if __name__ == '__main__':
#     main()