"""
TODO:
1. Multiple ML models
- train multiple models
  create super class ML_Model (this sklearn API class will implement hyperparam grid search)
- rank models based on performance
- optimize each model using hyperparam search
  https://stats.stackexchange.com/questions/183984/how-to-use-xgboost-cv-with-hyperparameters-optimization
  https://github.com/raymon-tian/trend_ml_toolkit_xgboost/blob/master/xg_train_slower.py
  https://github.com/LevinJ/Supply-demand-forecasting/blob/master/utility/xgbbasemodel.py
- ensemble/stack models
  http://blog.kaggle.com/2017/06/15/stacking-made-easy-an-introduction-to-stacknet-by-competitions-grandmaster-marios-michailidis-kazanova/
  https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard

Explore Auto-ML models:
- tpot
- auto-sklearn
- data robot

ML models:
- NN (consider various normalization methods) - https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html
- xgboost (gbtree or gblinear)
- lightgbm
- catboost
- RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor
- naive bayes
- svm
- knn
- elastic net
- use features generated using t-SNE, PCA, etc.

Hyperparam schemes:
- CANDLE
- https://medium.com/@mikkokotila/a-comprehensive-list-of-hyperparameter-optimization-tuning-solutions-88e067f19d9

2. Learning curve (performance vs training set size)
https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html

3. Outliers and transformations
https://www.analyticsvidhya.com/blog/2015/11/8-ways-deal-continuous-variables-predictive-modeling/
- unskew the data; drop outliers based on boxplot (stratified by drug and tissue type)
- IsolationForest

4. Another feature to add would be cluster-based:
http://blog.kaggle.com/2015/07/27/taxi-trajectory-winners-interview-1st-place-team-%F0%9F%9A%95/
- Apply clustering to rna-seq. The clusters vector will become a categorical variable. In this case
  we avoid using tissue type labels but rather use proximity in the actual feature space.

5. Features; data pre-processing
- create code preproc_tidy_data.py
- rna-seq clusters
- bin descriptors
- embedding on mutation data
- imputation --> create boolean indicator of NA values

6. Feature importance
- Explore X_SHAP_values in predict() method in lightgbm

Run-time problems:
When running on Mac, lightgbm gives an error:
- https://github.com/dmlc/xgboost/issues/1715
- https://lightgbm.readthedocs.io/en/latest/FAQ.html
- This has been solved by installing "nomkl":  conda install nomkl
- What is nomkl: https://docs.continuum.io/mkl-optimizations/
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
from sklearn.model_selection import learning_curve, KFold, StratifiedKFold

# Utils
file_path = os.getcwd()
file_path = os.path.join(file_path, 'src/models')
os.chdir(file_path)

# DATADIR = './tidy_data_from_combined'
# FILENAME = 'tidy_data.parquet'
# OUTDIR = os.path.join(file_path, 'ml_tidy_combined')
# os.makedirs(OUTDIR, exist_ok=True)

# file_path = os.path.dirname(os.path.realpath(__file__))  # os.path.dirname(os.path.abspath(__file__))
##utils_path = os.path.abspath(os.path.join(file_path, 'utils'))
##sys.path.append(utils_path)
import utils_models as utils

DATADIR = os.path.join(file_path, '../../data/processed/from_combined')
OUTDIR = os.path.join(file_path, '../../models/from_combined')
FILENAME = 'tidy_data_no_fibro.parquet'
# FILENAME = 'tidy_data.parquet'
os.makedirs(OUTDIR, exist_ok=True)

SEED = 0


# ========================================================================
#       Args TODO: add to argparse
# ========================================================================
# Train and infer data
train_sources = ['ccle']  # ['ccle', 'gcsi', 'gdsc', 'ctrp']
infer_sources = ['gcsi']

# Traget (response)
# target_name = 'AUC'
target_name = 'AUC1'

# Features to use
# TODO: instead of using these names, just use the values of fea_prefix dict
cell_features = ['rna'] # ['rna', cnv', 'rna_latent']
drug_features = ['dsc'] # [] # ['dsc', 'fng', 'dsc_latent', 'fng_latent']
other_features = [] # ['drug_labels'] # ['cell_labels', 'drug_labels', 'ctype', 'csite', 'rna_clusters']
feature_list = cell_features + drug_features + other_features

# Feature prefix (some already present in the tidy dataframe)
fea_prfx_dict = {'rna': 'cell_rna.',
                 'cnv': 'cell_cnv.',
                 'dsc': 'drug_dsc.',
                 'fng': 'drug_fng.',
                 'clb': 'cell_lbl.',
                 'dlb': 'drug_lbl.'}
# fea_prefix = {'rna': 'cell_rna',
#               'cnv': 'cell_cnv',
#               'dsc': 'drug_dsc',
#               'fng': 'drug_fng'}

# Models
# ml_models = ['tpot_reg']
ml_models = ['lgb_reg']

trasform_target = True
outlier_remove = False  # IsolationForest
verbose = True
n_jobs = 4



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
data = pd.read_parquet(datapath, engine='auto', columns=None)
logger.info(f'data.shape {data.shape}')
logger.info('data memory usage (GB): {:.3f}'.format(sys.getsizeof(data)/1e9))
# print(data.groupby('SOURCE').agg({'CELL': 'nunique', 'DRUG': 'nunique', 'PUBCHEM': 'nunique'}).reset_index())


# Replace characters that are illegal for xgboost feature names
# xdata.columns = list(map(lambda s: s.replace('[','_').replace(']','_'), xdata.columns.tolist())) # required by xgboost
import re
regex = re.compile(r'\[|\]|<', re.IGNORECASE)
data.columns = [regex.sub('_', c) if any(x in str(c) for x in set(('[', ']', '<'))) else c for c in data.columns.values]


# Extract infer sources
logger.info('\nExtract infer sources ... {}'.format(infer_sources))
te_data = data[data['SOURCE'].isin(infer_sources)].reset_index(drop=True)
logger.info(f'te_data.shape {te_data.shape}')
logger.info('data memory usage (GB): {:.3f}'.format(sys.getsizeof(te_data)/1e9))
# print(data.groupby('SOURCE').agg({'CELL': 'nunique', 'DRUG': 'nunique'}).reset_index())


# Extract training sources
logger.info('\nExtract train sources ... {}'.format(train_sources))
data = data[data['SOURCE'].isin(train_sources)].reset_index(drop=True)
logger.info(f'data.shape {data.shape}')
logger.info('data memory usage (GB): {:.3f}'.format(sys.getsizeof(data)/1e9))
# print(data.groupby('SOURCE').agg({'CELL': 'nunique', 'DRUG': 'nunique'}).reset_index())


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
if trasform_target:
    y = data[target_name].copy()
    # y = np.log1p(ydata); plot_hist(x=y, var_name=target_name+'_log1p')
    # # y = np.log(ydata+1); plot_hist(x=y, var_name=target_name+'_log+1')
    # y = np.log10(ydata+1); plot_hist(x=y, var_name=target_name+'_log10')
    # y = np.log2(ydata+1); plot_hist(x=y, var_name=target_name+'_log2')
    # y = ydata**2; plot_hist(x=ydata, var_name=target_name+'_x^2')
    y, lmbda = stats.boxcox(y+1); # utils.plot_hist(x=y, var_name=target_name+'_boxcox', path=)
    data[target_name] = y
    # ydata = pd.DataFrame(y)

    y = te_data[target_name].copy()
    y, lmbda = stats.boxcox(y+1); # utils.plot_hist(x=y, var_name=target_name+'_boxcox', path=)
    te_data[target_name] = y


if 'drug_labels' in other_features:
    # print('\nAdd drug labels to features ...')
    logger.info('\nAdd drug labels to features ...')
    # print(data['DRUG'].value_counts())

    # http://queirozf.com/entries/one-hot-encoding-a-feature-on-a-pandas-dataframe-an-example
    # One-hot encoder
    drug_labels = pd.get_dummies(data=data[['DRUG']], prefix=fea_prfx_dict['dlb'],
                                 dummy_na=False).reset_index(drop=True)

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
# TODO: modify impute_values to accept feature_list instead of fea_prefix!!
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
from sklearn.metrics import r2_score, mean_absolute_error, median_absolute_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, ParameterGrid
from sklearn.externals import joblib

train_runtime = OrderedDict() # {}
preds_filename_prefix = 'dev'

# ---------------------
# RandomForestRegressor
# ---------------------
if 'rf_reg' in ml_models:
    model_name = 'rf_reg'
    try:
        from sklearn.ensemble import RandomForestRegressor
    except ImportError:
        # install??
        logger.error(f'Module not found (RandomForestRegressor)')

    logger.info('\nTrain RandomForestRegressor ...')
    # ----- rf hyper-param start
    rf_reg = RandomForestRegressor(max_features='sqrt', bootstrap=True, oob_score=True,
                                verbose=0, random_state=SEED, n_jobs=n_jobs)

    random_search_params = {'n_estimators': [100, 500, 1000], # [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
                            'max_depth': [None, 5, 10], # [None] + [int(x) for x in np.linspace(10, 110, num = 11)]
                            'min_samples_split': [2, 5, 9]}
    logger.info('hyper-params:\n{}'.format(random_search_params))

    rf_reg_randsearch = RandomizedSearchCV(
        estimator=rf_reg,
        param_distributions=random_search_params,
        n_iter=20,  # num of parameter settings that are sampled and used for training (num of models trained)
        scoring=None, # string or callable used to evaluate the predictions on the test set
        n_jobs=n_jobs,
        cv=5,
        refit=True,  # Refit an estimator using the best found parameters on the whole dataset
        verbose=0)

    # Run search
    t0 = time.time()
    rf_reg_randsearch.fit(xtr, ytr)
    train_runtime[model_name+'_randsearch'] = time.time() - t0
    logger.info('Runtime: {:.2f} mins'.format(train_runtime[model_name+'_randsearch']/60))

    # Save best model
    rf_reg = rf_reg_randsearch.best_estimator_
    joblib.dump(rf_reg, filename=os.path.join(run_outdir, model_name+'_hypsearch_best_model.pkl'))

    # Print preds
    utils.print_scores(model=rf_reg, xdata=xvl, ydata=yvl, logger=logger)

    # Save resutls
    rf_reg_hypsearch = pd.DataFrame(rf_reg_randsearch.cv_results_)
    rf_reg_hypsearch.to_csv(os.path.join(run_outdir, model_name+'_hypsearch_summary.csv'))  # save hyperparam search results

    logger.info(f'{model_name} best score (random search): {rf_reg_randsearch.best_score_:.3f}')
    logger.info('{} best params (random search): \n{}'.format(model_name, rf_reg_randsearch.best_params_))

    # Dump preds
    utils.dump_preds(model=rf_reg, df_data=vl_data, xdata=xvl, target_name=target_name,
                    path=os.path.join(run_outdir, preds_filename_prefix+'_'+model_name+'_preds.csv'))
    # ----- rf hyper-param end


# ------------
# XGBRegressor
# ------------
if 'xgb_reg' in ml_models:
    try:
        import xgboost as xgb
    except ImportError:  # install??
        logger.error('Module not found (xgboost)')

    # https://xgboost.readthedocs.io/en/latest/python/python_api.html
    # xgboost does not support categorical features!
    # Rules of thumb
    # 1. learning_rate should be 0.1 or lower (smaller values will require more trees).
    # 2. tree_depth should be between 2 and 8 (where not much benefit is seen with deeper trees).
    # 3. subsample should be between 30% and 80% of the training dataset, and compared to a value of 100% for no sampling.
    logger.info('\nTrain XGBRegressor ...')
    # xgb_tr = xgb.DMatrix(data=xtr, label=ytr, nthread=n_jobs)
    # xgb_vl = xgb.DMatrix(data=xvl, label=yvl, nthread=n_jobs)
    # ----- xgboost hyper-param start
    xgb_reg = xgb.XGBRegressor(objective='reg:linear', # default: 'reg:linear', TODO: docs recommend funcs for different distributions (??)
                            booster='gbtree', # default: gbtree (others: gblinear, dart)
                            # max_depth=3, # default: 3
                            # learning_rate=0.1, # default: 0.1
                            # n_estimators=100, # default: 100
                            n_jobs=n_jobs, # default: 1
                            reg_alpha=0, # default=0, L1 regularization
                            reg_lambda=1, # default=1, L2 regularization
                            random_state=SEED)

    random_search_params = {'n_estimators': [30, 50, 70],
                            'learning_rate': [0.005, 0.01, 0.5],
                            'subsample': [0.5, 0.7, 0.8],
                            'max_depth': [2, 3, 5]}
    logger.info('hyper-params:\n{}'.format(random_search_params))

    xgb_reg_randsearch = RandomizedSearchCV(
        estimator=xgb_reg,
        param_distributions=random_search_params,
        n_iter=20,  # num of parameter settings that are sampled and used for training (num of models trained)
        scoring=None, # string or callable used to evaluate the predictions on the test set
        n_jobs=n_jobs,
        cv=5,
        refit=True,  # Refit an estimator using the best found parameters on the whole dataset
        verbose=False)   

    # Start search
    t0 = time.time()
    xgb_reg_randsearch.fit(xtr, ytr)
    train_runtime['xgb_reg_randsearch'] = time.time() - t0
    logger.info('Runtime: {:.2f} mins'.format(train_runtime['xgb_reg_randsearch']/60))

    # Save best model
    xgb_reg = xgb_reg_randsearch.best_estimator_
    joblib.dump(xgb_reg, filename=os.path.join(run_outdir, 'xgb_reg_hypsearch_best_model.pkl'))

    # Print preds
    utils.print_scores(model=xgb_reg, xdata=xvl, ydata=yvl, logger=logger)

    # Save resutls
    xgb_reg_hypsearch = pd.DataFrame(xgb_reg_randsearch.cv_results_)
    xgb_reg_hypsearch.to_csv(os.path.join(run_outdir, 'xgb_reg_hypsearch_summary.csv'))  # save hyperparam search results

    logger.info(f'rf_reg best score (random search): {xgb_reg_randsearch.best_score_:.3f}')
    logger.info('rf_reg best params (random search): \n{}'.format(xgb_reg_randsearch.best_params_))

    # Dump preds
    utils.dump_preds(model=xgb_reg, df_data=vl_data, xdata=xvl, target_name=target_name,
                    path=os.path.join(run_outdir, 'xgb_vl_preds.csv'))
    # ----- xgboost hyper-param end

    # ----- xgboost "Sklearn API" start
    xgb_reg = xgb.XGBRegressor(objective='reg:linear', # default: 'reg:linear', TODO: docs recommend funcs for different distributions (??)
                            booster='gbtree', # default: gbtree (others: gblinear, dart)
                            max_depth=3, # default: 3
                            learning_rate=0.1, # default: 0.1
                            n_estimators=100, # default: 100
                            n_jobs=n_jobs, # default: 1
                            reg_alpha=0, # default=0, L1 regularization
                            reg_lambda=1, # default=1, L2 regularization
                            random_state=SEED
    )
    eval_metric = ['mae', 'rmse']
    t0 = time.time()
    xgb_reg.fit(xtr, ytr, eval_metric=eval_metric, eval_set=[(xtr, ytr), (xvl, yvl)],
                early_stopping_rounds=10, verbose=False, callbacks=None)
    train_runtime['xgb_reg'] = time.time() - t0
    logger.info('Runtime: {:.2f} mins'.format(train_runtime['xgb_reg']/60))

    # Save model
    # xgb_reg.save_model(os.path.join(run_outdir, 'xgb_reg.model'))
    joblib.dump(xgb_reg, filename=os.path.join(run_outdir, 'xgb_reg_model.pkl'))
    # xgb_reg_ = joblib.load(filename=os.path.join(run_outdir, 'xgb_reg_model.pkl'))

    # Print preds
    utils.print_scores(model=xgb_reg, xdata=xvl, ydata=yvl, logger=logger)

    # Dump preds
    utils.dump_preds(model=xgb_reg, df_data=vl_data, xdata=xvl, target_name=target_name,
                    path=os.path.join(run_outdir, 'xgb_vl_preds.csv'))
    # ----- xgboost "Sklearn API" end
        
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


# -------------
# TPOTRegressor
# -------------
# Total evaluation pipelines:  population_size + generations × offspring_size 
if 'tpot_reg' in ml_models:
    try:
        import tpot
    except ImportError:
        logger.error('Module not found (tpot)')
    
    tpot_checkpoint_folder = os.path.join(run_outdir, 'tpot_reg_checkpoints')
    os.makedirs(tpot_checkpoint_folder)

    logger.info('\nTrain TPOTRegressor ...')
    tpot_reg = tpot.TPOTRegressor(generations=100,  # dflt: 100
                                  population_size=100, # dflt: 100
                                  offspring_size=100, # dflt: 100
                                  scoring='neg_mean_squared_error', # dflt: 'neg_mean_squared_error'
                                  cv=5,
                                  n_jobs=n_jobs,
                                  random_state=SEED,
                                  warm_start=False,
                                  periodic_checkpoint_folder=tpot_checkpoint_folder,
                                  verbosity=2,
                                  disable_update_check=True)
    t0 = time.time()
    tpot_reg.fit(xtr, ytr)
    train_runtime['tpot_reg'] = time.time() - t0
    logger.info('Runtime: {:.2f} mins'.format(ml_runtime['tpot_reg']/60))
    
    # Export model as .py script
    tpot_reg.export(os.path.join(run_outdir, 'tpot_reg_pipeline.py'))

    # Print scores
    utils.print_scores(model=tpot_reg, xdata=xvl, ydata=yvl, logger=logger)

    # Dump preds
    t0 = time.time()
    utils.dump_preds(model=tpot_reg, df_data=vl_data, xdata=xvl, target_name=target_name,
                    path=os.path.join(run_outdir, 'tpot_reg_vl_preds.csv'))
    logger.info('Predictions runtime: {:.2f} mins'.format(time.time()/60))





# ========================================================================
#       Infer
# ========================================================================
logger.info('\n=====================================================')
logger.info(f'Inference ... {infer_sources}')
logger.info('=====================================================')

preds_filename_prefix = 'infer'
model_name = 'lgb_reg'

# Prepare infer data for predictions
##te_data = utils.impute_values(data=te_data, fea_prefix=fea_prefix, logger=logger)
te_data = utils.impute_values(data=te_data, fea_prfx_dict=fea_prfx_dict, logger=logger)
##xte = utils.extract_features(data=te_data, feature_list=feature_list, fea_prefix=fea_prefix)
xte, _ = utils.split_features_and_other_cols(te_data, fea_prfx_dict=fea_prfx_dict)
yte = utils.extract_target(data=te_data, target_name=target_name)

# Print feature shapes
print_feature_shapes(df=xte, name='xte')

# Print preds
utils.print_scores(model=lgb_reg, xdata=xte, ydata=yte, logger=logger)

# Dump preds
utils.dump_preds(model=lgb_reg, df_data=te_data, xdata=xte, target_name=target_name,
                 path=os.path.join(run_outdir, preds_filename_prefix+'_'+model_name+'_preds.csv'))





# =====================================================================================
# if __name__ == '__main__':
#     main()