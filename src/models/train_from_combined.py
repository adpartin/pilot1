"""
TODO:
1. ML models
- rank models based on performance
- optimize models using hyperparam search
  https://stats.stackexchange.com/questions/183984/how-to-use-xgboost-cv-with-hyperparameters-optimization
  https://github.com/raymon-tian/trend_ml_toolkit_xgboost/blob/master/xg_train_slower.py
  https://github.com/LevinJ/Supply-demand-forecasting/blob/master/utility/xgbbasemodel.py
  https://blog.cambridgespark.com/hyperparameter-tuning-in-xgboost-4ff9100a3b2f
- ensemble/stack models
  http://blog.kaggle.com/2017/06/15/stacking-made-easy-an-introduction-to-stacknet-by-competitions-grandmaster-marios-michailidis-kazanova/
  https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard
- best practices
  http://dnc1994.com/2016/05/rank-10-percent-in-first-kaggle-competition-en/

Explore AutoML:
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

3. Outliers and transformations
https://www.analyticsvidhya.com/blog/2015/11/8-ways-deal-continuous-variables-predictive-modeling/
https://scikit-learn.org/stable/auto_examples/plot_anomaly_comparison.html
https://scikit-learn.org/stable/auto_examples/neighbors/plot_lof_novelty_detection.html
- unskew the data; drop outliers based on boxplot (stratified by drug and tissue type)
- IsolationForest

4. A cluster-based feature(s):
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

7. Stratify and group
- https://github.com/scikit-learn/scikit-learn/pull/9413


Run-time problems:
When running on Mac, lightgbm gives an error:
- https://github.com/dmlc/xgboost/issues/1715
- https://lightgbm.readthedocs.io/en/latest/FAQ.html
- This has been solved by installing "nomkl":  conda install nomkl
- What is nomkl: https://docs.continuum.io/mkl-optimizations/
"""
from __future__ import print_function
from __future__ import division

import warnings
warnings.filterwarnings('ignore')

import os
import sys
import time
import datetime
import logging
import psutil
from pprint import pprint
from collections import OrderedDict
import numpy as np
import pandas as pd

import argparse
import configparser

import matplotlib
# matplotlib.use('TkAgg')
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats
import sklearn
from sklearn.preprocessing import Imputer, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import learning_curve
from sklearn.model_selection import cross_val_score, cross_validate, learning_curve
from sklearn.model_selection import ShuffleSplit, GroupShuffleSplit, KFold, GroupKFold

# Utils
# file_path = os.getcwd()
# file_path = os.path.join(file_path, 'src/models')
# os.chdir(file_path)

file_path = os.path.dirname(os.path.realpath(__file__))  # os.path.dirname(os.path.abspath(__file__))
# utils_path = os.path.abspath(os.path.join(file_path, 'utils'))
# sys.path.append(utils_path)
import utils
import utils_tidy
import argparser
import classlogger
import lrn_curve
from cvsplitter import GroupSplit, SimpleSplit, plot_ytr_yvl_dist

DATADIR = os.path.join(file_path, '../../data/processed/from_combined')
OUTDIR = os.path.join(file_path, '../../models/from_combined')
DATAFILENAME = 'tidy_data_no_fibro.parquet'
CONFIGFILENAME = 'config_params.txt'
# FILENAME = 'tidy_data.parquet'
os.makedirs(OUTDIR, exist_ok=True)

SEED = 0

# Feature prefix (some already present in the tidy dataframe)
fea_prfx_dict = {'rna': 'cell_rna.',
                 'cnv': 'cell_cnv.',
                 'dsc': 'drug_dsc.',
                 'fng': 'drug_fng.',
                 'clb': 'cell_lbl.',
                 'dlb': 'drug_lbl.'}

np.set_printoptions(precision=3)


def run(args):
    target_name = args.target_name
    target_trasform = args.target_trasform    
    train_sources = args.train_sources
    test_sources = args.test_sources
    row_sample = args.row_sample
    col_sample = args.col_sample
    tissue_type = args.tissue_type
    cell_features = args.cell_features
    drug_features = args.drug_features
    other_features = args.other_features
    model_name = args.ml_models
    cv_method = args.cv_method
    cv_folds = args.cv_folds
    verbose = args.verbose
    n_jobs = args.n_jobs

    # Feature list
    feature_list = cell_features + drug_features + other_features



    # ========================================================================
    #       Logger
    # ========================================================================
    t = datetime.datetime.now()
    t = [t.year, '-', t.month, '-', t.day, '_', 'h', t.hour, '-', 'm', t.minute]
    t = ''.join([str(i) for i in t])
    name_sufix = '~' + '.'.join(train_sources + [model_name] + [cv_method] + cell_features + drug_features + [target_name])
    run_outdir = os.path.join(OUTDIR, 'run_' + t + name_sufix)
    os.makedirs(run_outdir)
    logfilename = os.path.join(run_outdir, 'logfile.log')
    lg = classlogger.Logger(logfilename=logfilename)

    lg.logger.info(f'File path: {file_path}')
    lg.logger.info(f'System CPUs: {psutil.cpu_count()}')
    lg.logger.info(f'n_jobs: {n_jobs}')



    # ========================================================================
    #       Load data and pre-proc
    # ========================================================================
    datapath = os.path.join(DATADIR, DATAFILENAME)
    lg.logger.info(f'\nLoad tidy data ... {datapath}')
    data = pd.read_parquet(datapath, engine='auto', columns=None)
    lg.logger.info(f'data.shape {data.shape}')
    lg.logger.info('data memory usage (GB): {:.3f}'.format(sys.getsizeof(data)/1e9))
    # print(data.groupby('SOURCE').agg({'CELL': 'nunique', 'DRUG': 'nunique', 'PUBCHEM': 'nunique'}).reset_index())


    # Replace characters that are illegal for xgboost/lightgbm feature names
    # xdata.columns = list(map(lambda s: s.replace('[','_').replace(']','_'), xdata.columns.tolist())) # required by xgboost
    import re
    regex = re.compile(r'\[|\]|<', re.IGNORECASE)
    data.columns = [regex.sub('_', c) if any(x in str(c) for x in set(('[', ']', '<'))) else c for c in data.columns.values]


    if tissue_type is not None:
        data = data[data[''].isin([tissue_type])].reset_index(drop=True)


    # Subsample
    if row_sample:
        row_sample = eval(row_sample)
        data = utils.subsample(df=data, v=row_sample, axis=0)
        print('data.shape', data.shape)

    if col_sample:
        col_sample = eval(col_sample)
        data = utils.subsample(df=data, v=col_sample, axis=1)
        print('data.shape', data.shape)


    # Extract test sources
    lg.logger.info('\nExtract test sources ... {}'.format(test_sources))
    te_data = data[data['SOURCE'].isin(test_sources)].reset_index(drop=True)
    lg.logger.info(f'te_data.shape {te_data.shape}')
    lg.logger.info('data memory usage (GB): {:.3f}'.format(sys.getsizeof(te_data)/1e9))
    lg.logger.info(te_data.groupby('SOURCE').agg({'CELL': 'nunique', 'DRUG': 'nunique'}).reset_index())


    # Extract train sources
    lg.logger.info('\nExtract train sources ... {}'.format(train_sources))
    data = data[data['SOURCE'].isin(train_sources)].reset_index(drop=True)
    lg.logger.info(f'data.shape {data.shape}')
    lg.logger.info('data memory usage (GB): {:.3f}'.format(sys.getsizeof(data)/1e9))
    lg.logger.info(data.groupby('SOURCE').agg({'CELL': 'nunique', 'DRUG': 'nunique'}).reset_index())


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
    utils.plot_qq(x=data[target_name], var_name=target_name,
        path=os.path.join(run_outdir, target_name+'_qqplot.png'))
    utils.plot_hist_drugs(x=data['DRUG'], path=os.path.join(run_outdir, 'drugs_hist.png'))


    # Transform the target
    if target_trasform:
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


    if 'dlb' in other_features:
        lg.logger.info('\nAdd drug labels to features ...')
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
        lg.logger.info(f'dlb.shape {dlb.shape}')
        lg.logger.info(f'data.shape {data.shape}')


    if 'rna_clusters' in other_features:
        # TODO
        pass



    # ========================================================================
    #       TODO: outlier removal
    # ========================================================================
    pass



    # ========================================================================
    #       Keep a set of training features and impute missing values
    # ========================================================================
    data = utils_tidy.extract_subset_features(data=data, feature_list=feature_list, fea_prfx_dict=fea_prfx_dict)
    data = utils_tidy.impute_values(data=data, fea_prfx_dict=fea_prfx_dict, logger=lg.logger)



    # ========================================================================
    #       Initialize ML model
    # ========================================================================
    from ml_models import LGBM_REGRESSOR, RF_REGRESSOR
    def init_model(model_name, logger, verbose=False):
        if 'lgb_reg' in model_name:
            if verbose:
                logger.info('ML Model: lgb regressor')
            model = LGBM_REGRESSOR(n_jobs=n_jobs, random_state=SEED, logger=logger)
            fit_params = {'verbose': False}
        elif 'rf_reg' in model_name:
            if verbose:
                logger.info('ML Model: rf regressor')
            model = RF_REGRESSOR(n_jobs=4, random_state=SEED, logger=logger)
            fit_params = {'verbose': False, 'n_jobs': n_jobs, 'random_state': SEED}
        return model, fit_params


    # ========================================================================
    #       Run CV validation
    # ========================================================================
    lg.logger.info('\n{}'.format('='*50))
    lg.logger.info('Run CV training ...')
    lg.logger.info('='*50)

    # ------------
    # My CV method
    # ------------
    # from cvrun import my_cv_run
    # model, _ = init_model(model_name, logger=lg.logger)
    # tr_cv_scores, vl_cv_scores = my_cv_run(
    #     data=data,
    #     target_name=target_name,
    #     model=model.model,
    #     fea_prfx_dict=fea_prfx_dict,
    #     cv_method=cv_method, cv_folds=cv_folds,
    #     logger=lg.logger, verbose=True, random_state=SEED, outdir=run_outdir)


    # ----------------------------
    # sklearn CV method - method 1
    # ----------------------------
    # Split tr/vl data
    if cv_method=='simple':
        cv = KFold(n_splits=cv_folds, shuffle=False, random_state=SEED)
        groups = None
    elif cv_method=='group':
        cv = GroupKFold(n_splits=cv_folds)
        groups = data['CELL']
    elif cv_method=='stratify':
        pass
    else:
        raise ValueError(f'This cv_method ({cv_method}) is not supported')

    # Get the data
    xdata, _ = utils_tidy.split_features_and_other_cols(data, fea_prfx_dict=fea_prfx_dict)
    ydata = utils_tidy.extract_target(data=data, target_name=target_name)

    model, fit_params = init_model(model_name, logger=lg.logger)
    scores = cross_validate(
        estimator=sklearn.base.clone(model.model),
        X=xdata, y=ydata,
        scoring=['r2', 'neg_mean_absolute_error', 'neg_median_absolute_error'],
        cv=cv, groups=groups,
        n_jobs=n_jobs, fit_params=fit_params)
    scores.pop('fit_time', None)
    scores.pop('train_time', None)
    scores.pop('score_time', None)
    scores = utils.cv_scores_to_df(scores, decimals=3, calc_stats=True)
    lg.logger.info(f'scores\n{scores}')


    # ----------------------------
    # sklearn CV method - method 2
    # ----------------------------
    # # https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    # lgb_reg = LGBM_REGRESSOR(random_state=SEED, logger=lg.logger)
    # scores = cross_val_score(estimator=lgb_reg.model, X=xdata, y=ydata,
    #                          scoring='r2', cv=cv, n_jobs=n_jobs,
    #                          fit_params={'verbose': False, 'early_stopping_rounds': 10})    
    # lg.logger.info(scores)



    # ========================================================================
    #       Generate learning curves
    # ========================================================================
    lg.logger.info('\n{}'.format('='*50))
    lg.logger.info('Generate learning curves ...')
    lg.logger.info('='*50)

    # Prepare data
    xdata, _ = utils_tidy.split_features_and_other_cols(data, fea_prfx_dict=fea_prfx_dict)
    ydata = utils_tidy.extract_target(data=data, target_name=target_name)
    utils_tidy.print_feature_shapes(df=xdata, logger=lg.logger)

    # Split tr/vl data
    if cv_method=='simple':
        cv = KFold(n_splits=cv_folds, shuffle=False, random_state=SEED)
        groups = None
    elif cv_method=='group':
        cv = GroupKFold(n_splits=cv_folds)
        groups = data['CELL']
    elif cv_method=='stratify':
        pass
    else:
        raise ValueError('This cv_method ({}) is not supported'.format(cv_method))


    # -----------------------------------------------
    # Generate learning curve - my method
    # (*) ...
    # -----------------------------------------------
    # from cvrun import my_cv_run
    # df_tr = []
    # df_vl = []
    # lr_curve_ticks = 5
    # data_sizes_frac = np.linspace(0.1, 1.0, lr_curve_ticks)
    # data_sizes = [int(n) for n in data.shape[0]*data_sizes_frac]
    
    # model, _ = init_model(model_name, logger=lg.logger)
    # for d_size in data_sizes:
    #     lg.logger.info(f'Data size: {d_size}')
    #     data_sample = data.sample(n=d_size)

    #     tr_cv_scores, vl_cv_scores = my_cv_run(
    #         data=data_sample,
    #         target_name=target_name,
    #         fea_prfx_dict=fea_prfx_dict,
    #         model=model.model, #model_name=model_name,
    #         cv_method=cv_method, cv_folds=cv_folds,
    #         logger=lg.logger, random_state=SEED, outdir=run_outdir)

    #     # Add col that indicates d_size
    #     tr_cv_scores.insert(loc=1, column='data_size', value=data_sample.shape[0])
    #     vl_cv_scores.insert(loc=1, column='data_size', value=data_sample.shape[0])
        
    #     # Append results to master dfs
    #     df_tr.append(tr_cv_scores)
    #     df_vl.append(vl_cv_scores)

    # # Concat the results for all the data_sizes
    # df_tr = pd.concat(df_tr, axis=0)
    # df_vl = pd.concat(df_vl, axis=0)

    # lrn_curve.plt_learning_curve_multi_metric(df_tr=df_tr, df_vl=df_vl,
    #                                           cv_folds=cv_folds, target_name=target_name,
    #                                           outdir=run_outdir)


    # -----------------------------------------------
    # Generate learning curve - semi automatic method
    # (*) uses cross_validate from sklearn.
    # -----------------------------------------------
    # Generate learning curves
    model, fit_params = init_model(model_name='lgb_reg', logger=lg.logger)
    lrn_curve.my_learning_curve(
        estimator=model.model,
        X=xdata, Y=ydata,  # data
        target_name=target_name,
        fit_params=fit_params,
        lr_curve_ticks=5, data_sizes_frac=None,
        metrics=['r2', 'neg_mean_absolute_error', 'neg_median_absolute_error'],
        cv_method=cv_method, cv_folds=cv_folds, groups=None,
        n_jobs=n_jobs, random_state=SEED, logger=lg.logger, outdir=run_outdir)


    # -------------------------------------------------
    # Generate learning curve - complete sklearn method
    # (*) can't generate multiple metrics.
    # -------------------------------------------------
    lg.logger.info("\nStart sklearn.model_selection.learning_curve ...")
    model, _ = init_model(model_name, lg.logger)
    metric_name = 'r2' # 'neg_mean_absolute_error', 'neg_median_absolute_error'
    lr_curve_ticks = 5
    train_sizes_frac = np.linspace(0.1, 1.0, lr_curve_ticks)
    t0 = time.time()
    rslt = learning_curve(estimator=model.model, X=xdata, y=ydata,
                          train_sizes=train_sizes_frac, cv=cv, groups=groups,
                          scoring=metric_name,
                          n_jobs=n_jobs, exploit_incremental_learning=False,
                          random_state=SEED, verbose=1, shuffle=False)
    lg.logger.info('Run-time: {:.3f} mins'.format((time.time()-t0)/60))
    
    lrn_curve.plt_learning_curve(rslt=rslt, metric_name=metric_name,
        title='Learning curve (target: {})'.format(target_name),
        path=os.path.join(run_outdir, 'auto_learning_curve_' + metric_name + '.png'))



    # ========================================================================
    #       Train final model (entire dataset)
    # ========================================================================
    lg.logger.info('\n{}'.format('='*50))
    lg.logger.info(f'Train final model (use entire dataset) ... {train_sources}')
    lg.logger.info('='*50)

    xdata, _ = utils_tidy.split_features_and_other_cols(data, fea_prfx_dict=fea_prfx_dict)
    ydata = utils_tidy.extract_target(data=data, target_name=target_name)
    utils_tidy.print_feature_shapes(df=xdata, logger=lg.logger)

    # Train model
    model_final, _ = init_model(model_name, lg.logger)
    if 'lgb_reg' in model_name:
        model_final.fit(xdata, ydata, eval_set=[(xdata, ydata)])
    else:
        model_final.fit(xdata, ydata)

    # # Compute scores
    # scores = model_final.calc_scores(xdata=xdata, ydata=ydata, to_print=True)

    # # Dump preds
    # model_final.dump_preds(df_data=data, xdata=xdata, target_name=target_name,
    #                        outpath=os.path.join(run_outdir, 'preds.csv'))

    # # Save model
    # model_final.save_model(outdir=run_outdir)



    # ========================================================================
    #       Infer
    # ========================================================================
    lg.logger.info('\n{}'.format('='*50))
    lg.logger.info(f'Inference ... {test_sources}')
    lg.logger.info('='*50)

    csv_scores = []  # cross-study-validation scores
    for i, src in enumerate(test_sources):
        lg.logger.info(f'\nTest source {i+1}:  _____ {src} _____')

        te_src_data = te_data[te_data['SOURCE'].isin([src])].reset_index(drop=True)
        lg.logger.info(f'src_data.shape {te_src_data.shape}')
        if te_src_data.shape[0] == 0:
            continue

        preds_filename_prefix = 'test_' + src
        model_name = 'lgb_reg_final'

        # Prepare test data for predictions
        te_src_data = utils_tidy.impute_values(data=te_src_data, fea_prfx_dict=fea_prfx_dict, logger=lg.logger)
        xte, _ = utils_tidy.split_features_and_other_cols(te_src_data, fea_prfx_dict=fea_prfx_dict)
        yte = utils_tidy.extract_target(data=te_src_data, target_name=target_name)

        # Plot dist of response
        utils.plot_hist(x=te_src_data[target_name], var_name=target_name,
                        path=os.path.join(run_outdir, src+'_'+target_name+'_hist.png'))

        # Print feature shapes
        lg.logger.info(f'\nxte_'+src)
        utils_tidy.print_feature_shapes(df=xte, logger=lg.logger)

        # Compute scores
        lg.logger.info('\nscores:')
        scores = model_final.calc_scores(xdata=xte, ydata=yte, to_print=True)
        scores = utils.cv_scores_to_df([scores])

        # Dump preds
        model_final.dump_preds(df_data=te_src_data, xdata=xte, target_name=target_name,
                               outpath=os.path.join(run_outdir, preds_filename_prefix+'_'+model_name+'_preds.csv'))                 

        # Calc and save scores
        csv_scores.append(model_final.calc_scores(xdata=xte, ydata=yte, to_print=False))


    # Summarize cv scores
    df_csv_scores = pd.DataFrame(csv_scores).T
    df_csv_scores.columns = test_sources
    df_csv_scores = df_csv_scores.reset_index().rename(columns={'index': 'metric'})
    df_csv_scores.insert(loc=1, column='train_src', value='_'.join(train_sources))
    lg.logger.info('\ncsv_scores\n{}'.format(df_csv_scores))
    df_csv_scores.to_csv(os.path.join(run_outdir, 'df_csv_scores.csv'), index=False)


    # ========================================================================
    #       Grid Search CV (TODO: didn't finish)
    # ========================================================================
    # # https://scikit-learn.org/stable/tutorial/statistical_inference/model_selection.html
    # from sklearn.neighbors import KNeighborsRegressor
    # from sklearn.svm import LinearSVR, SVR, NuSVR
    # from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler, QuantileTransformer
    # from sklearn.model_selection import GridSearchCV, cross_val_score
    # from sklearn.model_selection import GroupKFold

    # xdata, _ = utils_tidy.split_features_and_other_cols(data, fea_prfx_dict=fea_prfx_dict)
    # ydata = utils_tidy.extract_target(data=data, target_name=target_name)
    # xdata = StandardScaler().fit_transform(xdata)

    # # SVR
    # model_name = 'svr'
    # params_search = {'C': [0.1, 1, 10],
    #                  'epsilon': [0.01, 0.1, 1]}
    # model = SVR(gamma='scale', kernel='rbf')

    # # KNN
    # model_name = 'knn_reg'
    # params_search = {'n_neighbors': [5, 10, 15]}
    # model = KNeighborsRegressor(n_jobs=n_jobs)

    # # CV splitter
    # groups = data['CELL'].values
    # groups = LabelEncoder().fit_transform(groups)
    # cv_splitter = GroupKFold(n_splits=n_splits)
    # cv_splitter = cv_splitter.split(data, groups=groups)

    # # Define grid search
    # gs = GridSearchCV(estimator=model, param_grid=params_search,
    #                   n_jobs=n_jobs, refit=True, cv=cv_splitter, verbose=True)

    # # Train
    # t0 = time.time()
    # gs.fit(xdata, ydata)
    # runtime_fit = time.time() - t0
    # print('runtime_fit', runtime_fit)

    # # Get the best model
    # best_model = gs.best_estimator_
    # results = pd.DataFrame(gs.cv_results_)
    # print('results:\n{}'.format(results))
    # print('{} best score (random search): {:.3f}'.format(model_name, gs.best_score_))
    # print('{} best params (random search): \n{}'.format(model_name, gs.best_params_))



    # ========================================================================
    #       Ensemble models
    # ========================================================================
    # pass                         

    # Kill logger
    lg.kill_logger()

    return df_csv_scores


def main(args):
    config_fname = os.path.join(file_path, CONFIGFILENAME)
    args = argparser.get_args(args=args, config_fname=config_fname)
    pprint(vars(args))
    df_csv_scores = run(args)
    return df_csv_scores, OUTDIR  # TODO: instead of OUTDIR, return all globals(??)
    

if __name__ == '__main__':
    # python -m pdb src/models/train_from_combined.py -te ccle gcsi -tr gcsi
    """ __name__ == '__main__' explained:
    https://www.youtube.com/watch?v=sugvnHA7ElY
    """
    """
    https://stackoverflow.com/questions/14500183/in-python-can-i-call-the-main-of-an-imported-module
    How to run code with input args from another code?
    This will be used with multiple train and test sources.
    For example: in launch_model_transfer.py
        import train_from_combined.py
        train_from_combined.main([tr_src, tst_src])
    """
    # python -m pdb src/models/train_from_combined.py -te ccle gcsi -tr gcsi
    main(sys.argv[1:])
