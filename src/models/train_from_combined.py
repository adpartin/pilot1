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
from __future__ import division
from __future__ import print_function

import warnings
warnings.filterwarnings('ignore')

import os
import sys
import time
import datetime
import logging
import psutil
from collections import OrderedDict
import numpy as np
import pandas as pd

import argparse
import configparser
# import configargparse

import matplotlib
# matplotlib.use('TkAgg')
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats
from sklearn.preprocessing import Imputer, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import learning_curve

# Utils
# file_path = os.getcwd()
# file_path = os.path.join(file_path, 'src/models')
# os.chdir(file_path)

file_path = os.path.dirname(os.path.realpath(__file__))  # os.path.dirname(os.path.abspath(__file__))
##utils_path = os.path.abspath(os.path.join(file_path, 'utils'))
##sys.path.append(utils_path)
import utils
import utils_tidy
import arg_parser

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


def run(args):
    print(args)
    train_sources = args.train_sources
    test_sources = args.test_sources
    target_name = args.target_name
    target_trasform = args.target_trasform
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
    name_sufix = '~' + '.'.join(model_name + [cv_method] + cell_features + drug_features + [target_name])
    run_outdir = os.path.join(OUTDIR, 'run_' + t + name_sufix)
    os.makedirs(run_outdir)
    logfilename = os.path.join(run_outdir, 'logfile.log')
    logger = utils.setup_logger(logfilename=logfilename)

    logger.info(f'File path: {file_path}')
    logger.info(f'System CPUs: {psutil.cpu_count()}')
    logger.info(f'n_jobs: {n_jobs}')



    # ========================================================================
    #       Load data and pre-proc
    # ========================================================================
    datapath = os.path.join(DATADIR, DATAFILENAME)
    logger.info(f'\nLoad tidy data ... {datapath}')
    data = pd.read_parquet(datapath, engine='auto', columns=None)
    logger.info(f'data.shape {data.shape}')
    logger.info('data memory usage (GB): {:.3f}'.format(sys.getsizeof(data)/1e9))
    # print(data.groupby('SOURCE').agg({'CELL': 'nunique', 'DRUG': 'nunique', 'PUBCHEM': 'nunique'}).reset_index())


    # Replace characters that are illegal for xgboost/lightgbm feature names
    # xdata.columns = list(map(lambda s: s.replace('[','_').replace(']','_'), xdata.columns.tolist())) # required by xgboost
    import re
    regex = re.compile(r'\[|\]|<', re.IGNORECASE)
    data.columns = [regex.sub('_', c) if any(x in str(c) for x in set(('[', ']', '<'))) else c for c in data.columns.values]


    # Extract test sources
    logger.info('\nExtract test sources ... {}'.format(test_sources))
    te_data = data[data['SOURCE'].isin(test_sources)].reset_index(drop=True)
    logger.info(f'te_data.shape {te_data.shape}')
    logger.info('data memory usage (GB): {:.3f}'.format(sys.getsizeof(te_data)/1e9))
    logger.info(te_data.groupby('SOURCE').agg({'CELL': 'nunique', 'DRUG': 'nunique'}).reset_index())


    # Extract train sources
    logger.info('\nExtract train sources ... {}'.format(train_sources))
    data = data[data['SOURCE'].isin(train_sources)].reset_index(drop=True)
    logger.info(f'data.shape {data.shape}')
    logger.info('data memory usage (GB): {:.3f}'.format(sys.getsizeof(data)/1e9))
    logger.info(data.groupby('SOURCE').agg({'CELL': 'nunique', 'DRUG': 'nunique'}).reset_index())


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
    data = utils_tidy.impute_values(data=data, fea_prfx_dict=fea_prfx_dict, logger=logger)



    # ========================================================================
    #       Initialize ML model
    # ========================================================================
    from ml_models import LGBM_REGRESSOR, RF_REGRESSOR
    def init_model(model_name, logger):
        if 'lgb_reg' in model_name:
            logger.info('ML Model: lgb regressor')
            model = LGBM_REGRESSOR(random_state=SEED, logger=logger)
            fit_params = {'verbose': False}
        elif 'rf_reg' in model_name:
            logger.info('ML Model: rf regressor')
            model = RF_REGRESSOR(random_state=SEED, logger=logger)
            fit_params = None
        return model, fit_params


    # ========================================================================
    #       Run CV validation (my implementation)
    # ========================================================================
    # logger.info('\n=====================================================')
    # logger.info(f'Run CV validation (my implementation) ...')
    # logger.info('=====================================================')
    from split_tr_vl import GroupSplit, SimpleSplit, plot_ytr_yvl_dist
    # from ml_models import LGBM_REGRESSOR, RF_REGRESSOR


    # # Split tr/vl data
    # if cv_method=='simple':
    #     splitter = SimpleSplit(n_splits=cv_folds, random_state=SEED)
    #     splitter.split(X=data)
    # elif cv_method=='group':
    #     splitter = GroupSplit(n_splits=cv_folds, random_state=SEED)
    #     splitter.split(X=data, groups=data['CELL'])
    # elif cv_method=='stratify':
    #     pass
    # else:
    #     raise ValueError('This cv_method ({}) is not supported'.format(cv_method))


    # # Run CV training
    # logger.info(f'\nCV splitting method: {cv_method}')
    # tr_scores = []
    # vl_scores = []
    # for i in range(splitter.n_splits):
    #     logger.info(f'\nFold {i+1}/{splitter.n_splits}')
    #     tr_idx = splitter.tr_cv_idx[i]
    #     vl_idx = splitter.vl_cv_idx[i]
    #     tr_data = data.iloc[tr_idx, :]
    #     vl_data = data.iloc[vl_idx, :]

    #     # print(tr_idx[:5])
    #     # print(vl_idx[:5])

    #     # tr_cells = set(tr_data['CELL'].values)
    #     # vl_cells = set(vl_data['CELL'].values)
    #     # print('total cell intersections btw tr and vl: ', len(tr_cells.intersection(vl_cells)))
    #     # print('a few intersections : ', list(tr_cells.intersection(vl_cells))[:3])

    #     xtr, _ = utils_tidy.split_features_and_other_cols(tr_data, fea_prfx_dict=fea_prfx_dict)
    #     xvl, _ = utils_tidy.split_features_and_other_cols(vl_data, fea_prfx_dict=fea_prfx_dict)

    #     # utils_tidy.print_feature_shapes(df=xtr, logger=logger)
    #     # utils_tidy.print_feature_shapes(df=xvl, logger=logger)

    #     ytr = utils_tidy.extract_target(data=tr_data, target_name=target_name)
    #     yvl = utils_tidy.extract_target(data=vl_data, target_name=target_name)

    #     title = f'{target_name}; split {str(i)}'
    #     plot_ytr_yvl_dist(ytr, yvl, title=title, outpath=os.path.join(run_outdir, title+'.png'))

    #     # Train model
    #     lgb_reg = LGBM_Regressor(target_name=target_name, random_state=SEED, logger=logger)
    #     lgb_reg.fit(xtr, ytr, eval_set=[(xtr, ytr), (xvl, yvl)])

    #     # Calc and save scores
    #     tr_scores.append(lgb_reg.calc_scores(xdata=xtr, ydata=ytr, to_print=False))
    #     vl_scores.append(lgb_reg.calc_scores(xdata=xvl, ydata=yvl, to_print=False))


    # # Summarize cv scores
    # cv_tr_scores = utils.cv_scores_to_df(tr_scores)
    # cv_vl_scores = utils.cv_scores_to_df(vl_scores)
    # print('\ntr scores\n{}'.format(cv_tr_scores))
    # print('\nvl scores\n{}'.format(cv_vl_scores))
    # cv_tr_scores.to_csv(os.path.join(run_outdir, 'cv_tr_scores.csv'))
    # cv_vl_scores.to_csv(os.path.join(run_outdir, 'cv_vl_scores.csv'))



    # ========================================================================
    #       Automatic CV runs
    # ========================================================================
    logger.info('\n=====================================================')
    logger.info(f'Automatic CV runs ...')
    logger.info('=====================================================')
    # https://scikit-learn.org/stable/modules/cross_validation.html
    from sklearn.model_selection import cross_val_score, cross_validate, learning_curve
    from sklearn.model_selection import ShuffleSplit, GroupShuffleSplit, KFold, GroupKFold
    from sklearn.ensemble import RandomForestRegressor

    # Prepare data
    xdata, _ = utils_tidy.split_features_and_other_cols(data, fea_prfx_dict=fea_prfx_dict)
    ydata = utils_tidy.extract_target(data=data, target_name=target_name)
    utils_tidy.print_feature_shapes(df=xdata, logger=logger)

    # Split tr/vl data
    if cv_method=='simple':
        cv = KFold(n_splits=cv_folds, random_state=SEED)
    elif cv_method=='group':
        cv = GroupKFold(n_splits=cv_folds, random_state=SEED)
    elif cv_method=='stratify':
        pass
    else:
        raise ValueError('This cv_method ({}) is not supported'.format(cv_method))


    # Run CV estimator
    # https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    # logger.info("\nStart cross_val_score ...")
    # lgb_reg = LGBM_REGRESSOR(random_state=SEED, logger=logger)
    # t0 = time.time()
    # scores = cross_val_score(estimator=lgb_reg.model, X=xdata, y=ydata,
    #                          scoring='r2', cv=cv, n_jobs=n_jobs,
    #                          fit_params={'verbose': False, 'early_stopping_rounds': 10})    
    # logger.info('Run-time: {:.3f} mins'.format((time.time()-t0)/60))
    # logger.info(scores)


    logger.info("\nStart cross_validate ...")
    # lgb_reg = LGBM_REGRESSOR(random_state=SEED, logger=logger)
    model, fit_params = init_model(model_name, logger)
    t0 = time.time()
    score_metric = ['r2', 'neg_mean_absolute_error']
    scores = cross_validate(
        estimator=model.model, X=xdata, y=ydata,
        scoring=score_metric, cv=cv, n_jobs=n_jobs,
        fit_params=fit_params)
    logger.info('Run-time: {:.3f} mins'.format((time.time()-t0)/60))
    for k, v in scores.items():
        logger.info(f'{k}: {v}')


    logger.info("\nStart learning_curve ...")
    # lgb_reg = LGBM_REGRESSOR(random_state=SEED, logger=logger)
    model, _ = init_model(model_name, logger)
    t0 = time.time()
    score_metric = 'r2'
    rslt = learning_curve(estimator=model.model, X=xdata, y=ydata,
                          train_sizes=np.linspace(0.1, 1.0, 5),
                          scoring=score_metric, cv=cv, n_jobs=n_jobs,
                          exploit_incremental_learning=False, random_state=SEED)
    logger.info('Run-time: {:.3f} mins'.format((time.time()-t0)/60))
    

    from size_vs_score import plot_learning_curve
    plot_learning_curve(rslt=rslt, score_metric=score_metric,
                        title='Training set size vs score (target: {})'.format(target_name),
                        path=os.path.join(run_outdir, 'scores_vs_train_size.png'))
    


    # ========================================================================
    #       Train final model (entire dataset)
    # ========================================================================
    logger.info('\n=====================================================')
    logger.info(f'Train final model (use entire dataset) ... {train_sources}')
    logger.info('=====================================================')
    xdata, _ = utils_tidy.split_features_and_other_cols(data, fea_prfx_dict=fea_prfx_dict)
    ydata = utils_tidy.extract_target(data=data, target_name=target_name)
    utils_tidy.print_feature_shapes(df=xdata, logger=logger)

    # Train model
    # lgb_reg_final = LGBM_REGRESSOR(random_state=SEED, logger=logger)
    model_final, _ = init_model(model_name, logger)
    # lgb_reg_final.fit(xdata, ydata, eval_set=[(xdata, ydata)])
    if 'lgb_reg' in model_name:
        model_final.fit(xdata, ydata, eval_set=[(xdata, ydata)])
    else:
        model_final.fit(xdata, ydata)

    # Compute scores
    # scores = lgb_reg_final.calc_scores(xdata=xdata, ydata=ydata, to_print=True)
    scores = model_final.calc_scores(xdata=xdata, ydata=ydata, to_print=True)
    # scores = utils.cv_scores_to_df([scores])
    # lgb_reg_final.plot_fi(outdir=run_outdir)

    # Dump preds
    # lgb_reg_final.dump_preds(df_data=data, xdata=xdata, target_name=target_name,
    #                          outpath=os.path.join(run_outdir, 'preds.csv'))
    model_final.dump_preds(df_data=data, xdata=xdata, target_name=target_name,
                           outpath=os.path.join(run_outdir, 'preds.csv'))

    # Save model
    # lgb_reg_final.save_model(outdir=run_outdir)
    model_final.save_model(outdir=run_outdir)



    # ========================================================================
    #       Infer
    # ========================================================================
    logger.info('\n=====================================================')
    logger.info(f'Inference ... {test_sources}')
    logger.info('=====================================================')

    csv_scores = []  # cross-study-validation scores
    for i, src in enumerate(test_sources):
        logger.info(f'\nTest source {i+1}:  ___ {src} ___')

        te_src_data = te_data[te_data['SOURCE'].isin([src])].reset_index(drop=True)
        logger.info(f'src_data.shape {te_src_data.shape}')

        preds_filename_prefix = 'test_' + src
        model_name = 'lgb_reg_final'

        # Prepare test data for predictions
        te_src_data = utils_tidy.impute_values(data=te_src_data, fea_prfx_dict=fea_prfx_dict, logger=logger)
        xte, _ = utils_tidy.split_features_and_other_cols(te_src_data, fea_prfx_dict=fea_prfx_dict)
        yte = utils_tidy.extract_target(data=te_src_data, target_name=target_name)

        # Plot dist of response
        utils.plot_hist(x=te_src_data[target_name], var_name=target_name,
                        path=os.path.join(run_outdir, src+'_'+target_name+'_hist.png'))

        # Print feature shapes
        logger.info(f'\nxte_'+src)
        utils_tidy.print_feature_shapes(df=xte, logger=logger)

        # Compute scores
        # scores = lgb_reg_final.calc_scores(xdata=xte, ydata=yte, to_print=True)
        scores = model_final.calc_scores(xdata=xte, ydata=yte, to_print=True)
        scores = utils.cv_scores_to_df([scores])

        # Dump preds
        # lgb_reg_final.dump_preds(df_data=te_src_data, xdata=xte, target_name=target_name,
        #                          outpath=os.path.join(run_outdir, preds_filename_prefix+'_'+model_name+'_preds.csv'))
        model_final.dump_preds(df_data=te_src_data, xdata=xte, target_name=target_name,
                               outpath=os.path.join(run_outdir, preds_filename_prefix+'_'+model_name+'_preds.csv'))                 

        # Calc and save scores
        # csv_scores.append(lgb_reg_final.calc_scores(xdata=xte, ydata=yte, to_print=False))
        csv_scores.append(model_final.calc_scores(xdata=xte, ydata=yte, to_print=False))

    # Summarize cv scores
    df_csv_scores = pd.DataFrame(csv_scores).T
    df_csv_scores.columns = test_sources
    print('\ncsv_scores\n{}'.format(df_csv_scores))
    df_csv_scores.to_csv(os.path.join(run_outdir, 'df_csv_scores.csv'))


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


def main(args):
# def main(parser):  

    config_fname = os.path.join(file_path, CONFIGFILENAME)
    args = arg_parser.get_args(args=args, config_fname=config_fname)


    # # Parsing priority: command-file > config-file > defualt params
    # defaults = {
    #     'target_name': 'AUC',
    #     'target_trasform': 'f',
    #     'train_sources': 'ccle',
    #     'test_sources': 'ccle',
    #     'tissue_type': ___ ,
    #     'cell_features': 'rna', 
    #     'drug_features': 'dsc',
    #     'other_features': ___ ,
    #     'ml_models': 'lgb_reg',
    #     'cv_method': 'simple',
    #     'cv_folds': 5,
    #     'verbose': 't',
    #     'n_jobs': 4
    # }  ## new
    
    # parser = argparse.ArgumentParser(description="Cell-drug sensitivity parser.")
    # # https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser.add_argument

    # def str_to_bool(s):
    #     """ Convert string to bool (in argparse context).
    #     https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    #     """
    #     if s.lower() not in ['t', 'f']:
    #         raise ValueError("Need 't' or 'f'; got %r" % s)
    #     return {'t': True, 'f': False}[s.lower()]

    # # Select target to predict
    # parser.add_argument("-t", "--target_name",
    #     default="AUC", choices=["AUC", "AUC1", "IC50"],
    #     help="Column name of the target variable.") # target_name = 'AUC1'
    # parser.add_argument("-tt", "--target_trasform",
    #     default='f', choices=['t', 'f'], type=str_to_bool,
    #     help="'t': transform target, 'f': do not transform target.") # target_name = 'AUC1'

    # # Select train and test (inference) sources
    # parser.add_argument("-tr", "--train_sources", nargs="+",
    #     default=["ccle"], choices=["ccle", "gcsi", "gdsc", "ctrp"],
    #     help="Data sources to use for training.")
    # parser.add_argument("-te", "--test_sources", nargs="+",
    #     default=["ccle"], choices=["ccle", "gcsi", "gdsc", "ctrp"],
    #     help="Data sources to use for testing.")

    # # Select tissue types
    # parser.add_argument("-ts", "--tissue_type",
    #     default=argparse.SUPPRESS, choices=[],
    #     help="Tissue types to use.")

    # # Select feature types
    # parser.add_argument("-cf", "--cell_features", nargs="+",
    #     default=['rna'], choices=["rna", "cnv"],
    #     help="Cell line feature types.") # ['rna', cnv', 'rna_latent']
    # parser.add_argument("-df", "--drug_features", nargs="+",
    #     default=['dsc'], choices=["dsc", "fng"],
    #     help="Drug feature types.") # ['dsc', 'fng', 'dsc_latent', 'fng_latent']
    # parser.add_argument("-of", "--other_features", default=[],
    #     choices=[],
    #     help="Other feature types (derived from cell lines and drugs). E.g.: cancer type, etc).") # ['cell_labels', 'drug_labels', 'ctype', 'csite', 'rna_clusters']

    # # Select ML models
    # parser.add_argument("-ml", "--ml_models",
    #     default=["lgb_reg"], choices=["lgb_reg", "rf_reg"],
    #     help="ML models to use for training.")

    # # Select CV scheme
    # parser.add_argument("-cvs", "--cv_method",
    #     default="simple", choices=["simple", "group"],
    #     help="Cross-val split method.")
    # parser.add_argument("-cvf", "--cv_folds",
    #     default=5, type=int,
    #     help="Number cross-val folds.")

    # # Take care of utliers
    # # parser.add_argument("--outlier", default=False)

    # # Define verbosity
    # parser.add_argument("-v", "--verbose",
    #     default="t", choices=["t", "f"], type=str_to_bool,
    #     help="'t': verbose, 'f': not verbose.")

    # # Define n_jobs
    # parser.add_argument("--n_jobs", default=4, type=int)

    # # parser.set_defaults(**defaults)  ## new

    # # Parse the args
    # args = parser.parse_args(args)
    # # args = parser.parse_known_args(args)

    run(args)
    

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
    
    # parser = argparse.ArgumentParser(description="Cell-drug sensitivity parser.")
    # main(parser)

    # args = vars(sys.argv[1:])
    # main(args)