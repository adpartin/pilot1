import warnings
warnings.filterwarnings('ignore')

import os
import time
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.metrics import r2_score, mean_absolute_error, median_absolute_error, explained_variance_score

import matplotlib
# matplotlib.use('TkAgg')
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb


# TODO: create a super class GBM models (xgboost and lightgbm)
# class SuperGBM():  # ModelTunedCVSearch
#     """ This is a super class for training and fine-tuning *Super* GBM models, i.e.,
#     xgboost and lightgbm.
#     This models share similar API so various methods are re-used.
#     """
#     # https://www.kaggle.com/spektrum/randomsearchcv-to-hyper-tune-1st-level


class BaseMLModel():

    def calc_scores(self, xdata, ydata, to_print=False):
        """ Create dict of scores. """
        # TODO: replace `if` with `try`
        if hasattr(self, 'model'):
            preds = self.model.predict(xdata)
            scores = {}
            scores['r2_score'] = r2_score(ydata, preds)
            scores['mean_abs_error'] = mean_absolute_error(ydata, preds)
            scores['median_abs_error'] = median_absolute_error(ydata, preds)
            # scores['explained_variance_score'] = explained_variance_score(ydata, preds)

            self.scores = scores
            if to_print:
                self.print_scores()

            return self.scores


    def print_scores(self):
        """ Print performance scores. """
        # TODO: replace `if` with `try`
        if hasattr(self, 'scores'):
            scores = self.scores

            if self.logger is not None:
                self.logger.info('r2_score: {:.2f}'.format(scores['r2_score']))
                self.logger.info('mean_absolute_error: {:.2f}'.format(scores['mean_abs_error']))
                self.logger.info('median_absolute_error: {:.2f}'.format(scores['median_abs_error']))
                # self.logger.info('explained_variance_score: {:.2f}'.format(scores['explained_variance_score']))
            else:
                print('r2_score: {:.2f}'.format(scores['r2_score']))
                print('mean_absolute_error: {:.2f}'.format(scores['mean_abs_error']))
                print('median_absolute_error: {:.2f}'.format(scores['median_abs_error']))
                # print('explained_variance_score: {:.2f}'.format(scores['explained_variance_score']))


    def dump_preds(self, df_data, xdata, target_name, outpath=None):
        """
        Args:
            df_data : df that contains the cell and drug names, and target value
            xdata : features to make predictions on
            target_name : name of the target as it appears in the df (e.g. 'AUC')
            outpath : full path to store the predictions
        """
        # TODO: replace `if` with `try`
        if hasattr(self, 'model'):
            combined_cols = ['CELL', 'DRUG', 'csite', 'ctype', 'simplified_csite', 'simplified_ctype', target_name]
            ccle_org_cols = ['CELL', 'DRUG', 'tissuetype', target_name]

            ##df1 = df_data[['CELL', 'DRUG', 'csite', 'ctype', 'simplified_csite', 'simplified_ctype', target_name]].copy()
            if set(combined_cols).issubset(set(df_data.columns.tolist())):
                df1 = df_data[combined_cols].copy()
            elif set(ccle_org_cols).issubset(set(df_data.columns.tolist())):
                df1 = df_data[ccle_org_cols].copy()
            else:
                df1 = df_data['CELL', 'DRUG'].copy()

            preds = self.model.predict(xdata)
            abs_error = abs(df_data[target_name] - preds)
            squared_error = (df_data[target_name] - preds)**2
            df2 = pd.DataFrame({target_name+'_pred': self.model.predict(xdata),
                                target_name+'_error': abs_error,
                                target_name+'_sq_error': squared_error})

            df_preds = pd.concat([df1, df2], axis=1).reset_index(drop=True)

            if outpath is not None:
                df_preds.to_csv(outpath)
            else:
                df_preds.to_csv('preds.csv')



class RF_REGRESSOR(BaseMLModel):
    model_name = 'rf_reg'

    def __init__(self, n_estimators=100, criterion='mse',
                 max_depth=None, min_samples_split=2,
                 bootstrap=True, oob_score=True, verbose=0, 
                 n_jobs=1, random_state=None,
                 logger=None):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.logger = logger                 

        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            max_features='sqrt', bootstrap=bootstrap, oob_score=oob_score,
            verbose=verbose, random_state=self.random_state, n_jobs=self.n_jobs)


    def fit(self, X, y):
        self.X = X
        self.y = y
        
        t0 = time.time()
        self.model.fit(self.X, self.y)
        self.train_runtime = time.time() - t0

        if self.logger is not None:
            self.logger.info('Train time: {:.2f} mins'.format(self.train_runtime/60))


    def plot_fi(self):
        # TODO
        pass


    def save_model(self, outdir='./'):
        joblib.dump(self.model, filename=os.path.join(outdir, RF_REGRESSOR.model_name+'_model.pkl'))
        # model_ = joblib.load(filename=os.path.join(run_outdir, 'lgb_reg_model.pkl'))


class LGBM_REGRESSOR(BaseMLModel):
    # Define class attributes (www.toptal.com/python/python-class-attributes-an-overly-thorough-guide)
    ml_objective = 'regression'
    model_name = 'lgb_reg'

    def __init__(self, eval_metric=['l1', 'l2'], n_jobs=1, random_state=None,
                 logger=None):
        # https://lightgbm.readthedocs.io/en/latest/Python-API.html
        # try:
        #     import lightgbm as lgb
        # except ImportError:  # install??
        #     logger.error('Module not found (lightgbm)')

        # TODO: use config file to set default parameters (like in candle)
        self.eval_metric = eval_metric
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.logger = logger

        # if logger is not None:
        #     self.logger.info('\nTrain LGBMRegressor ...')
        
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

        # ----- lightgbm "sklearn API" - start
        self.model = lgb.LGBMModel(objective=LGBM_REGRESSOR.ml_objective,
                                   n_jobs=self.n_jobs,
                                   random_state=self.random_state)


    def fit(self, X, y, eval_set=None):
        self.eval_set = eval_set
        self.X = X
        self.y = y
        
        t0 = time.time()
        self.model.fit(self.X, self.y,
                       eval_metric=self.eval_metric,
                       eval_set=self.eval_set,
                       early_stopping_rounds=10, verbose=False, callbacks=None)
        self.train_runtime = time.time() - t0

        if self.logger is not None:
            self.logger.info('Train time: {:.2f} mins'.format(self.train_runtime/60))


    def plot_fi(self, max_num_features=20, title='LGBMRegressor', outdir=None):
        lgb.plot_importance(booster=self.model, max_num_features=max_num_features, grid=True, title=title)
        plt.tight_layout()

        filename = LGBM_REGRESSOR.model_name+'_fi.png'
        if outdir is None:
            plt.savefig(filename, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(outdir, filename), bbox_inches='tight')


    def save_model(self, outdir='./'):
        # lgb_reg.save_model(os.path.join(run_outdir, 'lgb_'+ml_type+'_model.txt'))
        joblib.dump(self.model, filename=os.path.join(outdir, LGBM_REGRESSOR.model_name+'_model.pkl'))
        # lgb_reg_ = joblib.load(filename=os.path.join(run_outdir, 'lgb_reg_model.pkl'))


    # def calc_scores(self, xdata, ydata, to_print=False):
    #     """ Create dict of scores. """
    #     # TODO: replace `if` with `try`
    #     if hasattr(self, 'model'):
    #         scores = {}
    #         preds = self.model.predict(xdata)
    #         scores['r2_score'] = r2_score(ydata, preds)
    #         scores['mean_abs_error'] = mean_absolute_error(ydata, preds)
    #         scores['median_abs_error'] = median_absolute_error(ydata, preds)
    #         # scores['explained_variance_score'] = explained_variance_score(ydata, preds)

    #         self.scores = scores
    #         if to_print:
    #             self.print_scores()

    #         return self.scores


    # def print_scores(self):
    #     if hasattr(self, 'scores'):
    #         scores = self.scores

    #         if self.logger is not None:
    #             self.logger.info('r2_score: {:.2f}'.format(scores['r2_score']))
    #             self.logger.info('mean_absolute_error: {:.2f}'.format(scores['mean_abs_error']))
    #             self.logger.info('median_absolute_error: {:.2f}'.format(scores['median_abs_error']))
    #             # self.logger.info('explained_variance_score: {:.2f}'.format(scores['explained_variance_score']))
    #         else:
    #             print('r2_score: {:.2f}'.format(scores['r2_score']))
    #             print('mean_absolute_error: {:.2f}'.format(scores['mean_abs_error']))
    #             print('median_absolute_error: {:.2f}'.format(scores['median_abs_error']))
    #             # print('explained_variance_score: {:.2f}'.format(scores['explained_variance_score']))


    # def dump_preds(self, df_data, xdata, target_name, outpath=None):
    #     """
    #     Args:
    #         df_data : df that contains the cell and drug names, and target value
    #         xdata : features to make predictions on
    #         target_name : name of the target as it appears in the df (e.g. 'AUC')
    #         outpath : full path to store the predictions
    #     """
    #     if hasattr(self, 'model'):
    #         combined_cols = ['CELL', 'DRUG', 'csite', 'ctype', 'simplified_csite', 'simplified_ctype', target_name]
    #         ccle_org_cols = ['CELL', 'DRUG', 'tissuetype', target_name]

    #         ##df1 = df_data[['CELL', 'DRUG', 'csite', 'ctype', 'simplified_csite', 'simplified_ctype', target_name]].copy()
    #         if set(combined_cols).issubset(set(df_data.columns.tolist())):
    #             df1 = df_data[combined_cols].copy()
    #         elif set(ccle_org_cols).issubset(set(df_data.columns.tolist())):
    #             df1 = df_data[ccle_org_cols].copy()
    #         else:
    #             df1 = df_data['CELL', 'DRUG'].copy()

    #         preds = self.model.predict(xdata)
    #         abs_error = abs(df_data[target_name] - preds)
    #         squared_error = (df_data[target_name] - preds)**2
    #         df2 = pd.DataFrame({target_name+'_pred': self.model.predict(xdata),
    #                             target_name+'_error': abs_error,
    #                             target_name+'_sq_error': squared_error})

    #         df_preds = pd.concat([df1, df2], axis=1).reset_index(drop=True)

    #         if outpath is not None:
    #             df_preds.to_csv(outpath)
    #         else:
    #             df_preds.to_csv('preds.csv')



    # # Plot learning curves
    # # TODO: note, plot_metric didn't accept 'mae' although it's alias for 'l1' 
    # # TODO: plot_metric requires dict from train(), but train returns 'lightgbm.basic.Booster'??
    # for m in eval_metric:
    #     ax = lgb.plot_metric(booster=lgb_reg, metric=m, grid=True)
    #     plt.savefig(os.path.join(run_outdir, model_name+'_learning_curve_'+m+'.png'))



# # ========================================================================
# #       Train models - the code below was in train_from_combined.py
# # ========================================================================
# from sklearn.metrics import r2_score, mean_absolute_error, median_absolute_error
# from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, ParameterGrid
# from sklearn.externals import joblib

# train_runtime = OrderedDict() # {}
# preds_filename_prefix = 'dev'

# # ---------------------
# # RandomForestRegressor
# # ---------------------
# if 'rf_reg' in ml_models:
#     model_name = 'rf_reg'
#     try:
#         from sklearn.ensemble import RandomForestRegressor
#     except ImportError:
#         # install??
#         logger.error(f'Module not found (RandomForestRegressor)')

#     logger.info('\nTrain RandomForestRegressor ...')
#     # ----- rf hyper-param start
#     rf_reg = RandomForestRegressor(max_features='sqrt', bootstrap=True, oob_score=True,
#                                 verbose=0, random_state=SEED, n_jobs=n_jobs)

#     random_search_params = {'n_estimators': [100, 500, 1000], # [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
#                             'max_depth': [None, 5, 10], # [None] + [int(x) for x in np.linspace(10, 110, num = 11)]
#                             'min_samples_split': [2, 5, 9]}
#     logger.info('hyper-params:\n{}'.format(random_search_params))

#     rf_reg_randsearch = RandomizedSearchCV(
#         estimator=rf_reg,
#         param_distributions=random_search_params,
#         n_iter=20,  # num of parameter settings that are sampled and used for training (num of models trained)
#         scoring=None, # string or callable used to evaluate the predictions on the test set
#         n_jobs=n_jobs,
#         cv=5,
#         refit=True,  # Refit an estimator using the best found parameters on the whole dataset
#         verbose=0)

#     # Run search
#     t0 = time.time()
#     rf_reg_randsearch.fit(xtr, ytr)
#     train_runtime[model_name+'_randsearch'] = time.time() - t0
#     logger.info('Runtime: {:.2f} mins'.format(train_runtime[model_name+'_randsearch']/60))

#     # Save best model
#     rf_reg = rf_reg_randsearch.best_estimator_
#     joblib.dump(rf_reg, filename=os.path.join(run_outdir, model_name+'_hypsearch_best_model.pkl'))

#     # Print preds
#     utils.print_scores(model=rf_reg, xdata=xvl, ydata=yvl, logger=logger)

#     # Save resutls
#     rf_reg_hypsearch = pd.DataFrame(rf_reg_randsearch.cv_results_)
#     rf_reg_hypsearch.to_csv(os.path.join(run_outdir, model_name+'_hypsearch_summary.csv'))  # save hyperparam search results

#     logger.info(f'{model_name} best score (random search): {rf_reg_randsearch.best_score_:.3f}')
#     logger.info('{} best params (random search): \n{}'.format(model_name, rf_reg_randsearch.best_params_))

#     # Dump preds
#     utils.dump_preds(model=rf_reg, df_data=vl_data, xdata=xvl, target_name=target_name,
#                     path=os.path.join(run_outdir, preds_filename_prefix+'_'+model_name+'_preds.csv'))
#     # ----- rf hyper-param end


# # ------------
# # XGBRegressor
# # ------------
# if 'xgb_reg' in ml_models:
#     try:
#         import xgboost as xgb
#     except ImportError:  # install??
#         logger.error('Module not found (xgboost)')

#     # https://xgboost.readthedocs.io/en/latest/python/python_api.html
#     # xgboost does not support categorical features!
#     # Rules of thumb
#     # 1. learning_rate should be 0.1 or lower (smaller values will require more trees).
#     # 2. tree_depth should be between 2 and 8 (where not much benefit is seen with deeper trees).
#     # 3. subsample should be between 30% and 80% of the training dataset, and compared to a value of 100% for no sampling.
#     logger.info('\nTrain XGBRegressor ...')
#     # xgb_tr = xgb.DMatrix(data=xtr, label=ytr, nthread=n_jobs)
#     # xgb_vl = xgb.DMatrix(data=xvl, label=yvl, nthread=n_jobs)
#     # ----- xgboost hyper-param start
#     xgb_reg = xgb.XGBRegressor(objective='reg:linear', # default: 'reg:linear', TODO: docs recommend funcs for different distributions (??)
#                             booster='gbtree', # default: gbtree (others: gblinear, dart)
#                             # max_depth=3, # default: 3
#                             # learning_rate=0.1, # default: 0.1
#                             # n_estimators=100, # default: 100
#                             n_jobs=n_jobs, # default: 1
#                             reg_alpha=0, # default=0, L1 regularization
#                             reg_lambda=1, # default=1, L2 regularization
#                             random_state=SEED)

#     random_search_params = {'n_estimators': [30, 50, 70],
#                             'learning_rate': [0.005, 0.01, 0.5],
#                             'subsample': [0.5, 0.7, 0.8],
#                             'max_depth': [2, 3, 5]}
#     logger.info('hyper-params:\n{}'.format(random_search_params))

#     xgb_reg_randsearch = RandomizedSearchCV(
#         estimator=xgb_reg,
#         param_distributions=random_search_params,
#         n_iter=20,  # num of parameter settings that are sampled and used for training (num of models trained)
#         scoring=None, # string or callable used to evaluate the predictions on the test set
#         n_jobs=n_jobs,
#         cv=5,
#         refit=True,  # Refit an estimator using the best found parameters on the whole dataset
#         verbose=False)   

#     # Start search
#     t0 = time.time()
#     xgb_reg_randsearch.fit(xtr, ytr)
#     train_runtime['xgb_reg_randsearch'] = time.time() - t0
#     logger.info('Runtime: {:.2f} mins'.format(train_runtime['xgb_reg_randsearch']/60))

#     # Save best model
#     xgb_reg = xgb_reg_randsearch.best_estimator_
#     joblib.dump(xgb_reg, filename=os.path.join(run_outdir, 'xgb_reg_hypsearch_best_model.pkl'))

#     # Print preds
#     utils.print_scores(model=xgb_reg, xdata=xvl, ydata=yvl, logger=logger)

#     # Save resutls
#     xgb_reg_hypsearch = pd.DataFrame(xgb_reg_randsearch.cv_results_)
#     xgb_reg_hypsearch.to_csv(os.path.join(run_outdir, 'xgb_reg_hypsearch_summary.csv'))  # save hyperparam search results

#     logger.info(f'rf_reg best score (random search): {xgb_reg_randsearch.best_score_:.3f}')
#     logger.info('rf_reg best params (random search): \n{}'.format(xgb_reg_randsearch.best_params_))

#     # Dump preds
#     utils.dump_preds(model=xgb_reg, df_data=vl_data, xdata=xvl, target_name=target_name,
#                     path=os.path.join(run_outdir, 'xgb_vl_preds.csv'))
#     # ----- xgboost hyper-param end

#     # ----- xgboost "Sklearn API" start
#     xgb_reg = xgb.XGBRegressor(objective='reg:linear', # default: 'reg:linear', TODO: docs recommend funcs for different distributions (??)
#                             booster='gbtree', # default: gbtree (others: gblinear, dart)
#                             max_depth=3, # default: 3
#                             learning_rate=0.1, # default: 0.1
#                             n_estimators=100, # default: 100
#                             n_jobs=n_jobs, # default: 1
#                             reg_alpha=0, # default=0, L1 regularization
#                             reg_lambda=1, # default=1, L2 regularization
#                             random_state=SEED
#     )
#     eval_metric = ['mae', 'rmse']
#     t0 = time.time()
#     xgb_reg.fit(xtr, ytr, eval_metric=eval_metric, eval_set=[(xtr, ytr), (xvl, yvl)],
#                 early_stopping_rounds=10, verbose=False, callbacks=None)
#     train_runtime['xgb_reg'] = time.time() - t0
#     logger.info('Runtime: {:.2f} mins'.format(train_runtime['xgb_reg']/60))

#     # Save model
#     # xgb_reg.save_model(os.path.join(run_outdir, 'xgb_reg.model'))
#     joblib.dump(xgb_reg, filename=os.path.join(run_outdir, 'xgb_reg_model.pkl'))
#     # xgb_reg_ = joblib.load(filename=os.path.join(run_outdir, 'xgb_reg_model.pkl'))

#     # Print preds
#     utils.print_scores(model=xgb_reg, xdata=xvl, ydata=yvl, logger=logger)

#     # Dump preds
#     utils.dump_preds(model=xgb_reg, df_data=vl_data, xdata=xvl, target_name=target_name,
#                     path=os.path.join(run_outdir, 'xgb_vl_preds.csv'))
#     # ----- xgboost "Sklearn API" end
        
#     # Plot feature importance
#     xgb.plot_importance(booster=xgb_reg, max_num_features=20, grid=True, title='XGBRegressor')
#     plt.tight_layout()
#     plt.savefig(os.path.join(run_outdir, 'xgb_reg_importances.png'))

#     # Plot learning curves
#     xgb_results = xgb_reg.evals_result()
#     epoch_vec = np.arange(1, len(xgb_results['validation_0'][eval_metric[0]])+1)
#     for m in eval_metric:
#         fig, ax = plt.subplots()
#         for i, s in enumerate(xgb_results):
#             label = 'Train' if i==0 else 'Val'
#             ax.plot(epoch_vec, xgb_results[s][m], label=label)
#         plt.xlabel('Epochs')
#         plt.title(m)
#         plt.legend()
#         plt.grid(True)
#         plt.tight_layout()
#         plt.savefig(os.path.join(run_outdir, 'xgb_reg_leraning_curve_'+m+'.png'))


# # -------------
# # LGBMRegressor
# # -------------
# if 'lgb_reg' in ml_models:
#     model_name = 'lgb_reg'
#     try:
#         import lightgbm as lgb
#     except ImportError:  # install??
#         logger.error('Module not found (lightgbm)')

#     # https://lightgbm.readthedocs.io/en/latest/Python-API.html
#     # TODO: use config file to set default parameters
#     logger.info('\nTrain LGBMRegressor ...')
#     ml_objective = 'regression'
#     eval_metric = ['l1', # aliases: regression_l1, mean_absolute_error, mae
#                    'l2', # aliases: regression, regression_l2, mean_squared_error, mse, and more
#                    ]

#     # ----- lightgbm "Training API" - start
#     # lgb_tr = lgb.Dataset(data=xtr, label=ytr, categorical_feature='auto')
#     # lgb_vl = lgb.Dataset(data=xvl, label=yvl, categorical_feature='auto')
#     # # https://lightgbm.readthedocs.io/en/latest/Parameters.html
#     # params = {'task': 'train', # default='train'
#     #         'objective': ml_objective, # default='regression' which alias for 'rmse' and 'mse' (but these are different??)
#     #         'boosting': 'gbdt', # default='gbdt'
#     #         'num_iterations': 100, # default=100 (num of boosting iterations)
#     #         'learning_rate': 0.1, # default=0.1
#     #         'num_leaves': 31, # default=31 (num of leaves in 1 tree)
#     #         'seed': SEED,
#     #         'num_threads': n_jobs, # default=0 (set to the num of real CPU cores)
#     #         'device_type': 'cpu', # default='cpu'
#     #         'metric': eval_metric # metric(s) to be evaluated on the evaluation set(s)
#     #         }
#     # t0 = time.time()
#     # lgb_reg = lgb.train(params=params, train_set=lgb_tr, valid_sets=lgb_vl, verbose_eval=False)
#     # # lgb_cv = lgb.train(params=params, train_set=lgb_tr, nfolds=5)
#     # train_runtime['lgb_reg'] = time.time() - t0
#     # logger.info('Runtime: {:.2f} mins'.format(train_runtime['lgb_reg']/60))
#     # ----- lightgbm "Training API" - end 

#     # ----- lightgbm "sklearn API" appraoch 1 - start
#     lgb_reg = lgb.LGBMModel(objective=ml_objective,
#                             n_jobs=n_jobs,
#                             random_state=SEED)
#     # lgb_reg = lgb.LGBMRegressor()
#     t0 = time.time()
#     lgb_reg.fit(xtr, ytr, eval_metric=eval_metric, eval_set=[(xtr, ytr), (xvl, yvl)],
#                 early_stopping_rounds=10, verbose=False, callbacks=None)
#     train_runtime[model_name] = time.time() - t0
#     logger.info('Runtime: {:.2f} mins'.format(train_runtime[model_name]/60))
#     # ----- lightgbm "sklearn API" appraoch 1 - end

#     # Save model
#     # lgb_reg.save_model(os.path.join(run_outdir, 'lgb_'+ml_type+'_model.txt'))
#     joblib.dump(lgb_reg, filename=os.path.join(run_outdir, model_name+'_model.pkl'))
#     # lgb_reg_ = joblib.load(filename=os.path.join(run_outdir, 'lgb_reg_model.pkl'))

#     # Print preds
#     # utils.print_scores(model=lgb_reg, xdata=xtr, ydata=ytr)
#     # utils.print_scores(model=lgb_reg, xdata=xvl, ydata=yvl)
#     utils.print_scores(model=lgb_reg, xdata=xvl, ydata=yvl, logger=logger)

#     # Dump preds
#     utils.dump_preds(model=lgb_reg, df_data=vl_data, xdata=xvl, target_name=target_name,
#                      path=os.path.join(run_outdir, preds_filename_prefix+'_'+model_name+'_preds.csv'))

#     # Plot feature importance
#     lgb.plot_importance(booster=lgb_reg, max_num_features=20, grid=True, title='LGBMRegressor')
#     plt.tight_layout()
#     plt.savefig(os.path.join(run_outdir, model_name+'_importances.png'))

#     # Plot learning curves
#     # TODO: note, plot_metric didn't accept 'mae' although it's alias for 'l1' 
#     # TODO: plot_metric requires dict from train(), but train returns 'lightgbm.basic.Booster'??
#     for m in eval_metric:
#         ax = lgb.plot_metric(booster=lgb_reg, metric=m, grid=True)
#         plt.savefig(os.path.join(run_outdir, model_name+'_learning_curve_'+m+'.png'))


# # -------------
# # TPOTRegressor
# # -------------
# # Total evaluation pipelines:  population_size + generations × offspring_size 
# if 'tpot_reg' in ml_models:
#     try:
#         import tpot
#     except ImportError:
#         logger.error('Module not found (tpot)')
    
#     tpot_checkpoint_folder = os.path.join(run_outdir, 'tpot_reg_checkpoints')
#     os.makedirs(tpot_checkpoint_folder)

#     logger.info('\nTrain TPOTRegressor ...')
#     tpot_reg = tpot.TPOTRegressor(generations=100,  # dflt: 100
#                                   population_size=100, # dflt: 100
#                                   offspring_size=100, # dflt: 100
#                                   scoring='neg_mean_squared_error', # dflt: 'neg_mean_squared_error'
#                                   cv=5,
#                                   n_jobs=n_jobs,
#                                   random_state=SEED,
#                                   warm_start=False,
#                                   periodic_checkpoint_folder=tpot_checkpoint_folder,
#                                   verbosity=2,
#                                   disable_update_check=True)
#     t0 = time.time()
#     tpot_reg.fit(xtr, ytr)
#     train_runtime['tpot_reg'] = time.time() - t0
#     logger.info('Runtime: {:.2f} mins'.format(ml_runtime['tpot_reg']/60))
    
#     # Export model as .py script
#     tpot_reg.export(os.path.join(run_outdir, 'tpot_reg_pipeline.py'))

#     # Print scores
#     utils.print_scores(model=tpot_reg, xdata=xvl, ydata=yvl, logger=logger)

#     # Dump preds
#     t0 = time.time()
#     utils.dump_preds(model=tpot_reg, df_data=vl_data, xdata=xvl, target_name=target_name,
#                     path=os.path.join(run_outdir, 'tpot_reg_vl_preds.csv'))
#     logger.info('Predictions runtime: {:.2f} mins'.format(time.time()/60))