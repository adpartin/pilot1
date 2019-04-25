"""
This script contains various ML models.
"""
import warnings
warnings.filterwarnings('ignore')

import os
from pathlib import Path
from time import time
from collections import OrderedDict

import sklearn
import numpy as np
import pandas as pd

import matplotlib
# matplotlib.use('TkAgg')
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib

try:
    import lightgbm as lgb
except ImportError:
    print('Module not found (lightgbm).')


# class GBM():
#     """ This is a super class for training and fine-tuning GBM models, i.e.,
#     xgboost and lightgbm. These models share similar API where various methods are the same.
#     """
#     # https://www.kaggle.com/spektrum/randomsearchcv-to-hyper-tune-1st-level


def r2_krs(y_true, y_pred):
    from keras import backend as K
    SS_res =  K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res/(SS_tot + K.epsilon()))


def get_model(model_name, init_params=None):
    """ Return a model.
    Args:
        init_params : init parameters to the model
        model_name : model name
    """
    if model_name == 'lgb_reg':
        model = LGBM_REGRESSOR(**init_params)
        estimator = model
    elif model_name == 'rf_reg':
        model = RF_REGRESSOR(**init_params)
        estimator = model
    elif model_name == 'nn_reg':
        model = KERAS_REGRESSOR(**init_params)
        estimator = model
    else:
        pass
    return estimator


def save_krs_history(history, outdir='.'):
    fname = 'krs_history.csv'
    h = pd.DataFrame(history.history)
    h['epoch'] = np.asarray(history.epoch) + 1
    h.to_csv(outdir/fname, index=False)
    return h


def get_keras_prfrm_metrics(history):
    """ Extract names of all the recorded performance metrics from keras `history` variable for
    train and val sets. The performance metrics can be indentified as those starting with 'val'.
    """
    all_metrics = list(history.history.keys())  # all metrics including everything returned from callbacks
    pr_metrics = []  # performance metrics recorded for train and val such as 'loss', etc. (excluding callbacks)
    for m in all_metrics:
        if 'val' in m:
            pr_metrics.append('_'.join(m.split('_')[1:]))

    return pr_metrics


def plot_prfrm_metrics(history, title=None, skp_ep=0, outdir='.', add_lr=False):
    """ Plots keras training curves.
    Args:
        skp_ep: number of epochs to skip when plotting metrics 
        add_lr: add curve of learning rate progression over epochs
    """
    pr_metrics = get_keras_prfrm_metrics(history)
    epochs = np.asarray(history.epoch) + 1
    if len(epochs) <= skp_ep:
        skp_ep = 0
    eps = epochs[skp_ep:]
    hh = history.history
    
    for p, m in enumerate(pr_metrics):
        metric_name = m
        metric_name_val = 'val_' + m

        y_tr = hh[metric_name][skp_ep:]
        y_vl = hh[metric_name_val][skp_ep:]
        
        ymin = min(set(y_tr).union(y_vl))
        ymax = max(set(y_tr).union(y_vl))
        lim = (ymax - ymin) * 0.1
        ymin, ymax = ymin - lim, ymax + lim

        # Start figure
        fig, ax1 = plt.subplots()
        
        # Plot metrics
        ax1.plot(eps, y_tr, color='b', marker='.', linestyle='-', linewidth=1, alpha=0.6, label=metric_name)
        ax1.plot(eps, y_vl, color='r', marker='.', linestyle='-', linewidth=1, alpha=0.6, label=metric_name_val)
        ax1.set_xlabel('epoch')
        ax1.set_ylabel(metric_name)
        ax1.set_xlim([min(eps)-1, max(eps)+1])
        ax1.set_ylim([ymin, ymax])
        ax1.tick_params('y', colors='k')
        
        # ax1.tick_params(axis='both', which='major', labelsize=12)
        # ax1.tick_params(axis='both', which='minor', labelsize=12)        
        
        # Add learning rate
        if (add_lr is True) and ('lr' in hh):            
            ax2 = ax1.twinx()
            ax2.plot(eps, hh['lr'][skp_ep:], color='g', marker='.', linestyle=':', linewidth=1,
                     alpha=0.6, markersize=5, label='learning rate')
            ax2.set_ylabel('learning rate', color='g', fontsize=12)
            
            yscale = 'log'  # 'linear'
            ax2.set_yscale(yscale)
            ax2.tick_params('y', colors='g')
        
        ax1.grid(True)
        #plt.legend([metric_name, metric_name_val], loc='best')
        #https://medium.com/@samchaaa/how-to-plot-two-different-scales-on-one-plot-in-matplotlib-with-legend-46554ba5915a
        legend = ax1.legend(loc='best', prop={'size': 10})
        frame = legend.get_frame()
        frame.set_facecolor('0.95')
        if title is not None:
            plt.title(title)
        
        # fig.tight_layout()
        figpath = outdir / (metric_name+'_curve.png')
        plt.savefig(figpath, bbox_inches='tight')
        plt.close()
        

class BaseMLModel():
    """ A parent class with some general methods for children ML classes.
    The children classes are specific ML models such random forest regressor, lightgbm regressor, etc.
    """
    def __adj_r2_score(self, ydata, preds):
        """ Calc adjusted r^2.
        https://en.wikipedia.org/wiki/Coefficient_of_determination#Adjusted_R2
        https://dziganto.github.io/data%20science/linear%20regression/machine%20learning/python/Linear-Regression-101-Metrics/
        https://stats.stackexchange.com/questions/334004/can-r2-be-greater-than-1
        """
        r2_score = sklearn.metrics.r2_score(ydata, preds)
        adj_r2 = 1 - (1 - r2_score) * (self.x_size[0] - 1)/(self.x_size[0] - self.x_size[1] - 1)
        return adj_r2


    def calc_scores(self, xdata, ydata, metrics=None, to_print=False):
        """ Create dict of scores for regression. """
        # metrics = {'r2_score': sklearn.metrics.r2_score,
        #            'mean_absolute_error': sklearn.metrics.mean_absolute_error,
        #            'median_absolute_error': sklearn.metrics.median_absolute_error,
        #            'explained_variance_score': sklearn.metrics.explained_variance_score}
        # TODO: replace `if` with `try`
        if hasattr(self, 'model'):
            preds = self.model.predict(xdata)
            scores = OrderedDict()

            # for metric_name, metric in metrics.items():
            #     if isinstance(metric, str):
            #         scorer = sklearn.metrics.get_scorer(metric_name) # get a scorer from string
            #         scores[metric_name] = scorer(ydata, preds)
            #     else:
            #         scores[metric_name] = scorer(ydata, preds)

            scores['r2'] = sklearn.metrics.r2_score(ydata, preds)
            #scores['adj_r2_score'] = self.__adj_r2_score(ydata, preds)
            scores['mean_absolute_error'] = sklearn.metrics.mean_absolute_error(ydata, preds)
            scores['median_absolute_error'] = sklearn.metrics.median_absolute_error(ydata, preds)
            scores['mean_squared_error'] = sklearn.metrics.mean_squared_error(ydata, preds)

            # TODO:
            y_true = np.where(ydata < 0.5, 1, 0)
            y_score = np.where(preds < 0.5, 1, 0)
            scores['reg_auroc_score'] = sklearn.metrics.roc_auc_score(y_true, y_score)

            self.scores = scores
            if to_print:
                self.print_scores()

            return self.scores


    def print_scores(self):
        """ Print performance scores. """
        # TODO: replace `if` with `try`
        if hasattr(self, 'scores') and self.logger is not None:
            for score_name, score_value in self.scores.items():
                self.logger.info(f'{score_name}: {score_value:.2f}')


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
            #squared_error = (df_data[target_name] - preds)**2
            df2 = pd.DataFrame({target_name+'_pred': self.model.predict(xdata),
                                target_name+'_abs_err': abs_error,
                                #target_name+'_sq_err': squared_error
                                })
            df2 = df2.round(5)

            df_preds = pd.concat([df1, df2], axis=1).reset_index(drop=True)

            if outpath is not None:
                df_preds.to_csv(outpath, index=False)
            else:
                df_preds.to_csv('preds.csv', index=False)



class KERAS_REGRESSOR(BaseMLModel):
    """ Neural network regressor. """
    model_name = 'nn_reg'

    def __init__(self, input_dim, attn=False, dr_rate=0.2, optimizer=None,
                 logger=None):
        # Load keras modules only if keras model is invoked
        # TODO: there should be a better way to make this code compatible on machine with and w/o GPU!
        import tensorflow as tf
        import keras
        from keras import backend as K
        from keras.layers import Input, Dense, Dropout, Activation, BatchNormalization
        from keras import optimizers
        from keras.optimizers import SGD, Adam, RMSprop, Adadelta
        from keras.models import Sequential, Model, model_from_json, model_from_yaml
        from keras.utils import np_utils, multi_gpu_model
        from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard

        inputs = Input(shape=(input_dim,))
        if attn:
            a = Dense(1000, activation='relu')(inputs)
            b = Dense(1000, activation='softmax')(inputs)
            x = keras.layers.multiply( [a, b] )
        else:
            x = Dense(1000, activation='relu')(inputs)
            
        x = Dense(1000, activation='relu')(x)
        x = Dropout(dr_rate)(x)
        
        x = Dense(500, activation='relu')(x)
        x = Dropout(dr_rate)(x)
        
        x = Dense(250, activation='relu')(x)
        x = Dropout(dr_rate)(x)
        
        x = Dense(125, activation='relu')(x)
        x = Dropout(dr_rate)(x)
        
        x = Dense(60, activation='relu')(x)
        x = Dropout(dr_rate)(x)
        
        x = Dense(30, activation='relu')(x)
        x = Dropout(dr_rate)(x)
        
        outputs = Dense(1, activation='relu')(x)
        model = Model(inputs=inputs, outputs=outputs)
        model.summary()
        
        if optimizer is None:
            optimizer = SGD(lr=0.0001, momentum=0.9)
            
        model.compile(loss='mean_squared_error',
                      optimizer=optimizer,
                      metrics=['mae', r2_krs])
        self.model = model


    def dump_model(self, outpath='model.h5'):
        """ Dump trained model. """        
        self.model.save(outpath)
        
#         # Serialize model to JSON
#         model_json = self.model.to_json()
#         modelpath = os.path.join(outdir, 'model.' + KERAS_REGRESSOR.model_name + '.json')
#         with open(modelpath, 'w') as mfile:
#             mfile.write(model_json)

#         # serialize weights to HDF5
#         weightpath = os.path.join(outdir, 'weights.' + KERAS_REGRESSOR.model_name + '.h5')
#         self.model.save_weights(weightpath)



class TORCH_REGRESSOR(BaseMLModel):
    """ Neural network regressor. """
    model_name = 'nn_reg'
    
    def __init__(self):
        pass

    def dump_model(self):
        pass



class LGBM_REGRESSOR(BaseMLModel):
    """ LightGBM regressor. """
    ml_objective = 'regression'
    model_name = 'lgb_reg'

    def __init__(self, n_estimators=100, eval_metric=['l2', 'l1'], n_jobs=1, random_state=None,
                 logger=None):
        # TODO: use config file to set default parameters (like in candle)
        
        self.model = lgb.LGBMModel(
            objective=LGBM_REGRESSOR.ml_objective,
            n_estimators=n_estimators,
            n_jobs=n_jobs,
            random_state=random_state)


    # def fit(self, X, y, eval_set=None, **fit_params):
    #     #self.eval_set = eval_set
    #     #self.X = X
    #     #self.y = y
    #     #self.x_size = X.shape  # this is used to calc adjusteed r^2
        
    #     t0 = time.time()
    #     self.model.fit(X, y,
    #                    eval_metric=self.eval_metric,
    #                    eval_set=eval_set,
    #                    **fit_params)
    #     self.train_runtime = time.time() - t0

    #     if self.logger is not None:
    #         self.logger.info('Train time: {:.2f} mins'.format(self.train_runtime/60))


    def dump_model(self, outdir='.'):
        # lgb_reg.save_model(os.path.join(run_outdir, 'lgb_'+ml_type+'_model.txt'))
        joblib.dump(self.model, filename=Path(outdir)/('model.' + LGBM_REGRESSOR.model_name + '.pkl'))
        # lgb_reg_ = joblib.load(filename=os.path.join(run_outdir, 'lgb_reg_model.pkl'))

        
    def plot_fi(self, max_num_features=20, title='LGBMRegressor', outdir=None):
        lgb.plot_importance(booster=self.model, max_num_features=max_num_features, grid=True, title=title)
        plt.tight_layout()

        filename = LGBM_REGRESSOR.model_name + '_fi.png'
        if outdir is None:
            plt.savefig(filename, bbox_inches='tight')
        else:
            plt.savefig(Path(outdir)/filename, bbox_inches='tight')


    # # Plot training curves
    # # TODO: note, plot_metric didn't accept 'mae' although it's alias for 'l1' 
    # # TODO: plot_metric requires dict from train(), but train returns 'lightgbm.basic.Booster'??
    # for m in eval_metric:
    #     ax = lgb.plot_metric(booster=lgb_reg, metric=m, grid=True)
    #     plt.savefig(os.path.join(run_outdir, model_name+'_learning_curve_'+m+'.png'))
    


class RF_REGRESSOR(BaseMLModel):
    """ Random forest regressor. """
    # Define class attributes (www.toptal.com/python/python-class-attributes-an-overly-thorough-guide)
    model_name = 'rf_reg'

    def __init__(self, n_estimators=100, criterion='mse',
                 max_depth=None, min_samples_split=2,
                 max_features='sqrt',
                 bootstrap=True, oob_score=True, verbose=0, 
                 n_jobs=1, random_state=None,
                 logger=None):               

        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            max_features=max_features, bootstrap=bootstrap, oob_score=oob_score,
            verbose=verbose, random_state=random_state, n_jobs=n_jobs)


    def plot_fi(self):
        pass # TODO


    def dump_model(self, outdir='.'):
        joblib.dump(self.model, filename=os.path.join(outdir, 'model.' + RF_REGRESSOR.model_name + '.pkl'))
        # model_ = joblib.load(filename=os.path.join(run_outdir, 'lgb_reg_model.pkl'))



class LGBM_CLASSIFIER(BaseMLModel):
    # TODO: finish
    """ LightGBM classifier. """
    ml_objective = 'binary'
    model_name = 'lgb_cls'

    def __init__(self, eval_metric=['l2', 'l1'], n_jobs=1, random_state=None,
                 logger=None):
        self.eval_metric = eval_metric
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.logger = logger

        # ----- lightgbm "sklearn API" - start
        # self.model = lgb.LGBMModel(objective=LGBM_REGRESSOR.ml_objective,
        #                            n_jobs=self.n_jobs,
        #                            random_state=self.random_state)
        # ----- lightgbm "sklearn API" - end

        # ----- lightgbm "sklearn API" - start
        self.model = lgb.LGBMClassifier(
            n_jobs=self.n_jobs,
            random_state=self.random_state)
        # ----- lightgbm "sklearn API" - end



class XGBM_REGRESSOR(BaseMLModel):
    """ xgboost regressor. """
    ml_objective = 'regression'
    model_name = 'xgb_reg'



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


