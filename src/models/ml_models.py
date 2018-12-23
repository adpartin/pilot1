import os
import time
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.metrics import r2_score, mean_absolute_error, median_absolute_error, explained_variance_score


# TODO: create a super class ml model
# class BASE_ML_MODEL(): 


class LGBM_Regressor():
    def __init__(self, target_name=None, eval_metric=['l1', 'l2'], n_jobs=1, random_state=None, logger=None):
        # https://lightgbm.readthedocs.io/en/latest/Python-API.html
        try:
            import lightgbm as lgb
        except ImportError:  # install??
            logger.error('Module not found (lightgbm)')

        ml_objective = 'regression'
        self.model_name = 'lgb_reg'
        
        # TODO: use config file to set default parameters (like in candle)
        self.target_name = target_name
        self.eval_metric = eval_metric
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.logger = logger

        if logger is not None:
            self.logger.info('\nTrain LGBMRegressor ...')
        
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
        self.model = lgb.LGBMModel(objective=ml_objective,
                                   n_jobs=self.n_jobs,
                                   random_state=self.random_state)


    def fit(self, xtr, ytr, eval_set=None):
        self.eval_set = eval_set
        self.xtr = xtr
        self.ytr = ytr
        
        t0 = time.time()
        self.model.fit(self.xtr, self.ytr, eval_metric=self.eval_metric, eval_set=self.eval_set,
                       early_stopping_rounds=10, verbose=False, callbacks=None)
        self.train_runtime = time.time() - t0

        if self.logger is not None:
            self.logger.info('Runtime: {:.2f} mins'.format(self.train_runtime/60))


    def save_model(self, outdir='./'):
        # lgb_reg.save_model(os.path.join(run_outdir, 'lgb_'+ml_type+'_model.txt'))
        joblib.dump(self.model, filename=os.path.join(outdir, self.model_name+'_model.pkl'))
        # lgb_reg_ = joblib.load(filename=os.path.join(run_outdir, 'lgb_reg_model.pkl'))


    def calc_scores(self, xdata, ydata, to_print=False):
        scores = {}
        preds = self.model.predict(xdata)
        scores['r2_score'] = r2_score(ydata, preds)
        scores['mean_abs_error'] = mean_absolute_error(ydata, preds)
        scores['median_abs_error'] = median_absolute_error(ydata, preds)
        # scores['explained_variance_score'] = explained_variance_score(ydata, preds)

        if to_print:
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

        self.scores = scores
        return self.scores


    def dump_preds(self, df_data, xdata, target_name, path, model_name=None):
        """
        Args:
            df_data : df that contains the cell and drug names, and target value
            xdata : features to make predictions
            target_name : name of the target as it appears in the df (e.g. 'AUC')
        """
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
        df_preds.to_csv(path)




    # # Print preds
    # # utils.print_scores(model=lgb_reg, xdata=xtr, ydata=ytr)
    # # utils.print_scores(model=lgb_reg, xdata=xvl, ydata=yvl)
    # utils.print_scores(model=lgb_reg, xdata=xvl, ydata=yvl, logger=logger)

    # # Dump preds
    # utils.dump_preds(model=lgb_reg, df_data=vl_data, xdata=xvl, target_name=target_name,
    #                  path=os.path.join(run_outdir, preds_filename_prefix+'_'+model_name+'_preds.csv'))

    # # Plot feature importance
    # lgb.plot_importance(booster=lgb_reg, max_num_features=20, grid=True, title='LGBMRegressor')
    # plt.tight_layout()
    # plt.savefig(os.path.join(run_outdir, model_name+'_importances.png'))

    # # Plot learning curves
    # # TODO: note, plot_metric didn't accept 'mae' although it's alias for 'l1' 
    # # TODO: plot_metric requires dict from train(), but train returns 'lightgbm.basic.Booster'??
    # for m in eval_metric:
    #     ax = lgb.plot_metric(booster=lgb_reg, metric=m, grid=True)
    #     plt.savefig(os.path.join(run_outdir, model_name+'_learning_curve_'+m+'.png'))


    