"""
Functions to generate learning curves.
Performance (error or score) vs training set size.
"""
import os
import numpy as np
import pandas as pd
from collections import OrderedDict

import matplotlib
# matplotlib.use('TkAgg')
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sklearn
from sklearn.model_selection import cross_validate

from sklearn.model_selection import ShuffleSplit, KFold
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

from pandas.api.types import is_string_dtype
from sklearn.preprocessing import LabelEncoder

import utils
import ml_models


# def reg_auroc(y_true, y_pred):
#     y_true = np.where(y_true < 0.5, 1, 0)
#     y_score = np.where(y_pred < 0.5, 1, 0)
#     auroc = sklearn.metrics.roc_auc_score(y_true, y_score)
#     return auroc


# def calc_preds(estimator, xdata, ydata, mltype):
#     """ Calc predictions. """
#     if mltype == 'cls':    
#         if ydata.ndim > 1 and ydata.shape[1] > 1:
#             y_preds = estimator.predict_proba(xdata)
#             y_preds = np.argmax(y_preds, axis=1)
#             y_true = np.argmax(ydata, axis=1)
#         else:
#             y_preds = estimator.predict_proba(xdata)
#             y_preds = np.argmax(y_preds, axis=1)
#             y_true = ydata

#     elif mltype == 'reg':
#         y_preds = estimator.predict(xdata)
#         y_true = ydata

#     return y_preds, y_true


# def calc_scores(y_true, y_preds, mltype, metrics=None):
#     """ Create dict of scores.
#     Args:
#         metrics : TODO allow to pass a string of metrics
#     """
#     scores = OrderedDict()

#     if mltype == 'cls':    
#         scores['auroc'] = sklearn.metrics.roc_auc_score(y_true, y_preds)
#         scores['f1_score'] = sklearn.metrics.f1_score(y_true, y_preds, average='micro')
#         scores['acc_blnc'] = sklearn.metrics.balanced_accuracy_score(y_true, y_preds)

#     elif mltype == 'reg':
#         scores['r2'] = sklearn.metrics.r2_score(y_true=y_true, y_pred=y_preds)
#         scores['mean_absolute_error'] = sklearn.metrics.mean_absolute_error(y_true=y_true, y_pred=y_preds)
#         scores['median_absolute_error'] = sklearn.metrics.median_absolute_error(y_true=y_true, y_pred=y_preds)
#         scores['mean_squared_error'] = sklearn.metrics.mean_squared_error(y_true=y_true, y_pred=y_preds)
#         scores['auroc_reg'] = reg_auroc(y_true=y_true, y_pred=y_preds)

#     # score_names = ['r2', 'mean_absolute_error', 'median_absolute_error', 'mean_squared_error']

#     # # https://scikit-learn.org/stable/modules/model_evaluation.html
#     # for metric_name, metric in metrics.items():
#     #     if isinstance(metric, str):
#     #         scorer = sklearn.metrics.get_scorer(metric_name) # get a scorer from string
#     #         scores[metric_name] = scorer(ydata, preds)
#     #     else:
#     #         scores[metric_name] = scorer(ydata, preds)

#     return scores


def my_learning_curve(X, Y,
                      mltype,
                      model_name='lgb_reg',
                      cv=5, groups=None,
                      lr_curve_ticks=5, data_sizes_frac=None,
                      args=None, fit_params=None, init_params=None,
                      metrics=['r2', 'neg_mean_absolute_error', 'neg_median_absolute_error', 'neg_mean_squared_error'],
                      n_jobs=1, random_state=None, logger=None, outdir='./'):
    """
    Train estimator using various train set sizes and generate learning curves for different metrics.
    The CV splitter splits the input dataset into cv_folds data subsets.
    Args:
        X : features matrix
        Y : target
        mltype : type to ML problem (`reg` or `cls`)
        cv : number cv folds or sklearn's cv splitter --> scikit-learn.org/stable/glossary.html#term-cv-splitter
        groups : groups for the cv splits (used for strict cv partitions --> non-overlapping cell lines)
        lr_curve_ticks : number of ticks in the learning curve (used if data_sizes_frac is None)
        data_sizes_frac : relative numbers of training samples that will be used to generate learning curves
        fit_params : dict of parameters to the estimator's "fit" method

        metrics : allow to pass a string of metrics  TODO!
        args : command line args

    Examples:
        cv = sklearn.model_selection.KFold(n_splits=5, shuffle=False, random_state=0)
        lrn_curve.my_learning_curve(X=xdata, Y=ydata, mltype='reg', cv=cv, lr_curve_ticks=5)
    """
    X = pd.DataFrame(X).values
    Y = pd.DataFrame(Y).values

    # TODO: didn't test!
    if isinstance(cv, int) and groups is None:
        cv_folds = cv
        cv = KFold(n_splits=cv_folds, shuffle=False, random_state=random_state)
    if isinstance(cv, int) and groups is not None:
        cv_folds = cv
        cv = GroupKFold(n_splits=cv_folds)
    else:
        cv_folds = cv.get_n_splits()

    # Define training set sizes
    if data_sizes_frac is None:
        #data_sizes_frac = np.linspace(0.1, 1.0, lr_curve_ticks)
        base = 10
        data_sizes_frac = np.logspace(0.0, 1.0, lr_curve_ticks, endpoint=True, base=base)/base

    if cv_folds == 1:
        train_sizes = [int(n) for n in (1-cv.test_size) * X.shape[0] * data_sizes_frac]
    elif cv_folds > 1:
        train_sizes = [int(n) for n in (cv_folds-1)/cv_folds * X.shape[0] * data_sizes_frac]

    if logger is not None:
        logger.info('Train sizes: {}'.format(train_sizes))

    # ---------------------------------------------------------------
    # Method 1
    # ---------------------------------------------------------------
    if is_string_dtype(groups):
        group_encoder = LabelEncoder()
        groups = group_encoder.fit_transform(groups)
    
    # ... Now start a nested loop of train size and cv folds ...
    tr_scores_all = [] # list dicts
    vl_scores_all = [] # list dicts

    if mltype == 'cls':
        if Y.ndim > 1 and Y.shape[1] > 1:
            splitter = cv.split(X, np.argmax(Y, axis=1), groups=groups)
        else:
            splitter = cv.split(X, Y, groups=groups)
    elif mltype == 'reg':
        splitter = cv.split(X, Y, groups=groups)

    for fold_id, (tr_idx, vl_idx) in enumerate(splitter):
        if logger is not None:
            logger.info(f'Fold {fold_id+1}/{cv_folds}')

        # Samples from this dataset are sampled for training
        xtr = X[tr_idx, :]
        ytr = Y[tr_idx, :]

        # A fixed set of validation samples for the current CV split
        xvl = X[vl_idx, :]
        yvl = np.squeeze(Y[vl_idx, :])        

        # # Confirm that group splits are correct ...
        # tr_grps_unq = set(groups[tr_idx])
        # vl_grps_unq = set(groups[vl_idx])
        # print('Total group (e.g., cell) intersections btw tr and vl: ', len(tr_grps_unq.intersection(vl_grps_unq)))
        # print('A few intersections : ', list(tr_grps_unq.intersection(vl_grps_unq))[:3])

        # Start run across data sizes
        idx = np.random.permutation(len(xtr))
        for i, tr_sz in enumerate(train_sizes):
            if logger:
                logger.info(f'    Train size: {tr_sz} ({i+1}/{len(train_sizes)})')   

            # Sequentially get subset of samples (the input dataset X must be shuffled)
            xtr_sub = xtr[idx[:tr_sz], :]
            ytr_sub = np.squeeze(ytr[idx[:tr_sz], :])            
            #sub_grps = groups[idx[:tr_sz]]

            # Get the estimator
            estimator = ml_models.get_model(model_name=model_name, init_params=init_params)

            if 'nn' in model_name:
                from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard
                from keras.utils import plot_model

                # Create output dir
                out_nn_model = os.path.join(outdir, 'cv'+str(fold_id+1) + '_sz'+str(tr_sz))
                plot_model(estimator.model, to_file=os.path.join(out_nn_model, 'model.png'))

                # Add keras callbacks
                os.makedirs(out_nn_model, exist_ok=False)
                checkpointer = ModelCheckpoint(filepath=os.path.join(out_nn_model, 'autosave.model.h5'), verbose=0, save_weights_only=False, save_best_only=True)
                csv_logger = CSVLogger(filename=os.path.join(out_nn_model, 'training.log'))
                reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.75, patience=20, verbose=1, mode='auto',
                                              min_delta=0.0001, cooldown=3, min_lr=0.000000001)
                early_stop = EarlyStopping(monitor='val_loss', patience=100, verbose=1, mode='auto')
                callback_list = [checkpointer, csv_logger, early_stop, reduce_lr]
                fit_params['callbacks'] = callback_list

                # Set validation set
                #fit_params['validation_data'] = (xvl, yvl)
                fit_params['validation_split'] = 0.1

            # Train model
            history = estimator.model.fit(xtr_sub, ytr_sub, **fit_params)

            # Calc preds and scores TODO: dump preds
            # ... training set
            y_preds, y_true = utils.calc_preds(estimator=estimator.model, xdata=xtr_sub, ydata=ytr_sub, mltype=mltype)
            tr_scores = utils.calc_scores(y_true=y_true, y_preds=y_preds, mltype=mltype, metrics=None)
            # ... val set
            y_preds, y_true = utils.calc_preds(estimator=estimator.model, xdata=xvl, ydata=yvl, mltype=mltype)
            vl_scores = utils.calc_scores(y_true=y_true, y_preds=y_preds, mltype=mltype, metrics=None)

            if 'nn' in model_name:
                # Summarize history for loss     
                pr_metrics = ml_models.get_keras_performance_metrics(history)
                epochs = np.asarray(history.epoch) + 1
                hh = history.history
                for p, m in enumerate(pr_metrics):
                    metric_name = m
                    metric_name_val = 'val_' + m
                    
                    ymin = min(set(hh[metric_name]).union(hh[metric_name_val]))
                    ymax = max(set(hh[metric_name]).union(hh[metric_name_val]))

                    plt.figure()
                    plt.plot(epochs, hh[metric_name], 'b.-', alpha=0.6, label=metric_name)
                    plt.plot(epochs, hh[metric_name_val], 'r.-', alpha=0.6, label=metric_name_val)
                    plt.title(f'train size: {tr_sz}')
                    plt.xlabel('epoch')
                    plt.ylabel(metric_name)
                    plt.xlim([0.5, len(epochs) + 0.5])
                    plt.ylim([ymin-0.1, ymax+0.1])
                    plt.grid(True)
                    plt.legend([metric_name, metric_name_val], loc='best')
                    
                    plt.savefig(os.path.join(out_nn_model, metric_name+'_curve.png'), bbox_inches='tight')
                    plt.close()

            # Add info
            tr_scores['tr_set'] = True
            vl_scores['tr_set'] = False
            tr_scores['fold'] = 'f'+str(fold_id)
            vl_scores['fold'] = 'f'+str(fold_id)
            tr_scores['tr_size'] = tr_sz
            vl_scores['tr_size'] = tr_sz

            # Aggregate scores
            tr_scores_all.append(tr_scores)
            vl_scores_all.append(vl_scores)

            # Delete the estimator/model
            del estimator, history

        # Dump intermediate results
        tr_df_tmp = scores_to_df(tr_scores_all)
        vl_df_tmp = scores_to_df(vl_scores_all)
        scores_all_df_tmp = pd.concat([tr_df_tmp, vl_df_tmp], axis=0)
        scores_all_df_tmp.to_csv(os.path.join(outdir, model_name + '_lrn_curve_scores_tmp_cv' + str(fold_id+1) + '.csv'), index=False)

    tr_df = scores_to_df(tr_scores_all)
    vl_df = scores_to_df(vl_scores_all)
    scores_all_df = pd.concat([tr_df, vl_df], axis=0)
    # ---------------------------------------------------------------

    # ---------------------------------------------------------------
    # Method 2
    # ---------------------------------------------------------------
    #  # Define training set sizes
    # if data_sizes_frac is None:
    #     data_sizes_frac = np.linspace(0.1, 1.0, lr_curve_ticks)
    # data_sizes = [int(n) for n in X.shape[0]*data_sizes_frac]

    # if logger is not None:
    #     logger.info('Dataset sizes: {}'.format(data_sizes))

    # # List to collect cv trianing scores for different data sizes
    # scores_all_list = [] 
    
    # # Start run across data sizes
    # for d_size in data_sizes:
    #     if logger:
    #         logger.info(f'Data size: {d_size}')      

    #     xdata = X.iloc[idx[:d_size], :]
    #     ydata = Y[idx[:d_size]]
    #     sub_groups = groups[idx[:d_size]]

    #     cv_scores = cross_validate(
    #         estimator=sklearn.base.clone(estimator),
    #         X=xdata, y=ydata,
    #         scoring=metrics, cv=cv, groups=sub_groups,
    #         n_jobs=n_jobs, fit_params=fit_params)

    #     df = utils.update_cross_validate_scores(cv_scores)
    #     df.insert(loc=df.shape[1]-cv_folds, column='tr_size', value=d_size)
                
    #     # Append results to master df
    #     scores_all_list.append(df)

    # # Concat results for data_sizes
    # scores_all_df = pd.concat(scores_all_list, axis=0).reset_index(drop=True)
    # ---------------------------------------------------------------   

    # Plot learning curves
    plt_learning_curve_multi_metric(df=scores_all_df, cv_folds=cv_folds,
                                    outdir=outdir, args=args)
    return scores_all_df


def plt_learning_curve_multi_metric(df, cv_folds, outdir, args=None):
    """
    Args:
        df : contains train and val scores for cv folds (the scores are the last cv_folds cols)
            metric | tr_set | tr_size |  f0  |  f1  |  f2  |  f3  |  f4
          ----------------------------------------------------------------
              r2   |  True  |   200   | 0.95 | 0.98 | 0.97 | 0.91 | 0.92
              r2   |  False |   200   | 0.21 | 0.27 | 0.22 | 0.25 | 0.24
              mae  |  True  |   200   | 0.11 | 0.12 | 0.15 | 0.10 | 0.18
              mae  |  False |   200   | 0.34 | 0.37 | 0.35 | 0.33 | 0.30
              r2   |  True  |   600   | 0.75 | 0.78 | 0.77 | 0.71 | 0.72
              r2   |  False |   600   | 0.41 | 0.47 | 0.42 | 0.45 | 0.44
              mae  |  True  |   600   | 0.21 | 0.22 | 0.25 | 0.20 | 0.28
              mae  |  False |   600   | 0.34 | 0.37 | 0.35 | 0.33 | 0.30
              ...  |  ..... |   ...   | .... | .... | .... | .... | ....
        cv_folds : (int) number of cv folds
        outdir : dir to save the plots
        args : command line args
    """
    df = df.copy()
    data_sizes = sorted(df['tr_size'].unique())

    for metric_name in df['metric'].unique():
        aa = df[df['metric']==metric_name].reset_index(drop=True)
        aa.sort_values('tr_size', inplace=True)

        tr = aa[aa['tr_set']==True]
        vl = aa[aa['tr_set']==False]

        tr = tr.iloc[:, -cv_folds:]
        vl = vl.iloc[:, -cv_folds:]

        rslt = []
        rslt.append(data_sizes)
        rslt.append(tr.values)
        rslt.append(vl.values)

        if args is not None:
            fname = 'learning_curve_' + args['target_name'] + '_' + metric_name + '.png'
            title = 'Learning curve (target: {}, data: {})'.format(args['target_name'], '_'.join(args['train_sources']))
        else:
            fname = 'learning_curve_' + metric_name + '.png'
            title = 'Learning curve'

        path = os.path.join(outdir, fname)
        plt_learning_curve(rslt=rslt, metric_name=metric_name,
            title=title, path=path)


def plt_learning_curve(rslt, metric_name='score', ylim=None, title=None, path=None):
    """ 
    Args:
        rslt : output from sklearn.model_selection.learning_curve()
            rslt[0] : 1-D array (n_ticks, ) -> vector of train set sizes
            rslt[1] : 2-D array (n_ticks, n_cv_folds) -> tr scores
            rslt[2] : 2-D array (n_ticks, n_cv_folds) -> vl scores
    """
    tr_sizes  = rslt[0]
    tr_scores = rslt[1]
    te_scores = rslt[2]
    
    fig = plt.figure()
    tr_scores_mean = np.mean(tr_scores, axis=1)
    tr_scores_std  = np.std(tr_scores, axis=1)
    te_scores_mean = np.mean(te_scores, axis=1)
    te_scores_std  = np.std(te_scores, axis=1)

    plt.plot(tr_sizes, tr_scores_mean, 'o-', color='r', label="Train score")
    plt.plot(tr_sizes, te_scores_mean, 'o-', color='g', label="Val score")
    plt.fill_between(tr_sizes,
                     tr_scores_mean - tr_scores_std,
                     tr_scores_mean + tr_scores_std,
                     alpha=0.1, color='r')
    plt.fill_between(tr_sizes,
                     te_scores_mean - te_scores_std,
                     te_scores_mean + te_scores_std,
                     alpha=0.1, color='g')
    
    if title is not None:
        plt.title(title)
    else:
        plt.title('Learning curve')
    plt.xlabel('Train set size')
    plt.ylabel(metric_name)
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    if ylim is not None:
        plt.ylim(ylim)
    plt.savefig(path, bbox_inches='tight')


def scores_to_df(scores_all):
    df = pd.DataFrame(scores_all)
    df = df.melt(id_vars=['fold', 'tr_size', 'tr_set'])
    df = df.rename(columns={'variable': 'metric'})
    df = df.pivot_table(index=['metric', 'tr_size', 'tr_set'], columns=['fold'], values='value')
    df = df.reset_index(drop=False)
    df.columns.name = None
    return df

