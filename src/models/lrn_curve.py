"""
Functions to generate learning curves (analysis of how score changes with training set size).
"""
import os
import numpy as np
import pandas as pd

import matplotlib
# matplotlib.use('TkAgg')
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sklearn
from sklearn.model_selection import cross_validate
from sklearn.model_selection import ShuffleSplit, GroupShuffleSplit, KFold, GroupKFold

from pandas.api.types import is_string_dtype
from collections import OrderedDict
from sklearn.preprocessing import LabelEncoder

import utils
# from cvsplitter import GroupSplit, SimpleSplit, plot_ytr_yvl_dist


def calc_scores(model, xdata, ydata, metrics=None):
    """ Create dict of scores.
    Args:
        metrics : TODO allow to pass a string of metrics
    """
    preds = model.predict(xdata)
    scores = OrderedDict()

    scores['r2'] = sklearn.metrics.r2_score(y_true=ydata, y_pred=preds)
    scores['mean_absolute_error'] = sklearn.metrics.mean_absolute_error(y_true=ydata, y_pred=preds)
    scores['median_absolute_error'] = sklearn.metrics.median_absolute_error(y_true=ydata, y_pred=preds)
    scores['mean_squared_error'] = sklearn.metrics.mean_squared_error(y_true=ydata, y_pred=preds)

    # score_names = ['r2', 'mean_absolute_error', 'median_absolute_error', 'mean_squared_error']

    # # https://scikit-learn.org/stable/modules/model_evaluation.html
    # for metric_name, metric in metrics.items():
    #     if isinstance(metric, str):
    #         scorer = sklearn.metrics.get_scorer(metric_name) # get a scorer from string
    #         scores[metric_name] = scorer(ydata, preds)
    #     else:
    #         scores[metric_name] = scorer(ydata, preds)

    return scores


def my_learning_curve(estimator, X, Y,
                      cv=5,
                      groups=None,
                      lr_curve_ticks=5, data_sizes_frac=None,
                      args=None, fit_params=None,
                      metrics=['r2', 'neg_mean_absolute_error', 'neg_median_absolute_error', 'neg_mean_squared_error'],
                      n_jobs=1, random_state=None, logger=None, outdir='./'):
    """
    Train estimator using various train set sizes and generate learning curves for different metrics.
    The CV splitter splits the input dataset into cv_folds data subsets.
    Args:
        estimator : estimator that is consistent with sklearn api (has "fit" and "predict" methods)
        X : features matrix
        Y : target
        cv : sklearn's cv splitter --> scikit-learn.org/stable/glossary.html#term-cv-splitter
        groups : groups for the cv splits (used for strict cross-validation partitions --> non-overlapping cell lines)
        lr_curve_ticks : number of ticks in the learning curve (used if data_sizes_frac is None)
        data_sizes_frac : relative numbers of training samples that will be used to generate learning curves
        fit_params : dict of parameters to the estimator's "fit" method

        metrics : TODO allow to pass a string of metrics
        args : command line args (ignore this arg!)

    Examples:
        cv = sklearn.model_selection.KFold(n_splits=5, shuffle=False, random_state=0)
        estimator = sklearn.ensemble.RandomForestRegressor()
        my_learning_curve(estimator=estimator, cv=cv, lr_curve_ticks=5)
    """
    X = X.copy().reset_index(drop=True)
    Y = Y.copy().reset_index(drop=True)

    # TODO: try this! make this work!
    if isinstance(cv, int):
        cv_folds = cv
        cv = KFold(n_splits=cv_folds, shuffle=False, random_state=random_state)
    else:
        cv_folds = cv.get_n_splits()

    # Define training set sizes
    if data_sizes_frac is None:
        data_sizes_frac = np.linspace(0.1, 1.0, lr_curve_ticks)
    train_sizes = [int(n) for n in (cv_folds-1)/cv_folds * X.shape[0] * data_sizes_frac]

    if logger is not None:
        logger.info('Train sizes: {}'.format(train_sizes))

    # ---------------------------------------------------------------
    # Method 1
    # ---------------------------------------------------------------
    if is_string_dtype(groups):  # groups=='object'
        group_encoder = LabelEncoder()
        groups = group_encoder.fit_transform(groups)
    
    # ... Now start a nested loop of train size and cv folds ...
    tr_scores_all = [] # list dicts
    vl_scores_all = [] # list dicts

    # Start CV splits of the full dataset 
    #tr_cv_idx = OrderedDict()
    #vl_cv_idx = OrderedDict()
    for fold_id, (tr_idx, vl_idx) in enumerate(cv.split(X, Y, groups)):
        if logger is not None:
            logger.info(f'Fold {fold_id+1}/{cv_folds}')
        #tr_cv_idx[fold_id] = tr_idx
        #vl_cv_idx[fold_id] = vl_idx

        # Samples from this dataset are sampled for training
        xtr = X.iloc[tr_idx, :].reset_index(drop=True)
        ytr = Y[tr_idx].reset_index(drop=True)

        # A fixed set validation samples for the current CV split
        xvl = X.iloc[vl_idx, :].reset_index(drop=True)
        yvl = Y[vl_idx].reset_index(drop=True)

        # # Get the groups
        # tr_grps = groups[tr_idx]
        # vl_grps = groups[vl_idx]

        # # Confirm that group splits are correct ...
        # tr_grps_unq = set(tr_grps)
        # vl_grps_unq = set(vl_grps)
        # print('Total group (e.g., cell) intersections btw tr and vl: ', len(tr_grps_unq.intersection(vl_grps_unq)))
        # print('A few intersections : ', list(tr_grps_unq.intersection(vl_grps_unq))[:3])

        # Start run across data sizes
        idx = np.random.permutation(len(xtr))
        for tr_sz in train_sizes:
            if logger:
                logger.info(f'    Train size: {tr_sz}')      

            # Sequentially get subset of samples (the input dataset X must be shuffled)
            xtr_sub = xtr.iloc[idx[:tr_sz], :]
            ytr_sub = ytr[idx[:tr_sz]]
            #sub_grps = groups[idx[:tr_sz]]

            # Train model
            estimator.fit(xtr_sub, ytr_sub, **fit_params)

            # Calc scores
            tr_scores = calc_scores(model=estimator, xdata=xtr_sub, ydata=ytr_sub, metrics=None)
            vl_scores = calc_scores(model=estimator, xdata=xvl, ydata=yvl, metrics=None)

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
        cv_folds : (int) number of cv folds
        outdir : dir to store the plots
        args : 
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


def plt_learning_curve(rslt, metric_name='score', title=None, path=None):
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
    plt.savefig(path, bbox_inches='tight')


def scores_to_df(scores_all):
    df = pd.DataFrame(scores_all)
    df = df.melt(id_vars=['fold', 'tr_size', 'tr_set'])
    df = df.rename(columns={'variable': 'metric'})
    df = df.pivot_table(index=['metric', 'tr_size', 'tr_set'], columns=['fold'], values='value')
    df = df.reset_index(drop=False)
    return df

    