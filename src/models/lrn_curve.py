"""
Functions to generate learning curves (i.e., analysis of how score changes with training set size).
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

import utils
# from cvsplitter import GroupSplit, SimpleSplit, plot_ytr_yvl_dist


def my_learning_curve(estimator, X, Y,
                      args=None, fit_params=None,
                      lr_curve_ticks=5, data_sizes_frac=None,
                      metrics=['r2', 'neg_mean_absolute_error', 'neg_median_absolute_error', 'mean_squared_error'],
                      cv_method='simple', cv_folds=5, groups=None,
                      n_jobs=1, random_state=None, logger=None, outdir='./'):
    """
    Train estimator using multiple train set sizes and multiple metrics.
    Generate learning rate curve for each metric.
    Uses sklearn cv splits.
    Uses sklearn cross_validate.
    Args:
        estimator : estimator that is consistent with sklearn api 
        X : features matrix
        Y : target
        args : command line args
        lr_curve_ticks : number of ticks in the learning curve (used is data_sizes_frac is None)
        data_sizes_frac : relative numbers of training samples that will be used to generate the learning curve
        cv_method : 'simple', 'group'
        cv_folds : number of cv folds
        groups : groups for the cv splits when cv_method='group'
    """
    X.copy().reset_index(drop=True, inplace=True)
    Y.copy().reset_index(drop=True, inplace=True)

    # Define CV method
    if groups is not None:
        cv = GroupKFold(n_splits=cv_folds)
    else:
        cv = KFold(n_splits=cv_folds, shuffle=False, random_state=random_state)

    # Define training set sizes
    if data_sizes_frac is None:
        data_sizes_frac = np.linspace(0.1, 1.0, lr_curve_ticks)
    data_sizes = [int(n) for n in X.shape[0]*data_sizes_frac]

    # List to collect cv trianing scores for different data sizes
    scores_all_list = [] 

    # Start across data sizes
    idx = np.random.permutation(len(X))
    for d_size in data_sizes:
        if logger:
            logger.info(f'Data size: {d_size}')      

        xdata = X.iloc[idx[:d_size], :]
        ydata = Y[idx[:d_size]]

        scores = cross_validate(
            estimator=sklearn.base.clone(estimator),
            X=xdata, y=ydata,
            scoring=metrics, cv=cv, groups=groups,
            n_jobs=n_jobs, fit_params=fit_params)

        df = utils.cv_scores_to_df(scores, decimals=3, calc_stats=False)
        df.insert(loc=0, column='data_size', value=d_size)

        v = list(map(lambda x: '_'.join(x.split('_')[1:]), df.index))
        df.insert(loc=0, column='metric', value=v)

        # Convert `neg` metric to positive and update metric names
        # scikit-learn.org/stable/modules/model_evaluation.html --> explains the `neg` in `neg_mean_absolute_error`
        idx_bool = [True if 'neg_' in s else False for s in df['metric']]
        for i, bl in enumerate(idx_bool):
            if bl:
                df.iloc[i, -cv_folds:] = abs(df.iloc[i, -cv_folds:])
        df['metric'] = df['metric'].map(lambda s: s.split('neg_')[-1] if 'neg_' in s else s)

        v = list(map(lambda x: True if 'train' in x else False, df.index))
        df.insert(loc=1, column='train_set', value=v)
                
        # Append results to master df
        scores_all_list.append(df)

    # Concat results for data_sizes
    scores_all_df = pd.concat(scores_all_list, axis=0).reset_index(drop=True)

    # Plot learning curves
    plt_learning_curve_multi_metric(df=scores_all_df, cv_folds=cv_folds,
                                    outdir=outdir, args=args)
    return scores_all_df


def plt_learning_curve_multi_metric(df, cv_folds, outdir, args=None):
    """
    Args:
        df : contains train and val scores for cv folds (the scores are the last cv_folds cols)
        cv_folds : (int) number of cv folds
        target_name : (str) target name 
    """
    df = df.copy()
    data_sizes = sorted(df['data_size'].unique())

    for metric_name in df['metric'].unique():
        aa = df[df['metric']==metric_name].reset_index(drop=True)
        aa.sort_values('data_size', inplace=True)

        tr = aa[aa['train_set']==True]
        vl = aa[aa['train_set']==False]

        tr = tr.iloc[:, -cv_folds:]
        vl = vl.iloc[:, -cv_folds:]

        rslt = []
        rslt.append(data_sizes)
        rslt.append(tr.values)
        rslt.append(vl.values)

        fname = 'learning_curve_' + args.target_name + '_' + metric_name + '.png'
        path = os.path.join(outdir, fname)
        plt_learning_curve(rslt=rslt, metric_name=metric_name,
            title='Learning curve (target: {}, data: {})'.format(args.target_name, '_'.join(args.train_sources)),
            path=path)


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


