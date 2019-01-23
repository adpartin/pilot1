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

# from cvsplitter import GroupSplit, SimpleSplit, plot_ytr_yvl_dist


def my_learning_curve(estimator,
                      X, Y,  # data
                      target_name,
                      fit_params=None,
                      lr_curve_ticks=5, data_sizes_frac=None,
                      metrics=['r2', 'neg_mean_absolute_error', 'neg_median_absolute_error'],
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

    # Lists to collect results for the data size runs
    df_tr = []
    df_vl = []

    idx = np.random.permutation(len(X))
    for d_size in data_sizes:
        if logger:
            logger.info(f'Data size: {d_size}')
        #data_sample = data.sample(n=d_size)
        #xdata, _ = utils_tidy.split_features_and_other_cols(data=data_sample, fea_prfx_dict=fea_prfx_dict)
        #ydata = utils_tidy.extract_target(data=data_sample, target_name=target_name)        

        xdata = X.iloc[idx[:d_size], :]
        ydata = Y[idx[:d_size]]

        scores = cross_validate(
            estimator=sklearn.base.clone(estimator),
            X=xdata, y=ydata,
            scoring=metrics, cv=cv, groups=groups,
            n_jobs=n_jobs, fit_params=fit_params)
        
        df = pd.DataFrame(scores).drop(columns=['fit_time', 'score_time']).T
        df.columns = ['f'+str(c) for c in df.columns]
        
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

        df.reset_index(inplace=True)
        v = list(map(lambda x: 'tr' if 'train' in x else 'vl', df['index'].values))
        df.insert(loc=1, column='type', value=v)
        
        tr_cv_scores = df[df['type']=='tr'].drop(columns=['index'])
        vl_cv_scores = df[df['type']=='vl'].drop(columns=['index'])
        
        # Append results to master dfs
        df_tr.append(tr_cv_scores)
        df_vl.append(vl_cv_scores)

    # Concat results for data_sizes
    df_tr = pd.concat(df_tr, axis=0)
    df_vl = pd.concat(df_vl, axis=0)

    # Plot learning curves
    plt_learning_curve_multi_metric(
        df_tr=df_tr, df_vl=df_vl, cv_folds=cv_folds,
        target_name=target_name, outdir=outdir)



def plt_learning_curve_multi_metric(df_tr, df_vl, cv_folds, target_name, outdir):
    """
    Args:
        df_tr : (df) contains train scores across folds (last cv_folds columns)
        df_vl : 
        cv_folds : (int) number of cv folds
        target_name : (str) target name 
    """
    data_sizes = sorted(df_tr['data_size'].unique())

    for metric_name in df_tr['metric'].unique():
        tr = df_tr[df_tr['metric']==metric_name].reset_index(drop=True)
        tr.sort_values('data_size', inplace=True)
        tr = tr.iloc[:, -cv_folds:]

        vl = df_vl[df_vl['metric']==metric_name].reset_index(drop=True)
        vl.sort_values('data_size', inplace=True)
        vl = vl.iloc[:, -cv_folds:]

        rslt = []
        rslt.append(data_sizes)
        rslt.append(tr.values)
        rslt.append(vl.values)

        plt_learning_curve(rslt=rslt, metric_name=metric_name,
            title='Learning curve (target: {})'.format(target_name),
            path=os.path.join(outdir, 'learning_curve_' + metric_name + '.png'))


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


