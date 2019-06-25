"""
Functions to generate learning curves.
Records performance (error or score) vs training set size.
"""
from comet_ml import Experiment
import os

import sys
from pathlib import Path
from collections import OrderedDict

import sklearn
import numpy as np
import pandas as pd

import matplotlib
# matplotlib.use('TkAgg')
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_validate

from sklearn.model_selection import ShuffleSplit, KFold
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

from pandas.api.types import is_string_dtype
from sklearn.preprocessing import LabelEncoder

from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard
from keras.utils import plot_model

# Utils
import utils
import ml_models

# Import custom callbacks
keras_contrib = '/vol/ml/apartin/projects/keras-contrib/keras_contrib/callbacks'
sys.path.append(keras_contrib)
from cyclical_learning_rate import CyclicLR


def my_learning_curve(X, Y,
                      mltype,
                      model_name='lgb_reg',
                      init_params=None, 
                      fit_params=None,
                      cv=5,
                      cv_groups=None,
                      cv_splits=None,
                      lc_ticks=5,
                      data_sz_frac=None,
                      args=None,
                      metrics=['r2', 'neg_mean_absolute_error', 'neg_median_absolute_error', 'neg_mean_squared_error'],
                      n_jobs=1, random_state=None, logger=None, outdir='.'):
    """
    Train estimator using various train set sizes and generate learning curves for different metrics.
    The CV splitter splits the input dataset into cv_folds data subsets.
    Args:
        X : features matrix
        Y : target
        mltype : type to ML problem (`reg` or `cls`)
        cv : number cv folds or sklearn's cv splitter --> scikit-learn.org/stable/glossary.html#term-cv-splitter
        cv_splits : tuple of 2 dicts cv_splits[0] and cv_splits[1] contain the tr and vl splits, respectively 
        cv_groups : groups for the cv splits (used for strict cv partitions --> non-overlapping cell lines)
        lc_ticks : number of ticks in the learning curve (used if data_sz_frac is None)
        data_sz_frac : relative numbers of training samples that will be used to generate learning curves
        fit_params : dict of parameters to the estimator's "fit" method

        metrics : allow to pass a string of metrics  TODO!
        args : command line args

    Examples:
        cv = sklearn.model_selection.KFold(n_splits=5, shuffle=False, random_state=0)
        lrn_curve.my_learning_curve(X=xdata, Y=ydata, mltype='reg', cv=cv, lc_ticks=5)
    """ 
    X = pd.DataFrame(X).values
    Y = pd.DataFrame(Y).values

    # --------------------------------
    # Store splits of indices in dicts 
    # --------------------------------
    tr_dct = {}
    vl_dct = {}

    # Use splits passed as input arg
    if cv_splits is not None:
        tr_id = cv_splits[0]
        vl_id = cv_splits[1]
        assert tr_id.shape[1]==vl_id.shape[1], 'tr and vl must have the same of folds.'
        cv_folds = tr_id.shape[1]

        for i in range(tr_id.shape[1]):
            tr_dct[i] = tr_id.iloc[:, i].dropna().values.astype(int).tolist()
            vl_dct[i] = vl_id.iloc[:, i].dropna().values.astype(int).tolist()

        if tr_id.shape[1] == 1:
            vl_size = vl_id.shape[0]/(vl_id.shape[0] + tr_id.shape[0])

    # Generate splits on the fly
    else:
        # TODO: didn't test!
        if isinstance(cv, int) and cv_groups is None:
            cv_folds = cv
            cv = KFold(n_splits=cv_folds, shuffle=False, random_state=random_state)
        if isinstance(cv, int) and cv_groups is not None:
            cv_folds = cv
            cv = GroupKFold(n_splits=cv_folds)
        else:
            cv_folds = cv.get_n_splits() # cv is a sklearn splitter

        if cv_folds == 1:
            vl_size = cv.test_size

        # Encode the group labels
        if is_string_dtype(cv_groups):
            grp_enc= LabelEncoder()
            cv_grp = grp_enc.fit_transform(cv_grp)
    
        # Create sklearn splitter 
        if mltype == 'cls':
            if Y.ndim > 1 and Y.shape[1] > 1:
                splitter = cv.split(X, np.argmax(Y, axis=1), groups=cv_grp)
            else:
                splitter = cv.split(X, Y, groups=cv_grp)
        elif mltype == 'reg':
            splitter = cv.split(X, Y, groups=cv_grp)
        
        # Generate the splits
        for i, (tr_vec, vl_vec) in enumerate(splitter):
            tr_dct[i] = tr_vec
            vl_dct[i] = vl_vec


    # Define training set sizes
    if data_sz_frac is None:
        # data_sz_frac = np.linspace(0.1, 1.0, lc_ticks) # linear spacing
        base = 10
        data_sz_frac = np.logspace(0.0, 1.0, lc_ticks, endpoint=True, base=base)/base # log spacing

    if cv_folds == 1:
        # tr_sizes = [int(n) for n in (1-cv.test_size) * X.shape[0] * data_sz_frac]
        tr_sizes = [int(n) for n in (1-vl_size) * X.shape[0] * data_sz_frac]
    elif cv_folds > 1:
        tr_sizes = [int(n) for n in (cv_folds-1)/cv_folds * X.shape[0] * data_sz_frac]

    if logger is not None:
        logger.info('Train sizes: {}'.format(tr_sizes))

    
    # Now start nested loop of train size and cv folds
    tr_scores_all = [] # list of dicts
    vl_scores_all = [] # list of dicts

    # CV loop
    for fold_id, (tr_k, vl_k) in enumerate(zip( tr_dct.keys(), vl_dct.keys() )):
        if logger is not None:
            logger.info(f'Fold {fold_id+1}/{cv_folds}')

        tr_id = tr_dct[tr_k]
        vl_id = vl_dct[vl_k]

        # Samples from this dataset are sampled for training
        xtr = X[tr_id, :]
        ytr = Y[tr_id, :]

        # A fixed set of validation samples for the current CV split
        xvl = X[vl_id, :]
        yvl = np.squeeze(Y[vl_id, :])        

        # Start run across data sizes
        idx = np.random.permutation(len(xtr))
        for i, tr_sz in enumerate(tr_sizes):
            if logger:
                logger.info(f'    Train size: {tr_sz} ({i+1}/{len(tr_sizes)})')   

            # Sequentially get a subset of samples (the input dataset X must be shuffled)
            xtr_sub = xtr[idx[:tr_sz], :]
            ytr_sub = np.squeeze(ytr[idx[:tr_sz], :])            
            #sub_grps = groups[idx[:tr_sz]]

            # Get the estimator
            estimator = ml_models.get_model(model_name, init_params=init_params)

            if 'nn' in model_name:
                plot_model(estimator.model, to_file=outdir/'nn_model.png')

                # Create output dir
                out_nn_model = outdir / ('cv'+str(fold_id+1) + '_sz'+str(tr_sz))
                os.makedirs(out_nn_model, exist_ok=False)
                
                # Callbacks (custom)
                clr = CyclicLR(base_lr=0.0001, max_lr=0.001, mode='triangular')
                
                # Keras callbacks
                checkpointer = ModelCheckpoint(str(out_nn_model/'autosave.model.h5'), verbose=0, save_weights_only=False, save_best_only=True)
                csv_logger = CSVLogger(out_nn_model/'training.log')
                reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.75, patience=20, verbose=1, mode='auto',
                                              min_delta=0.0001, cooldown=3, min_lr=0.000000001)
                early_stop = EarlyStopping(monitor='val_loss', patience=60, verbose=1, mode='auto')
                
                # Callbacks list
                if (args is not None) and (args['opt']=='clr'):
                    callback_list = [checkpointer, csv_logger, early_stop, reduce_lr, clr]
                else:
                    callback_list = [checkpointer, csv_logger, early_stop, reduce_lr]
                    # TODO: which val set should be used??
                    # fit_params['validation_data'] = (xvl, yvl)
                    # fit_params['validation_split'] = 0.2

                # Fit params
                fit_params['validation_split'] = 0.2
                fit_params['callbacks'] = callback_list

            # Train model
            history = estimator.model.fit(xtr_sub, ytr_sub, **fit_params)

            # Calc preds and scores TODO: dump preds
            # ... training set
            y_preds, y_true = utils.calc_preds(estimator=estimator.model, x=xtr_sub, y=ytr_sub, mltype=mltype)
            tr_scores = utils.calc_scores(y_true=y_true, y_preds=y_preds, mltype=mltype, metrics=None)
            # ... val set
            y_preds, y_true = utils.calc_preds(estimator=estimator.model, x=xvl, y=yvl, mltype=mltype)
            vl_scores = utils.calc_scores(y_true=y_true, y_preds=y_preds, mltype=mltype, metrics=None)

            if 'nn' in model_name:
                ml_models.plot_prfrm_metrics(history=history, title=f'Train size: {tr_sz}',
                                             skp_ep=20, add_lr=True, outdir=out_nn_model)

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
        scores_all_df_tmp.to_csv(outdir / (model_name + '_lrn_crv_scores_cv' + str(fold_id+1) + '.csv'), index=False)

    tr_df = scores_to_df(tr_scores_all)
    vl_df = scores_to_df(vl_scores_all)
    scores_all_df = pd.concat([tr_df, vl_df], axis=0)

    # Plot learning curves
    figs = plt_lrn_crv_multi_metric(df=scores_all_df, cv_folds=cv_folds, outdir=outdir, args=args)
    
    return scores_all_df


def plt_lrn_crv_multi_metric(df, cv_folds, outdir, args=None):
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

    figs = []
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

        fname = 'lrn_crv_' + metric_name + '.png'
        title = 'Learning curve'

        path = outdir / fname
        fig = plt_lrn_crv(rslt=rslt, metric_name=metric_name, title=title, path=path)
        figs.append(fig)
        
    return figs


def plt_lrn_crv(rslt, metric_name='score', ylim=None, title=None, path=None):
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

    return fig


def scores_to_df(scores_all):
    df = pd.DataFrame(scores_all)
    df = df.melt(id_vars=['fold', 'tr_size', 'tr_set'])
    df = df.rename(columns={'variable': 'metric'})
    df = df.pivot_table(index=['metric', 'tr_size', 'tr_set'], columns=['fold'], values='value')
    df = df.reset_index(drop=False)
    df.columns.name = None
    return df

