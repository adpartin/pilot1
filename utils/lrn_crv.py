"""
Functions to generate learning curves.
Records performance (error or score) vs training set size.
"""
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

import keras
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard
from keras.utils import plot_model

# Utils
import utils
import ml_models
from ml_models import r2_krs

# Import custom callbacks
keras_contrib = '/vol/ml/apartin/projects/keras-contrib/keras_contrib/callbacks'
sys.path.append(keras_contrib)
from cyclical_learning_rate import CyclicLR


def my_learning_curve(
        X, Y,
        mltype: str,
        model_name: str='lgb_reg',
        init_kwargs: dict=None, 
        fit_kwargs: dict=None,
        cv=5,
        cv_splits=None,
        lc_ticks: int=5,
        tick_spacing: str='log',
        data_sz_frac=None,
        args=None,
        metrics: list=['r2', 'neg_mean_absolute_error', 'neg_median_absolute_error', 'neg_mean_squared_error'],
        n_jobs=1, random_state=None, logger=None, outdir='.'):
    """
    Train estimator using multiple train set sizes and generate learning curves for multiple metrics.
    The CV splitter splits the input dataset into cv_folds data subsets.
    Args:
        X : features matrix
        Y : target
        mltype : type to ML problem (`reg` or `cls`)
        cv : number cv folds or sklearn's cv splitter --> scikit-learn.org/stable/glossary.html#term-cv-splitter
        cv_splits : tuple of 2 dicts cv_splits[0] and cv_splits[1] contain the tr and vl splits, respectively 
        lc_ticks : number of ticks in the learning curve (used if data_sz_frac is None)
        data_sz_frac : relative numbers of training samples that will be used to generate learning curves
        fit_kwargs : dict of parameters to the estimator's "fit" method

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
        assert tr_id.shape[1] == vl_id.shape[1], 'tr and vl must have the same of folds.'
        cv_folds = tr_id.shape[1]

        for i in range(tr_id.shape[1]):
            tr_dct[i] = tr_id.iloc[:, i].dropna().values.astype(int).tolist()
            vl_dct[i] = vl_id.iloc[:, i].dropna().values.astype(int).tolist()

        if tr_id.shape[1] == 1:
            vl_size = vl_id.shape[0]/(vl_id.shape[0] + tr_id.shape[0])

    # If pre-defined splits are not passed, then generate splits on the fly
    else:
        if isinstance(cv, int):
            cv_folds = cv
            cv = KFold(n_splits=cv_folds, shuffle=False, random_state=random_state)
        else:
            cv_folds = cv.get_n_splits() # cv is a sklearn splitter

        if cv_folds == 1:
            vl_size = cv.test_size

        # Create sklearn splitter 
        if mltype == 'cls':
            if Y.ndim > 1 and Y.shape[1] > 1:
                splitter = cv.split(X, np.argmax(Y, axis=1))
            else:
                splitter = cv.split(X, Y)
        elif mltype == 'reg':
            splitter = cv.split(X, Y)
        
        # Generate the splits
        for i, (tr_vec, vl_vec) in enumerate(splitter):
            tr_dct[i] = tr_vec
            vl_dct[i] = vl_vec


    # Define training set sizes
    if data_sz_frac is None:
        if any([tick_spacing.lower()==s for s in ['lin', 'linear']]):
            data_sz_frac = np.linspace(0.1, 1.0, lc_ticks) # linear spacing
        elif any([tick_spacing.lower()==s for s in ['log']]):
            base = 10
            data_sz_frac = np.logspace(0.0, 1.0, lc_ticks, endpoint=True, base=base)/base # log spacing
        logger.info(f'Ticks spacing: {tick_spacing}.')

    if cv_folds == 1:
        tr_sizes = [int(n) for n in (1-vl_size) * X.shape[0] * data_sz_frac]
    elif cv_folds > 1:
        tr_sizes = [int(n) for n in (cv_folds-1)/cv_folds * X.shape[0] * data_sz_frac]

    if logger is not None:
        logger.info('Train sizes: {}\n'.format(tr_sizes))

    
    # Now start nested loop of train size and cv folds
    tr_scores_all = [] # list of dicts
    vl_scores_all = [] # list of dicts

    # CV loop
    for fold, (tr_k, vl_k) in enumerate(zip( tr_dct.keys(), vl_dct.keys() )):
        if logger is not None:
            logger.info(f'Fold {fold+1}/{cv_folds}')

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
                logger.info(f'\tTrain size: {tr_sz} ({i+1}/{len(tr_sizes)})')   

            # Sequentially get a subset of samples (the input dataset X must be shuffled)
            xtr_sub = xtr[idx[:tr_sz], :]
            ytr_sub = np.squeeze(ytr[idx[:tr_sz], :])            

            # Get the estimator
            estimator = ml_models.get_model(model_name, init_kwargs=init_kwargs)

            if 'nn' in model_name:
                plot_model(estimator.model, to_file=outdir/'nn_model.png')

                # Create output dir
                out_nn_model = outdir / ('cv'+str(fold+1) + '_sz'+str(tr_sz))
                os.makedirs(out_nn_model, exist_ok=False)
                
                # Callbacks (custom)
                clr = CyclicLR(base_lr=1e-4, max_lr=1e-3, mode='triangular')
                
                # Keras callbacks
                checkpointer = ModelCheckpoint(str(out_nn_model/'model_best.h5'), verbose=0, save_weights_only=False, save_best_only=True)
                csv_logger = CSVLogger(out_nn_model/'training.log')
                reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.75, patience=20, verbose=1, mode='auto',
                                              min_delta=0.0001, cooldown=3, min_lr=0.000000001)
                early_stop = EarlyStopping(monitor='val_loss', patience=100, verbose=1, mode='auto')
                
                # Callbacks list
                callback_list = [checkpointer, csv_logger, early_stop, reduce_lr]
                if (args is not None) and (args['opt']=='clr'):
                    callback_list = callback_list + [clr]

                # Fit params
                # TODO: which val set should be used??
                fit_kwargs['validation_data'] = (xvl, yvl)
                # fit_kwargs['validation_split'] = 0.2
                fit_kwargs['callbacks'] = callback_list

            # Train model
            history = estimator.model.fit(xtr_sub, ytr_sub, **fit_kwargs)

            # If nn, load the best model
            if 'nn' in model_name:
                model = keras.models.load_model(str(out_nn_model/'autosave.model.h5'), custom_objects={'r2_krs': r2_krs}) # https://github.com/keras-team/keras/issues/5916
            else:
                # If not NN model
                model = estimator.model

            # Calc preds and scores TODO: dump preds
            # ... training set
            y_pred, y_true = utils.calc_preds(model, x=xtr_sub, y=ytr_sub, mltype=mltype)
            tr_scores = utils.calc_scores(y_true=y_true, y_pred=y_pred, mltype=mltype, metrics=None)
            # ... val set
            y_pred, y_true = utils.calc_preds(model, x=xvl, y=yvl, mltype=mltype)
            vl_scores = utils.calc_scores(y_true=y_true, y_pred=y_pred, mltype=mltype, metrics=None)

            nm = ((y_true - y_pred) ** 2).sum(axis=0, dtype=np.float64)
            dn = ((y_true - np.average(y_true, axis=0)) ** 2).sum(axis=0, dtype=np.float64)

            if 'nn' in model_name:
                ml_models.plot_prfrm_metrics(history, title=f'Train size: {tr_sz}',
                                             skp_ep=20, add_lr=True, outdir=out_nn_model)

            # Add info
            tr_scores['tr_set'] = True
            vl_scores['tr_set'] = False
            tr_scores['fold'] = 'f'+str(fold)
            vl_scores['fold'] = 'f'+str(fold)
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
        scores_all_df_tmp.to_csv(outdir / (model_name + '_lrn_crv_scores_cv' + str(fold+1) + '.csv'), index=False)

    tr_df = scores_to_df(tr_scores_all)
    vl_df = scores_to_df(vl_scores_all)
    scores_all_df = pd.concat([tr_df, vl_df], axis=0)

    # Plot learning curves
    figs = plt_lrn_crv_multi_metric(scores_all_df, cv_folds=cv_folds, outdir=outdir, args=args)
    
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

    plt.plot(tr_sizes, tr_scores_mean, 'o-', color='r', label='Train score')
    plt.plot(tr_sizes, te_scores_mean, 'o-', color='g', label='Val score')
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

