"""
Implementation of cross-validation run.
"""
import os
from pathlib import Path
import sys
from collections import OrderedDict

import sklearn
import numpy as np
import pandas as pd

import matplotlib
# matplotlib.use('TkAgg')
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import ShuffleSplit, KFold
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

from pandas.api.types import is_string_dtype
from sklearn.preprocessing import LabelEncoder

# Utils
import utils
import utils_tidy
import ml_models
# from cv_splitter import GroupSplit, SimpleSplit, plot_ytr_yvl_dist

# Import custom callbacks
keras_contrib = '/vol/ml/apartin/projects/keras-contrib/keras_contrib/callbacks'
sys.path.append(keras_contrib)
#from callbacks import *
from cyclical_learning_rate import CyclicLR


def my_cross_validate(X, Y,
                      mltype,
                      model_name='lgb_reg',
                      cv=5,
                      lr_crv_ticks=5, data_sizes_frac=None,
                      args=None, fit_params=None, init_params=None,
                      n_jobs=1, random_state=None, logger=None, outdir='.'):
    """
    Train estimator using various train set sizes and generate learning curves for different metrics.
    The CV splitter splits the input dataset into cv_folds data subsets.
    Args:
        X : features matrix
        Y : target
        mltype : type to ML problem (`reg` or `cls`)
        cv : number cv folds or sklearn's cv splitter --> scikit-learn.org/stable/glossary.html#term-cv-splitter
        lr_crv_ticks : number of ticks in the learning curve (used if data_sizes_frac is None)
        data_sizes_frac : relative numbers of training samples that will be used to generate learning curves
        fit_params : dict of parameters to the estimator's "fit" method

        metrics : allow to pass a list of metrics  TODO!
        args : command line args

    Examples:
        cv = sklearn.model_selection.KFold(n_splits=5, shuffle=False, random_state=0)
        lrn_curve.my_learning_curve(X=xdata, Y=ydata, mltype='reg', cv=cv, lr_crv_ticks=5)
    """
    X = pd.DataFrame(X).values
    Y = pd.DataFrame(Y).values

    if isinstance(cv, int):
        cv_folds = cv
        cv = KFold(n_splits=cv_folds, shuffle=False, random_state=random_state)
    else:
        cv_folds = cv.get_n_splits()

    # ... Now start a nested loop of train size and cv folds ...
    tr_scores_all = [] # list dicts
    vl_scores_all = [] # list dicts

    if mltype == 'cls':
        if Y.ndim > 1 and Y.shape[1] > 1:
            splitter = cv.split(X, np.argmax(Y, axis=1))
        else:
            splitter = cv.split(X, Y)
    elif mltype == 'reg':
        splitter = cv.split(X, Y)

    # Placeholder to save the best model
    best_model = None
    best_score = -np.Inf

    # Start CV iters
    for fold_id, (tr_idx, vl_idx) in enumerate(splitter):
        if logger is not None:
            logger.info(f'Fold {fold_id+1}/{cv_folds}')

        # Samples from this dataset are sampled for training
        xtr = X[tr_idx, :]
        ytr = np.squeeze(Y[tr_idx, :])

        # A fixed set of validation samples for the current CV split
        xvl = X[vl_idx, :]
        yvl = np.squeeze(Y[vl_idx, :])        

        # Get the estimator
        estimator = ml_models.get_model(model_name, init_params)

        if 'nn' in model_name:
            from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard

            # Create output dir
            out_nn_model = outdir / ('cv'+str(fold_id+1))
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
            callback_list = [checkpointer, csv_logger, early_stop, reduce_lr]
            if (args is not None) and (args['opt']=='clr'):
                callback_list = callback_list + [clr]
                # TODO: which val set should be used??
                # fit_params['validation_data'] = (xvl, yvl)
                # fit_params['validation_split'] = 0.2

            # Fit params
            fit_params['validation_data'] = (xvl, yvl)
            fit_params['callbacks'] = callback_list

        # Train model
        history = estimator.model.fit(xtr, ytr, **fit_params)
    
        # Dump model
        estimator.dump_model(out_nn_model)

        # Calc preds and scores TODO: dump preds
        # ... training set
        y_preds, y_true = utils.calc_preds(estimator.model, x=xtr, y=ytr, mltype=mltype)
        tr_scores = utils.calc_scores(y_true=y_true, y_preds=y_preds, mltype=mltype)
        # ... val set
        y_preds, y_true = utils.calc_preds(estimator.model, x=xvl, y=yvl, mltype=mltype)
        vl_scores = utils.calc_scores(y_true=y_true, y_preds=y_preds, mltype=mltype)

        # Save the best model
        if mltype == 'cls':
            vl_scores['f1_score'] > best_score
            best_score = vl_scores['f1_score']
            best_model = estimator
        elif mltype == 'reg':
            vl_scores['r2'] > best_score
            best_score = vl_scores['r2']
            best_model = estimator

        # Plot training curves
        if 'nn' in model_name:
            ml_models.plot_prfrm_metrics(history, title=f'cv fold: {fold_id+1}',
                                         skp_ep=7, add_lr=True, outdir=out_nn_model)

        # Add info
        tr_scores['tr_set'] = True
        vl_scores['tr_set'] = False
        tr_scores['fold'] = 'f'+str(fold_id)
        vl_scores['fold'] = 'f'+str(fold_id)

        # Aggregate scores
        tr_scores_all.append(tr_scores)
        vl_scores_all.append(vl_scores)

        # Delete the estimator/model
        del estimator, history

            
    tr_df = scores_to_df(tr_scores_all)
    vl_df = scores_to_df(vl_scores_all)
    scores_all_df = pd.concat([tr_df, vl_df], axis=0)

    return scores_all_df, best_model


def scores_to_df(scores_all):
    df = pd.DataFrame(scores_all)
    df = df.melt(id_vars=['fold', 'tr_set'])
    df = df.rename(columns={'variable': 'metric'})
    df = df.pivot_table(index=['metric', 'tr_set'], columns=['fold'], values='value')
    df = df.reset_index(drop=False)
    df.columns.name = None
    return df


