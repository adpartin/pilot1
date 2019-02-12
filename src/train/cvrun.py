"""
Implementation of cv run.
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
from sklearn.model_selection import ShuffleSplit, KFold
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

from pandas.api.types import is_string_dtype
from sklearn.preprocessing import LabelEncoder

import utils
import utils_tidy
import ml_models
from cvsplitter import GroupSplit, SimpleSplit, plot_ytr_yvl_dist


def my_cross_validate(X, Y,
                      mltype,
                      model_name='lgb_reg',
                      cv=5, groups=None,
                      lr_curve_ticks=5, data_sizes_frac=None,
                      args=None, fit_params=None, init_params=None,
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

        metrics : allow to pass a list of metrics  TODO!
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

        # # Confirm that group splits are correct ...
        # tr_grps_unq = set(groups[tr_idx])
        # vl_grps_unq = set(groups[vl_idx])
        # print('Total group (e.g., cell) intersections btw tr and vl: ', len(tr_grps_unq.intersection(vl_grps_unq)))
        # print('A few intersections : ', list(tr_grps_unq.intersection(vl_grps_unq))[:3])

        # Get the estimator
        estimator = ml_models.get_model(model_name=model_name, init_params=init_params)

        if 'nn' in model_name:
            from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard
            from keras.utils import plot_model

            # Create output dir
            out_nn_model = os.path.join(outdir, 'cv'+str(fold_id+1))
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
            fit_params['validation_data'] = (xvl, yvl)

        # Train model
        history = estimator.model.fit(xtr, ytr, **fit_params)

        # Calc preds and scores TODO: dump preds
        # ... training set
        y_preds, y_true = utils.calc_preds(estimator=estimator.model, xdata=xtr, ydata=ytr, mltype=mltype)
        tr_scores = utils.calc_scores(y_true=y_true, y_preds=y_preds, mltype=mltype)
        # ... val set
        y_preds, y_true = utils.calc_preds(estimator=estimator.model, xdata=xvl, ydata=yvl, mltype=mltype)
        vl_scores = utils.calc_scores(y_true=y_true, y_preds=y_preds, mltype=mltype)

        # Save the best model
        if mltype == 'cls':
            vl_scores['f1_score'] > best_score
            best_model = estimator
        elif mltype == 'reg':
            vl_scores['r2'] > best_score
            best_model = estimator

        # Plot training curves
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
                plt.title(f'cv fold: {fold_id+1}')
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



def my_cv_run(data, target_name,
              model, # model_name,
              fea_prfx_dict,
              cv_method, cv_folds, logger,
              verbose=False, random_state=None, outdir=None):
    """ Run cross-validation.
    Args:
        data : tidy dataset
        model_name : model name available ml_models.py
        cv_method : cv splitting method
        cv_folds : number cv folds
        logger : 
    """
    data = data.copy().reset_index(drop=True)

    # Split tr/vl data
    if cv_method=='simple':
        splitter = SimpleSplit(n_splits=cv_folds, random_state=random_state)
        splitter.split(X=data)
    elif cv_method=='group':
        splitter = GroupSplit(n_splits=cv_folds, random_state=random_state)
        splitter.split(X=data, groups=data['CELL'])
    elif cv_method=='stratify':
        pass
    else:
        raise ValueError(f'This cv_method ({cv_method}) is not supported.')

    # Model
    # model, _ = init_model(model_name, logger)

    # Run CV training
    # if verbose:
    #     logger.info(f'CV split method: {cv_method}')
    #     logger.info(f'ML model: {model_name}')
    tr_scores = []
    vl_scores = []
    for i in range(splitter.n_splits):
        estimator = sklearn.base.clone(model)

        if verbose:
            logger.info(f'Fold {i+1}/{splitter.n_splits}')
        tr_idx = splitter.tr_cv_idx[i]
        vl_idx = splitter.vl_cv_idx[i]
        tr_data = data.iloc[tr_idx, :]
        vl_data = data.iloc[vl_idx, :]

        # Confirm that group splits are correct ...
        # tr_cells = set(tr_data['CELL'].values)
        # vl_cells = set(vl_data['CELL'].values)
        # print('total cell intersections btw tr and vl: ', len(tr_cells.intersection(vl_cells)))
        # print('a few intersections : ', list(tr_cells.intersection(vl_cells))[:3])

        xtr, _ = utils_tidy.split_features_and_other_cols(tr_data, fea_prfx_dict=fea_prfx_dict)
        xvl, _ = utils_tidy.split_features_and_other_cols(vl_data, fea_prfx_dict=fea_prfx_dict)

        # utils_tidy.print_feature_shapes(df=xtr, logger=logger)
        # utils_tidy.print_feature_shapes(df=xvl, logger=logger)

        ytr = utils_tidy.extract_target(data=tr_data, target_name=target_name)
        yvl = utils_tidy.extract_target(data=vl_data, target_name=target_name)

        if outdir:
            title = f'{target_name}; split {str(i)}'
            plot_ytr_yvl_dist(ytr, yvl, title=title, outpath=os.path.join(outdir, title+'.png'))

        # Train
        #model, _ = init_model(model_name, logger)
        #model.fit(xtr, ytr, eval_set=[(xtr, ytr), (xvl, yvl)])
        estimator.fit(xtr, ytr, eval_set=[(xtr, ytr), (xvl, yvl)], verbose=False)

        # Combine scores from all cv folds
        ##tr_scores.append(model.calc_scores(xdata=xtr, ydata=ytr, to_print=False))
        ##vl_scores.append(model.calc_scores(xdata=xvl, ydata=yvl, to_print=False))
        # tr_scores.append(utils.calc_scores(model=estimator, xdata=xtr, ydata=ytr))
        # vl_scores.append(utils.calc_scores(model=estimator, xdata=xvl, ydata=yvl))
        tr_scores.append(calc_scores(model=estimator, xdata=xtr, ydata=ytr))
        vl_scores.append(calc_scores(model=estimator, xdata=xvl, ydata=yvl))

    # Summarize cv scores
    tr_cv_scores = utils.cv_scores_to_df(tr_scores, calc_stats=True)
    vl_cv_scores = utils.cv_scores_to_df(vl_scores, calc_stats=True)
    # print('\ntr scores\n{}'.format(tr_cv_scores))
    # print('\nvl scores\n{}'.format(vl_cv_scores))

    if outdir:
        tr_cv_scores.to_csv(os.path.join(outdir, 'tr_cv_scores.csv'), index=False)
        vl_cv_scores.to_csv(os.path.join(outdir, 'vl_cv_scores.csv'), index=False)

    return tr_cv_scores, vl_cv_scores


def adj_r2_score(ydata, preds, x_size):
    """ Calc adjusted r^2.
    https://en.wikipedia.org/wiki/Coefficient_of_determination#Adjusted_R2
    https://dziganto.github.io/data%20science/linear%20regression/machine%20learning/python/Linear-Regression-101-Metrics/
    https://stats.stackexchange.com/questions/334004/can-r2-be-greater-than-1
    """
    r2 = sklearn.metrics.r2_score(ydata, preds)
    adj_r2 = 1 - (1 - r2) * (x_size[0] - 1)/(x_size[0] - x_size[1] - 1)
    return adj_r2


def calc_scores(model, xdata, ydata):
    """ Create dict of scores. """
    # TODO: replace `if` with `try`
    preds = model.predict(xdata)
    scores = OrderedDict()

    scores['r2'] = sklearn.metrics.r2_score(ydata, preds)
    #scores['adj_r2_score'] = self.__adj_r2_score(ydata, preds)
    scores['mean_absolute_error'] = sklearn.metrics.mean_absolute_error(ydata, preds)
    scores['median_absolute_error'] = sklearn.metrics.median_absolute_error(ydata, preds)
    scores['mean_squared_error'] = sklearn.metrics.mean_squared_error(ydata, preds)

    return scores