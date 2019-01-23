"""
Implementation of cv run.
"""
import os
import numpy as np
import pandas as pd

import sklearn

import utils
import utils_tidy
from cvsplitter import GroupSplit, SimpleSplit, plot_ytr_yvl_dist


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
        tr_scores.append(utils.calc_scores(model=estimator, xdata=xtr, ydata=ytr))
        vl_scores.append(utils.calc_scores(model=estimator, xdata=xvl, ydata=yvl))

    # Summarize cv scores
    tr_cv_scores = utils.cv_scores_to_df(tr_scores, calc_stats=True)
    vl_cv_scores = utils.cv_scores_to_df(vl_scores, calc_stats=True)
    # print('\ntr scores\n{}'.format(tr_cv_scores))
    # print('\nvl scores\n{}'.format(vl_cv_scores))

    if outdir:
        tr_cv_scores.to_csv(os.path.join(outdir, 'tr_cv_scores.csv'), index=False)
        vl_cv_scores.to_csv(os.path.join(outdir, 'vl_cv_scores.csv'), index=False)

    return tr_cv_scores, vl_cv_scores



