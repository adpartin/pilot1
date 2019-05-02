"""
Util functions to split data into train/val.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
# matplotlib.use('TkAgg')
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pandas.api.types import is_string_dtype
from collections import OrderedDict
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import ShuffleSplit, KFold
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold


def cv_splitter(cv_method: str='simple', cv_folds: int=1, test_size: float=0.2,
                mltype: str='reg', shuffle: bool=True, random_state=None):
    """ Creates a cross-validation splitter.
    Args:
        cv_method: 'simple', 'stratify' (only for classification), 'groups' (only for regression)
        cv_folds: number of cv folds
        test_size: fraction of test set size (used only if cv_folds=1)
        mltype: 'reg', 'cls'
    """
    # Classification
    if mltype == 'cls':
        if cv_method == 'simple':
            if cv_folds == 1:
                cv = ShuffleSplit(n_splits=cv_folds, test_size=test_size, random_state=random_state)
            else:
                cv = KFold(n_splits=cv_folds, shuffle=shuffle, random_state=random_state)
            
        elif cv_method == 'stratify':
            if cv_folds == 1:
                cv = StratifiedShuffleSplit(n_splits=cv_folds, test_size=test_size, random_state=random_state)
            else:
                cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    # Regression
    elif mltype == 'reg':
        # Regression
        if cv_method == 'group':
            if cv_folds == 1:
                cv = GroupShuffleSplit(n_splits=cv_folds, random_state=random_state)
            else:
                cv = GroupKFold(n_splits=cv_folds)
            
        elif cv_method == 'simple':
            if cv_folds == 1:
                cv = ShuffleSplit(n_splits=cv_folds, test_size=test_size, random_state=random_state)
            else:
                cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    return cv


def plot_ytr_yvl_dist(ytr, yvl, title=None, outpath='.'):
    """ Plot distributions of response data of train and val sets. """
    fig, ax = plt.subplots()
    plt.hist(ytr, bins=100, label='ytr', color='b', alpha=0.5)
    plt.hist(yvl, bins=100, label='yvl', color='r', alpha=0.5)
    if title is None:
        title = ''
    plt.title(title)
    plt.tight_layout()
    plt.grid(True)
    plt.legend()
    # plt.savefig(os.path.join(outpath, title+'_ytr_yvl_dist.png'), bbox_inches='tight')
    if outpath is None:
        plt.savefig(Path(outpath)/'ytr_yvl_dist.png', bbox_inches='tight')
    else:
        plt.savefig(outpath, bbox_inches='tight')


class SimpleSplit():
    """ Split data using KFold or ShuffleSplit.
    This class supports both single split (i.e. single tr/vl split) or multiple splits via kfold splits.

    Example:
        splitter = SimpleSplit(n_splits=5, random_state=SEED)
        splitter.split(X=data)
    """
    def __init__(self, n_splits=5, test_size=0.2, random_state=None):
        self.random_state = random_state
        self.n_splits = n_splits

        if n_splits == 1:
            self.test_size = test_size
            self.cv_splitter = ShuffleSplit(n_splits=self.n_splits,
                                            test_size=self.test_size,
                                            random_state=self.random_state)
        elif n_splits > 1:
            self.cv_splitter = KFold(n_splits=self.n_splits,
                                     shuffle=False,
                                     random_state=self.random_state)


    def split(self, X, y=None):
        self.tr_cv_idx = OrderedDict()
        self.vl_cv_idx = OrderedDict()
        for i, (tr_idx, vl_idx) in enumerate(self.cv_splitter.split(X, y)):
            self.tr_cv_idx[i] = tr_idx
            self.vl_cv_idx[i] = vl_idx


    def summary(self):
        for i in range(self.n_splits):
            tr_idx = self.tr_cv_idx[i]
            vl_idx = self.vl_cv_idx[i]
            print(f'\nSplit {i}')


class GroupSplit():
    """ Split data such that groups don't overlap across train and val sets.
    This class supports both single split (i.e. single tr/vl split) or multiple splits via kfold splits.

    Example:
        splitter = GroupSplit(n_splits=5, random_state=SEED)
        splitter.split(X=data, groups=data['CELL'])
    """
    def __init__(self, n_splits=5, test_size=0.2, random_state=None):
        self.random_state = random_state
        self.n_splits = n_splits

        if n_splits == 1:
            self.test_size = test_size
            self.cv_splitter = GroupShuffleSplit(n_splits=self.n_splits,
                                                 test_size=self.test_size,
                                                 random_state=self.random_state)
        elif n_splits > 1:
            self.cv_splitter = GroupKFold(n_splits=n_splits)


    def split(self, X, groups, y=None):
        self.groups = groups

        if is_string_dtype(self.groups):  # self.groups=='object'
            self.group_encoder = LabelEncoder()
            self.groups = self.group_encoder.fit_transform(self.groups)

        self.tr_cv_idx = OrderedDict()
        self.vl_cv_idx = OrderedDict()
        for i, (tr_idx, vl_idx) in enumerate(self.cv_splitter.split(X, y, self.groups)):
            self.tr_cv_idx[i] = tr_idx
            self.vl_cv_idx[i] = vl_idx


    def summary(self):
        for i in range(self.n_splits):
            tr_idx = self.tr_cv_idx[i]
            vl_idx = self.vl_cv_idx[i]
            print(f'\nSplit {i}')
            print(f'Train size {len(self.groups[tr_idx])}, unique groups {len(np.unique(self.groups[tr_idx]))}')
            print(f'Val size   {len(self.groups[vl_idx])}, unique groups {len(np.unique(self.groups[vl_idx]))}')



# class StratifiedSplit():
#     def __init__(kfolds=1, test_size=0.2, random_state=None):
#         self.random_state = random_state
#         self.kfolds = kfolds
#         if kfolds == 1:
#             self.test_size = test_size


def plot_ytr_yvl_dist(ytr, yvl, title=None, outpath=None):
    """ Plot distributions of response data of train and val sets. """
    fig, ax = plt.subplots()
    plt.hist(ytr, bins=100, label='ytr', color='b', alpha=0.5)
    plt.hist(yvl, bins=100, label='yvl', color='r', alpha=0.5)
    if title is None:
        title = ''
    plt.title(title)
    plt.tight_layout()
    plt.grid(True)
    plt.legend()
    # plt.savefig(os.path.join(outpath, title+'_ytr_yvl_dist.png'), bbox_inches='tight')
    if outpath is None:
        plt.savefig('ytr_yvl_dist.png', bbox_inches='tight')
    else:
        plt.savefig(outpath, bbox_inches='tight')


