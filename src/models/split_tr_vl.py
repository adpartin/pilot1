"""
Util functions to split data into train/val.
"""
import os
import numpy as np
import pandas as pd

import matplotlib
# matplotlib.use('TkAgg')
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pandas.api.types import is_string_dtype
from collections import OrderedDict
from sklearn.preprocessing import LabelEncoder


class GroupSplit():
    """ Split data such that groups do not overlap across train and val sets.
    This class supports both single split (i.e. single tr/vl split) or multiple splits via kfold splits.

    Example:
        splitter = GroupSplit(n_splits=5, random_state=SEED)
        splitter.split(X=data, groups=data['CELL'])
    """
    def __init__(self, n_splits=5, test_size=0.2, random_state=None):
        from sklearn.model_selection import GroupShuffleSplit, GroupKFold
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



class SimpleSplit():
    """ Split data using KFold or ShuffleSplit.

    Example:
        splitter = SimpleSplit(n_splits=5, random_state=SEED)
        splitter.split(X=data)
    """
    def __init__(self, n_splits=5, test_size=0.2, random_state=None):
        from sklearn.model_selection import ShuffleSplit, KFold
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


