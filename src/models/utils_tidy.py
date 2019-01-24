"""
This script contains functions that work with the tidy dataframe.
"""
import os
import logging
import numpy as np
import pandas as pd

import matplotlib
# matplotlib.use('TkAgg')
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import seaborn as sns
from scipy import stats
from sklearn.metrics import r2_score, mean_absolute_error, median_absolute_error, explained_variance_score

DATADIR = '/Users/apartin/work/jdacs/Benchmarks/Data/Pilot1'


def split_features_and_other_cols(data, fea_prfx_dict):
    """ Extract two dfs from `data`: fea_data and other_data.
    TODO: this script is also in src/data/utils_data (put in a single place)
    Args:
        data : tidy dataset (df contains multiple cols including features, meta, and target)
    Returns:
        fea_data : contains only training features
        other_data : contains other cols (meta, target)
    """
    # Extract df that contains only features (no meta or response)
    other_data = data.copy()
    df_fea_list = []

    for prfx in fea_prfx_dict.values():

        # get cols with specific feature prfx
        cols = data.columns[[True if prfx in c else False for c in data.columns.tolist()]]

        # if feature present in data, add it to df_fea_list, and drop from other_data
        if len(cols) > 0:  
            df = data[cols].copy()
            other_data.drop(columns=cols, inplace=True)
            df_fea_list.append(df)

    fea_data = pd.DataFrame(pd.concat(df_fea_list, axis=1))
    return fea_data, other_data


def extract_subset_features(data, feature_list, fea_prfx_dict):
    """ Extract a subset of features for training.
    Args:
        data : tidy dataset (df contains multiple cols including features, meta, and target)
        feature_list : e.g., (cell_features + drug_features)
        fea_prfx_dict : dict of feature prefixes, e.g.:
            fea_prfx_dict = {'rna': 'cell_rna.', 'cnv': 'cell_cnv.', 'dsc': 'drug_dsc.', 'fng': 'drug_fng.'}
    Returns:
        data : updated data
    """
    fea_data, other_data = split_features_and_other_cols(data, fea_prfx_dict)
    fea_prfx_list = [fea_prfx_dict[fea] for fea in feature_list if fea in fea_prfx_dict.keys()]
    fea_data = fea_data[[c for c in fea_data.columns if (c.split('.')[0]+'.') in fea_prfx_list]].reset_index(drop=True).copy()

    # Concat feature set (fea_data) and other cols (other_data)
    data = pd.concat([other_data, fea_data], axis=1)
    return data


def impute_values(data, fea_prfx_dict, logger=None):
    """ Impute missing values.
    TODO: this script is also in src/data/utils_data (put in a single place)
    Args:
        data : tidy dataset (df contains multiple cols including features, meta, and target)
        fea_prfx_dict : dict of feature prefixes, e.g.:
            fea_prfx_dict = {'rna': 'cell_rna.', 'cnv': 'cell_cnv.', 'dsc': 'drug_dsc.', 'fng': 'drug_fng.'}
        logger : logging object
    TODO: consider more advanced imputation methods:
    - https://www.rdocumentation.org/packages/Amelia/versions/1.7.4/topics/amelia
    - try regressor (impute continuous features) or classifier (impute discrete features)
    """
    from sklearn.impute import SimpleImputer, MissingIndicator
    data = data.copy()
    logger.info('\nImpute missing features ... ({})'.format(list(fea_prfx_dict.keys())))

    # Extract df that contains only features (no meta or response)
    fea_data, other_data = split_features_and_other_cols(data=data, fea_prfx_dict=fea_prfx_dict)
    tot_miss_feas = sum(fea_data.isna().sum(axis=0) > 0)
    logger.info('Total features with missing values (before imputation): {}'.format(tot_miss_feas))

    if tot_miss_feas > 0:
        colnames = fea_data.columns

        # Impute missing values
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean', verbose=1)
        dtypes_dict = fea_data.dtypes # keep the original dtypes because fit_transform casts to np.float64
        fea_data_imputed = imputer.fit_transform(fea_data)
        fea_data_imputed = pd.DataFrame(fea_data_imputed, columns=colnames)
        fea_data_imputed = fea_data_imputed.astype(dtypes_dict) # cast back to the original data type

        logger.info('Total features with missing values (after impute): {}'.format(sum(fea_data_imputed.isna().sum(axis=0) > 0)))

        # Concat features (xdata_imputed) and other cols (other_data)
        data = pd.concat([other_data, fea_data_imputed], axis=1)
        
    return data


def extract_target(data, target_name):
    """ Extract vector of target values.
    Args:
        data : tidy dataset (df contains multiple cols including features, meta, and target)
    """
    y = data[target_name].copy()
    return y


def print_feature_shapes(df, logger):
    """ Print features shapes.
    Each feature name starts with a prefix indicating the feature type (`rna.`, `dsc.`, etc).
    Args:
        df : dataframe with feature columns
    """
    # logger.info(f'\n{name}')
    for prefx in np.unique(list(map(lambda x: x.split('.')[0], df.columns.tolist()))):
        cols = df.columns[[True if prefx in c else False for c in df.columns.tolist()]]
        logger.info('{}: {}'.format(prefx, df[cols].shape))


