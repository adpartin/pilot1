"""
This script contains functions that work with the tidy dataframe.
"""
import os
import sys
import logging
import numpy as np
import pandas as pd
import re

import matplotlib
# matplotlib.use('TkAgg')
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import seaborn as sns
from scipy import stats
from sklearn.metrics import r2_score, mean_absolute_error, median_absolute_error, explained_variance_score

import utils
import utils_tidy
import classlogger


DATADIR = '/Users/apartin/work/jdacs/Benchmarks/Data/Pilot1'


def load_data(datapath, fea_prfx_dict, args, logger, random_state=None):
    """ Load and pre-process the tidy data. """
    logger.info(f'\nLoad tidy data ... {datapath}')
    data = pd.read_parquet(datapath, engine='auto', columns=None)
    logger.info(f'data.shape {data.shape}')
    logger.info('data memory usage: {:.3f} GB'.format(sys.getsizeof(data)/1e9))
    # print(data.groupby('SOURCE').agg({'CELL': 'nunique', 'DRUG': 'nunique', 'PUBCHEM': 'nunique'}).reset_index())


    # Replace characters that are illegal for xgboost/lightgbm feature names
    # xdata.columns = list(map(lambda s: s.replace('[','_').replace(']','_'), xdata.columns.tolist())) # required by xgboost
    regex = re.compile(r'\[|\]|<', re.IGNORECASE)
    data.columns = [regex.sub('_', c) if any(x in str(c) for x in set(('[', ']', '<'))) else c for c in data.columns.values]


    if args['tissue_type'] is not None:
        data = data[data[''].isin([args['tissue_type']])].reset_index(drop=True)


    # Subsample
    if args['row_sample']:
        row_sample = eval(args['row_sample'])
        data = utils.subsample(df=data, v=row_sample, axis=0)
        print('data.shape', data.shape)

    if args['col_sample']:
        col_sample = eval(args['col_sample'])
        fea_data, other_data = utils_tidy.split_features_and_other_cols(data, fea_prfx_dict=fea_prfx_dict)
        fea_data = utils.subsample(df=fea_data, v=col_sample, axis=1)
        data = pd.concat([fea_data, other_data], axis=1)
        print('data.shape', data.shape)


    # Extract test sources
    logger.info('\nExtract test sources ... {}'.format(args['test_sources']))
    te_data = data[data['SOURCE'].isin(args['test_sources'])].reset_index(drop=True)
    logger.info(f'te_data.shape {te_data.shape}')
    logger.info('data memory usage: {:.3f} GB'.format(sys.getsizeof(te_data)/1e9))
    logger.info(te_data.groupby('SOURCE').agg({'CELL': 'nunique', 'DRUG': 'nunique'}).reset_index())


    # Extract train sources
    logger.info('\nExtract train sources ... {}'.format(args['train_sources']))
    data = data[data['SOURCE'].isin(args['train_sources'])].reset_index(drop=True)
    logger.info(f'data.shape {data.shape}')
    logger.info('data memory usage: {:.3f} GB'.format(sys.getsizeof(data)/1e9))
    logger.info(data.groupby('SOURCE').agg({'CELL': 'nunique', 'DRUG': 'nunique'}).reset_index())


    # Assign type to categoricals
    # cat_cols = data.select_dtypes(include='object').columns.tolist()
    # data[cat_cols] = data[cat_cols].astype('category', ordered=False)


    # Shuffle data
    data = data.sample(frac=1.0, axis=0, random_state=random_state).reset_index(drop=True)


    # Filter out AUC>1
    # print('\nFilter some AUC outliers (>1)')
    # print('data.shape', data.shape)
    # data = data[[False if x>1.0 else True for x in data[target_name]]].reset_index(drop=True)
    # print('data.shape', data.shape)


    # Transform the target
    if args['target_trasform']:
        y = data[args['target_name']].copy()
        # y = np.log1p(ydata); plot_hist(x=y, var_name=target_name+'_log1p')
        # # y = np.log(ydata+1); plot_hist(x=y, var_name=target_name+'_log+1')
        # y = np.log10(ydata+1); plot_hist(x=y, var_name=target_name+'_log10')
        # y = np.log2(ydata+1); plot_hist(x=y, var_name=target_name+'_log2')
        # y = ydata**2; plot_hist(x=ydata, var_name=target_name+'_x^2')
        y, lmbda = stats.boxcox(y+1); # utils.plot_hist(x=y, var_name=target_name+'_boxcox', path=)
        data[args['target_name']] = y
        # ydata = pd.DataFrame(y)

        y = te_data[args['target_name']].copy()
        y, lmbda = stats.boxcox(y+1); # utils.plot_hist(x=y, var_name=target_name+'_boxcox', path=)
        te_data[args['target_name']] = y


    if 'dlb' in args['other_features']:
        logger.info('\nAdd drug labels to features ...')
        # print(data['DRUG'].value_counts())

        # http://queirozf.com/entries/one-hot-encoding-a-feature-on-a-pandas-dataframe-an-example
        # One-hot encoder
        dlb = pd.get_dummies(data=data[['DRUG']], prefix=fea_prfx_dict['dlb'],
                            dummy_na=False).reset_index(drop=True)

        # Label encoder
        # dlb = data[['DRUG']].astype('category', ordered=False).reset_index(drop=True)
        # print(dlb.dtype)

        # Concat drug labels and other features
        data = pd.concat([dlb, data], axis=1).reset_index(drop=True)
        logger.info(f'dlb.shape {dlb.shape}')
        logger.info(f'data.shape {data.shape}')


    if 'rna_clusters' in args['other_features']:
        # TODO
        pass

    return data, te_data


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


