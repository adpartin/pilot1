"""
Functions that work with the tidy dataframe.
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
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# github.com/pandas-dev/pandas/blob/v0.24.2/pandas/core/dtypes/common.py
from pandas.api.types import is_string_dtype, is_numeric_dtype, is_categorical_dtype #, ensure_categorical, ensure_float

import utils


# DATADIR = '/Users/apartin/work/jdacs/Benchmarks/Data/Pilot1'

fea_prfx_dict = {'rna': 'cell_rna.', 'cnv': 'cell_cnv.',
                 'dsc': 'drug_dsc.', 'fng': 'drug_fng.',
                 'clb': 'cell_lbl.', 'dlb': 'drug_lbl.'}


class DrugSensDataset():
    def __init__(self):
        self.fea_prfx_dict = {}
        self.data = None

        
    def extract_subset_fea(self):
        """ Extract a subset of features for training.
        Args:
            data : tidy dataset (df contains multiple cols including features, meta, and target)
            fea_list : e.g., (cell_features + drug_features)
            fea_prfx_dict : dict of feature prefixes, e.g.:
                fea_prfx_dict = {'rna': 'cell_rna.', 'cnv': 'cell_cnv.', 'dsc': 'drug_dsc.', 'fng': 'drug_fng.'}
        Returns:
            data : updated data
        """
        fea_data, other_data = self.split_fea_and_other_cols(self.data)
        fea_prfx_list = [self.fea_prfx_dict[fea] for fea in self.fea_list if fea in self.fea_prfx_dict.keys()]
        fea_sep = '.'
        fea = [c for c in fea_data.columns if (c.split(fea_sep)[0] + fea_sep) in fea_prfx_list]
        fea_data = fea_data[fea].reset_index(drop=True)

        # Concat feature set (fea_data) and other cols (other_data)
        self.data = pd.concat([other_data, fea_data], axis=1)        
        

    def split_fea_and_other_cols(self, df):
        """ Extract two dfs from `data`: fea_data and other_data.
        TODO: this script is also in src/data/utils_data (put in a single place)
        Args:
            data : tidy dataset (df contains multiple cols including features, meta, and target)
        Returns:
            fea_data : contains only training features
            other_data : contains other cols (meta, target)
        """
        # Extract df that contains only features (no meta or response)
        other_data = df.copy()
        df_fea_list = []

        for prfx in self.fea_prfx_dict.values():
            # get cols with specific feature prfx
            cols = df.columns[[True if prfx in c else False for c in df.columns.tolist()]]

            # if feature present in data, add it to df_fea_list, and drop from other_data
            if len(cols) > 0:  
                other_data.drop(columns=cols, inplace=True)
                df_fea_list.append( df[cols] )

        fea_data = pd.DataFrame( pd.concat(df_fea_list, axis=1) )
        return fea_data, other_data


    def get_num_and_cat_cols(self, df):
        """ Returns 2 dfs. One with numerical cols and the other with categorical cols.
        TODO: this doesn't have to be within the class.
        """
        cat_cols = [x for x in df.columns if is_string_dtype(df[x]) is True]
        cat_df = df[cat_cols]
        num_df = df.drop(columns=cat_cols)
        return num_df, cat_df    
    
    
    def impute_values(logger=None):
        """ Impute missing values.
        TODO: this script is also in src/data/utils_data (put in a single place)
        Args:
            data : tidy dataset (df contains multiple cols including features, meta, and target)
            fea_prfx_dict : dict of feature prefixes, e.g.:
                fea_prfx_dict = {'rna': 'cell_rna.', 'cnv': 'cell_cnv.', 'dsc': 'drug_dsc.', 'fng': 'drug_fng.'}
            logger : logging object
        TODO: consider more advanced imputation methods:
        - www.rdocumentation.org/packages/Amelia/versions/1.7.4/topics/amelia
        - try regressor (impute continuous features) or classifier (impute discrete features)
        """
        from sklearn.impute import SimpleImputer, MissingIndicator
        data = self.data.copy()
        if logger is not None:
            logger.info('\nImpute missing features ... ({})'.format( list(self.fea_prfx_dict.keys())) )

        # Extract df that contains only features (no meta or response)
        fea_data, other_data = self.split_fea_and_other_cols()
        tot_miss_feas = sum(fea_data.isna().sum(axis=0) > 0)
        if logger is not None:
            logger.info('Total features with missing values (before impute): {}'.format(tot_miss_feas))

        if tot_miss_feas > 0:
            # Split numerical from other features (only numerical will be imputed;
            # The other features can be cell and drug labels)
            fea_data, non_num_data = self.get_num_and_cat_cols(fea_data)

            # Proceed with numerical featues
            colnames = fea_data.columns

            # Impute missing values
            imputer = SimpleImputer(missing_values=np.nan, strategy='mean', verbose=1)
            dtypes_dict = fea_data.dtypes # keep the original dtypes because fit_transform casts to np.float64
            fea_data_imputed = imputer.fit_transform(fea_data)
            fea_data_imputed = pd.DataFrame(fea_data_imputed, columns=colnames)
            fea_data_imputed = fea_data_imputed.astype(dtypes_dict) # cast back to the original data type
            
            if logger is not None:
                logger.info('Total features with missing values (after impute): {}'.format( sum(fea_data_imputed.isna().sum(axis=0) > 0)) )

            # Concat features (xdata_imputed) and other cols (other_data)
            data = pd.concat([other_data, non_num_data, fea_data_imputed], axis=1)
            
        return data


    def extract_target(data, target):
        """ Extract vector of target values.
        Args:
            data : tidy dataset (df contains multiple cols including features, meta, and target)
        """
        return self.data[target].copy()


    def make_colnames_gbm_compatible(self):
        """ Replace characters that are illegal for xgboost/lightgbm feature names. """
        # xdata.columns = list(map(lambda s: s.replace('[','_').replace(']','_'), xdata.columns.tolist())) # required by xgboost
        regex = re.compile(r'\[|\]|<', re.IGNORECASE)
        self.data.columns = [regex.sub('_', c) if any(x in str(c) for x in set(('[', ']', '<'))) else c for c in self.data.columns.values]


    def target_transform(self):
        "TODO: didn't test"
        # Transform the target
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.boxcox.html
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.inv_boxcox.html
        # https://otexts.com/fpp2/transformations.html
        # y = self.data[args['target_name']].copy()
        # y = np.log1p(ydata); plot_hist(x=y, var_name=target_name+'_log1p')
        # y = np.log(ydata+1); plot_hist(x=y, var_name=target_name+'_log+1')
        # y = np.log10(ydata+1); plot_hist(x=y, var_name=target_name+'_log10')
        # y = np.log2(ydata+1); plot_hist(x=y, var_name=target_name+'_log2')
        # y = ydata**2; plot_hist(x=ydata, var_name=target_name+'_x^2')
        y, lmbda = stats.boxcox(y+1); # utils.plot_hist(x=y, var_name=target_name+'_boxcox', path=)
        ydata = pd.DataFrame(y)
        return ydata



class TidyFromCombined(DrugSensDataset):
    def __init__(self, datapath, fea_list, args, shuffle=True, logger=None, random_state=None):
        # Feature prefix (some already present in the tidy dataframe)
        self.fea_prfx_dict = {'rna': 'cell_rna.', 'cnv': 'cell_cnv.',
                              'dsc': 'drug_dsc.', 'fng': 'drug_fng.',
                              'clb': 'cell_lbl.', 'dlb': 'drug_lbl.'}
        self.datapath = datapath
        self.fea_list = fea_list
        self.args = args
        self.logger = logger
        self.random_state = random_state
        
        if self.logger:
            self.logger.info(f'\nLoad tidy data from ... \n{self.datapath}')
        self.data = pd.read_parquet(self.datapath, engine='auto', columns=None)
        if self.logger:
            self.logger.info(f'data.shape {self.data.shape}')
            self.logger.info('data memory usage: {:.3f} GB'.format(sys.getsizeof(self.data)/1e9))
        # print(data.groupby('SOURCE').agg({'CELL': 'nunique', 'DRUG': 'nunique', 'PUBCHEM': 'nunique'}).reset_index())

        # Replace characters that are illegal for xgboost/lightgbm feature names
        self.make_colnames_gbm_compatible()
        
        # Keep subset of features
        self.extract_subset_fea()

        # Shuffle
        if shuffle:
            self.data = self.data.sample(frac=1.0, axis=0, random_state=self.random_state).reset_index(drop=True)        
        
        if args['tissue_type']:
            # never tested!
            self.data = self.data[self.data[''].isin([args['tissue_type']])].reset_index(drop=True)

        # Subsample
        if args['row_sample']:
            row_sample = eval(args['row_sample'])
            self.data = utils.subsample(df=self.data, v=row_sample, axis=0)
            print('data.shape', self.data.shape)

        if args['col_sample']:
            col_sample = eval(args['col_sample'])
            fea_data, other_data = self.split_fea_and_other_cols(self.data)
            fea_data = utils.subsample(df=fea_data, v=col_sample, axis=1)
            self.data = pd.concat([other_data, fea_data], axis=1)
            print('data.shape', self.data.shape)

        # Assign type to categoricals
        # cat_cols = data.select_dtypes(include='object').columns.tolist()
        # data[cat_cols] = data[cat_cols].astype('category', ordered=False)

        # Filter out AUC>1
        # print('\nFilter some AUC outliers (>1)')
        # print('data.shape', data.shape)
        # data = data[[False if x>1.0 else True for x in data[target_name]]].reset_index(drop=True)
        # print('data.shape', data.shape)

        # if 'dlb' in args['other_features']:
        #     if logger:
        #         logger.info('\nAdd drug labels to features ...')
        #     # print(data['DRUG'].value_counts())

        #     # http://queirozf.com/entries/one-hot-encoding-a-feature-on-a-pandas-dataframe-an-example
        #     # One-hot encoder
        #     dlb = pd.get_dummies(data=data[['DRUG']], prefix=fea_prfx_dict['dlb'],
        #                         dummy_na=False).reset_index(drop=True)

        #     # Label encoder
        #     # dlb = data[['DRUG']].astype('category', ordered=False).reset_index(drop=True)
        #     # print(dlb.dtype)

        #     # Concat drug labels and other features
        #     data = pd.concat([dlb, data], axis=1).reset_index(drop=True)
        #     if logger:
        #         logger.info(f'dlb.shape {dlb.shape}')
        #         logger.info(f'data.shape {data.shape}')

        # if 'rna_clusters' in args['other_features']:
        #     # TODO
        #     pass


    def get_data_by_src(self, src_names):
        """ ... """
        if self.logger:
            self.logger.info('\nExtract sources ... {}'.format( src_names ))

        if src_names is not None:
            data = self.data[self.data['SOURCE'].isin(src_names)].reset_index(drop=True)
            
        if self.logger:
            self.logger.info(f'data.shape {data.shape}')
            self.logger.info('data memory usage: {:.3f} GB'.format(sys.getsizeof(data)/1e9))
            self.logger.info(data.groupby('SOURCE').agg({'CELL': 'nunique', 'DRUG': 'nunique'}).reset_index())
            
        return data
    
    
    def break_data(data):
        pass
        
        
    def get_data_by_src(self, src_names=None, target=None,
            target_transform=False, scaler_method=None):
        """ ... """
        if self.logger:
            self.logger.info('\nExtract sources ... {}'.format( src_names ))

        # Drop data points where target is NaN 
        self.data = self.data[~self.data[target].isna()]

        if src_names is not None:
            df = self.data[self.data['SOURCE'].isin(src_names)].reset_index(drop=True)
            ydata = df[target]
            df.drop(columns=target, inplace=True)
            xdata, meta = self.split_fea_and_other_cols( df )
            # ydata = self.extract_target()
        else:
            df = self.data
            ydata = self.data[target].copy()
            xdata, meta = self.split_fea_and_other_cols( self.data )
            # ydata = self.extract_target()
            
        if self.logger:
            self.logger.info(f'xdata.shape {xdata.shape}')
            self.logger.info('xdata memory usage: {:.3f} GB'.format(sys.getsizeof(xdata)/1e9))
            self.logger.info(df.groupby('SOURCE').agg({'CELL': 'nunique', 'DRUG': 'nunique'}).reset_index())

        if scaler_method is not None:
            if scaler_method == 'stnd':
                scaler = StandardScaler()
            elif scaler_method == 'minmax':
                scaler = MinMaxScaler()
            elif scaler_method == 'rbst':
                scaler = RobustScaler()

            # Scale train data
            fea_data, cat_data = self.get_num_and_cat_cols( xdata )
            colnames = fea_data.columns
            fea_data = pd.DataFrame( scaler.fit_transform(fea_data), columns=colnames ).astype(np.float32)
            xdata = pd.concat([cat_data, fea_data], axis=1)
        else:
            scaler = None

        self.print_fea_shapes( xdata )
        return xdata, ydata, meta, scaler

    
    def print_fea_shapes(self, df):
        """ Print features shapes.
        TODO: this doesn't have to be within the class
        Each feature name starts with a prefix indicating the feature type (`cell_rna.`, `drug_dsc.`, etc).
        Args:
            df : dataframe with feature columns
        """
        for prfx in np.unique(list(map(lambda x: x.split('.')[0], df.columns.tolist()))):
            cols = df.columns[[True if prfx in c else False for c in df.columns.tolist()]]
            self.logger.info('{}: {}'.format(prfx, df[cols].shape))    
    
    
class SrcData(TidyFromCombined):
    def __init__(self, src_names):
        if self.logger:
            self.logger.info('\nExtract sources ... {}'.format( src_names ))
            
        self.data = self.data[self.data['SOURCE'].isin(src_names)].reset_index(drop=True)
    
    

class Top6(DrugSensDataset):
    def __init__(self, datapath, fea_prfx_dict, fea_list, args, logger=None, random_state=None):
        self.fea_prfx_dict = {'rna': 'GE', 'dsc': 'DD'}


    
# ===================================================================================== 
def load_tidy_combined(datapath:str, fea_list:list, logger=None, random_state=None):
    """ Load tidy dataset that was generated from the combined dataframe.
    Args:
        datapath : full path to the tidy df
        fea_list : list of feature names to use for training 
    Returns:
        dataset : return the tidy df 
    """
    if logger:
        logger.info(f'\nLoad tidy data from ... \n{datapath}')
    data = pd.read_parquet(datapath, engine='auto', columns=None)

    if logger:
        logger.info(f'data.shape {data.shape}')
        logger.info('data memory usage: {:.3f} GB'.format(sys.getsizeof(data)/1e9))
    # print(data.groupby('SOURCE').agg({'CELL': 'nunique', 'DRUG': 'nunique', 'PUBCHEM': 'nunique'}).reset_index())

    # Replace characters that are illegal for xgboost/lightgbm feature names
    data = make_colnames_gbm_compatible( data )
    
    # Shuffle
    data = data.sample(frac=1.0, axis=0, random_state=random_state).reset_index(drop=True)

    # if args['tissue_type']:
    #    # didn't tested!
    #    data = data[data[''].isin([args['tissue_type']])].reset_index(drop=True)

    # # Subsample
    # if args['row_sample']:
    #     row_sample = eval(args['row_sample'])
    #     data = utils.subsample(df=data, v=row_sample, axis=0)
    #     print('data.shape', data.shape)

    # if args['col_sample']:
    #     col_sample = eval(args['col_sample'])
    #     fea_data, other_data = split_fea_and_other_cols(data, fea_prfx_dict=fea_prfx_dict)
    #     fea_data = utils.subsample(df=fea_data, v=col_sample, axis=1)
    #     data = pd.concat([other_data, fea_data], axis=1)
    #    print('data.shape', data.shape)
    
    # Keep subset of features
    data = extract_subset_fea(data, fea_list)
    return data

    
def get_data_by_src(dataset:pd.DataFrame, src_names:list, logger=None):
    """ Returns data for specific sources. """
    if logger:
        logger.info('\nExtract sources ... {}'.format( src_names ))

    # Extract the sources
    data = dataset[dataset['SOURCE'].isin(src_names)].reset_index(drop=True)
            
    if logger:
        logger.info(f'data.shape {data.shape}')
        logger.info('data memory usage: {:.3f} GB'.format(sys.getsizeof(data)/1e9))
        logger.info(data.groupby('SOURCE').agg({'CELL': 'nunique', 'DRUG': 'nunique'}).reset_index())
        
    return data


def break_src_data(data:pd.DataFrame, target:str='AUC', scaler_method:str='stnd', target_transform:bool=False, logger=None):
    """ Returns xdata, ydata, and meta from a dataset. Also, returns the scaler
    if the features were scaled.
    """
    # Drop data points where target is NaN 
    data = data[~data[target].isna()]
  
    # Get ydata 
    ydata = data[target]
    data.drop(columns=target, inplace=True)
   
    # Transform the target
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.boxcox.html
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.inv_boxcox.html
    # https://otexts.com/fpp2/transformations.html
    if target_transform:
        # y = np.log1p(ydata); plot_hist(x=y, var_name=target_name+'_log1p')
        # y = np.log(ydata+1); plot_hist(x=y, var_name=target_name+'_log+1')
        # y = np.log10(ydata+1); plot_hist(x=y, var_name=target_name+'_log10')
        # y = np.log2(ydata+1); plot_hist(x=y, var_name=target_name+'_log2')
        # y = ydata**2; plot_hist(x=ydata, var_name=target_name+'_x^2')
        y, lmbda = stats.boxcox(ydata+1); # utils.plot_hist(x=y, var_name=target_name+'_boxcox')
        ydata = pd.DataFrame(y)
        
    # Get xdata and meta 
    xdata, meta = split_fea_and_other_cols(data)

    if scaler_method is not None:
        if scaler_method == 'stnd':
            scaler = StandardScaler()
        elif scaler_method == 'minmax':
            scaler = MinMaxScaler()
        elif scaler_method == 'rbst':
            scaler = RobustScaler()

        # Scale train data
        fea_data, cat_data = get_num_and_cat_cols( xdata )
        colnames = fea_data.columns
        fea_data = pd.DataFrame( scaler.fit_transform(fea_data), columns=colnames ).astype(np.float32)
        fea_data.reset_index(drop=True, inplace=True)
        cat_data.reset_index(drop=True, inplace=True)
        xdata = pd.concat([cat_data, fea_data], axis=1)
    else:
        scaler = None

    print_fea_shapes(xdata, logger)
    return xdata, ydata, meta, scaler


def make_colnames_gbm_compatible(df):
    """ Replace characters that are illegal for xgboost/lightgbm feature names. """
    # xdata.columns = list(map(lambda s: s.replace('[','_').replace(']','_'), xdata.columns.tolist())) # required by xgboost
    regex = re.compile(r'\[|\]|<', re.IGNORECASE)
    df.columns = [regex.sub('_', c) if any(x in str(c) for x in set(('[', ']', '<'))) else c for c in df.columns.values]
    return df


def extract_subset_fea(data, fea_list):
    """ Extract a subset of features for training.
    Args:
        data : tidy dataset (df contains multiple cols including features, meta, and target)
        fea_list : e.g., (cell_features + drug_features)
    Returns:
        data : updated data
    """
    fea_data, other_data = split_fea_and_other_cols(data)
    fea_prfx_list = [fea_prfx_dict[fea] for fea in fea_list if fea in fea_prfx_dict.keys()]
    fea_sep = '.'
    fea = [c for c in fea_data.columns if (c.split(fea_sep)[0] + fea_sep) in fea_prfx_list]
    fea_data = fea_data[fea].reset_index(drop=True)

    # Concat feature set (fea_data) and other cols (other_data)
    data = pd.concat([other_data, fea_data], axis=1)      
    return data


def split_fea_and_other_cols(data):
    """ Extract two dfs from `data`: fea_data and other_data.
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
        # Get cols with specific feature prfx
        cols = data.columns[[True if prfx in c else False for c in data.columns.tolist()]]

        # If feature present in data, add it to df_fea_list, and drop from other_data
        if len(cols) > 0:  
            df = data[cols].copy()
            other_data.drop(columns=cols, inplace=True)
            df_fea_list.append(df)

    fea_data = pd.DataFrame(pd.concat(df_fea_list, axis=1))
    return fea_data, other_data


def get_num_and_cat_cols(df):
    """ Returns 2 dataframes. One with numerical cols and the other with non-numerical cols. """
    cat_cols = [x for x in df.columns if is_string_dtype(df[x]) is True]
    cat_df = df[cat_cols]
    num_df = df.drop(columns=cat_cols)
    num_df.reset_index(drop=True, inplace=True)
    cat_df.reset_index(drop=True, inplace=True)
    return num_df, cat_df


def print_fea_shapes(df, logger=None):
    """ Print features shapes.
    Each feature name starts with a prefix indicating the feature type (`rna.`, `dsc.`, etc).
    Args:
        df : dataframe with feature columns
    """
    for prfx in np.unique(list(map(lambda x: x.split('.')[0], df.columns.tolist()))):
        cols = df.columns[[True if prfx in c else False for c in df.columns.tolist()]]
        if logger:
            logger.info('{}: {}'.format(prfx, df[cols].shape))
        else:
            print('{}: {}'.format(prfx, df[cols].shape))
# =====================================================================================

    
    
    
    
    
def load_data(datapath, fea_prfx_dict, fea_list, args, logger=None, random_state=None):
    """ Load and pre-process the tidy data.
    TODO: create class??
    """
    if logger:
        logger.info(f'\nLoad tidy data from ... \n{datapath}')
    data = pd.read_parquet(datapath, engine='auto', columns=None)
    if logger:
        logger.info(f'data.shape {data.shape}')
        logger.info('data memory usage: {:.3f} GB'.format(sys.getsizeof(data)/1e9))
    # print(data.groupby('SOURCE').agg({'CELL': 'nunique', 'DRUG': 'nunique', 'PUBCHEM': 'nunique'}).reset_index())


    # Replace characters that are illegal for xgboost/lightgbm feature names
    # xdata.columns = list(map(lambda s: s.replace('[','_').replace(']','_'), xdata.columns.tolist())) # required by xgboost
    regex = re.compile(r'\[|\]|<', re.IGNORECASE)
    data.columns = [regex.sub('_', c) if any(x in str(c) for x in set(('[', ']', '<'))) else c for c in data.columns.values]


    # Shuffle
    data = data.sample(frac=1.0, axis=0, random_state=random_state).reset_index(drop=True)


    if args['tissue_type']:
        data = data[data[''].isin([args['tissue_type']])].reset_index(drop=True)


    # Subsample
    if args['row_sample']:
        row_sample = eval(args['row_sample'])
        data = utils.subsample(df=data, v=row_sample, axis=0)
        print('data.shape', data.shape)

    if args['col_sample']:
        col_sample = eval(args['col_sample'])
        fea_data, other_data = split_fea_and_other_cols(data, fea_prfx_dict=fea_prfx_dict)
        fea_data = utils.subsample(df=fea_data, v=col_sample, axis=1)
        data = pd.concat([other_data, fea_data], axis=1)
        print('data.shape', data.shape)


    # Drop data points where target is NaN 
    data = data[~data[args['target_name']].isna()]


    # Keep subset of features (new)
    data = extract_subset_features(data=data, fea_list=fea_list, fea_prfx_dict=fea_prfx_dict)


    # Extract test sources
    if logger:
        logger.info('\nExtract test sources ... {}'.format(args['test_sources']))
    
    if args['test_sources']:    
        te_data = data[data['SOURCE'].isin(args['test_sources'])].reset_index(drop=True)
        if logger:
            logger.info(f'te_data.shape {te_data.shape}')
            logger.info('data memory usage: {:.3f} GB'.format(sys.getsizeof(te_data)/1e9))
            logger.info(te_data.groupby('SOURCE').agg({'CELL': 'nunique', 'DRUG': 'nunique'}).reset_index())
    else:
        te_data = None
        if logger:
            logger.info('No test data.')


    # Extract train sources
    if logger:
        logger.info('\nExtract train sources ... {}'.format(args['train_sources']))
    tr_data = data[data['SOURCE'].isin(args['train_sources'])].reset_index(drop=True)
    if logger:
        logger.info(f'tr_data.shape {tr_data.shape}')
        logger.info('tr_data memory usage: {:.3f} GB'.format(sys.getsizeof(tr_data)/1e9))
        logger.info(tr_data.groupby('SOURCE').agg({'CELL': 'nunique', 'DRUG': 'nunique'}).reset_index())


    # Scale features
    if args['scaler'] is not None:
        if args['scaler'] == 'stnd':
            scaler = StandardScaler()
        elif args['scaler'] == 'minmax':
            scaler = MinMaxScaler()
        elif args['scaler'] == 'rbst':
            scaler = RobustScaler()

        # Scale train data
        fea_data, other_data = split_fea_and_other_cols(tr_data, fea_prfx_dict=fea_prfx_dict)
        fea_data, non_num_data = get_num_and_cat_cols(fea_data)
        colnames = fea_data.columns
        fea_data = pd.DataFrame( scaler.fit_transform(fea_data), columns=colnames ).astype(np.float32)
        tr_data = pd.concat([other_data, non_num_data, fea_data], axis=1)

        # Scale test data
        if te_data is not None:
            fea_data, other_data = split_fea_and_other_cols(te_data, fea_prfx_dict=fea_prfx_dict)
            fea_data, non_num_data = get_num_and_cat_cols(fea_data)
            colnames = fea_data.columns
            fea_data = pd.DataFrame( scaler.transform(fea_data), columns=colnames ).astype(np.float32)
            te_data = pd.concat([other_data, non_num_data, fea_data], axis=1)


    # Assign type to categoricals
    # cat_cols = data.select_dtypes(include='object').columns.tolist()
    # data[cat_cols] = data[cat_cols].astype('category', ordered=False)


    # Filter out AUC>1
    # print('\nFilter some AUC outliers (>1)')
    # print('data.shape', data.shape)
    # data = data[[False if x>1.0 else True for x in data[target_name]]].reset_index(drop=True)
    # print('data.shape', data.shape)


    # Transform the target
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.boxcox.html
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.inv_boxcox.html
    # https://otexts.com/fpp2/transformations.html
    if args['target_transform']:
        y = data[args['target_name']].copy()
        # y = np.log1p(ydata); plot_hist(x=y, var_name=target_name+'_log1p')
        # # y = np.log(ydata+1); plot_hist(x=y, var_name=target_name+'_log+1')
        # y = np.log10(ydata+1); plot_hist(x=y, var_name=target_name+'_log10')
        # y = np.log2(ydata+1); plot_hist(x=y, var_name=target_name+'_log2')
        # y = ydata**2; plot_hist(x=ydata, var_name=target_name+'_x^2')
        y, lmbda = stats.boxcox(y+1); # utils.plot_hist(x=y, var_name=target_name+'_boxcox', path=)
        data[args['target_name']] = y
        # ydata = pd.DataFrame(y)
        
        if te_data is not None:
            y = te_data[args['target_name']].copy()
            y, lmbda = stats.boxcox(y+1); # utils.plot_hist(x=y, var_name=target_name+'_boxcox', path=)
            te_data[args['target_name']] = y


    # if 'dlb' in args['other_features']:
    #     if logger:
    #         logger.info('\nAdd drug labels to features ...')
    #     # print(data['DRUG'].value_counts())

    #     # http://queirozf.com/entries/one-hot-encoding-a-feature-on-a-pandas-dataframe-an-example
    #     # One-hot encoder
    #     dlb = pd.get_dummies(data=data[['DRUG']], prefix=fea_prfx_dict['dlb'],
    #                         dummy_na=False).reset_index(drop=True)

    #     # Label encoder
    #     # dlb = data[['DRUG']].astype('category', ordered=False).reset_index(drop=True)
    #     # print(dlb.dtype)

    #     # Concat drug labels and other features
    #     data = pd.concat([dlb, data], axis=1).reset_index(drop=True)
    #     if logger:
    #         logger.info(f'dlb.shape {dlb.shape}')
    #         logger.info(f'data.shape {data.shape}')


    # if 'rna_clusters' in args['other_features']:
    #     # TODO
    #     pass

    return tr_data, te_data



def impute_values(data, fea_prfx_dict, logger=None):
    """ Impute missing values.
    TODO: this script is also in src/data/utils_data (put in a single place)
    Args:
        data : tidy dataset (df contains multiple cols including features, meta, and target)
        fea_prfx_dict : dict of feature prefixes, e.g.:
            fea_prfx_dict = {'rna': 'cell_rna.', 'cnv': 'cell_cnv.', 'dsc': 'drug_dsc.', 'fng': 'drug_fng.'}
        logger : logging object
    TODO: consider more advanced imputation methods:
    - www.rdocumentation.org/packages/Amelia/versions/1.7.4/topics/amelia
    - try regressor (impute continuous features) or classifier (impute discrete features)
    """
    from sklearn.impute import SimpleImputer, MissingIndicator
    data = data.copy()
    if logger is not None:
        logger.info('\nImpute missing features ... ({})'.format( list(fea_prfx_dict.keys())) )

    # Extract df that contains only features (no meta or response)
    fea_data, other_data = split_fea_and_other_cols(data=data, fea_prfx_dict=fea_prfx_dict)
    tot_miss_feas = sum(fea_data.isna().sum(axis=0) > 0)
    if logger is not None:
        logger.info('Total features with missing values (before impute): {}'.format(tot_miss_feas))

    if tot_miss_feas > 0:
        # Split numerical from other features (only numerical will be imputed;
        # The other features can be cell and drug labels)
        fea_data, non_num_data = get_num_and_cat_cols(fea_data)

        # Proceed with numerical featues
        colnames = fea_data.columns

        # Impute missing values
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean', verbose=1)
        dtypes_dict = fea_data.dtypes # keep the original dtypes because fit_transform casts to np.float64
        fea_data_imputed = imputer.fit_transform(fea_data)
        fea_data_imputed = pd.DataFrame(fea_data_imputed, columns=colnames)
        fea_data_imputed = fea_data_imputed.astype(dtypes_dict) # cast back to the original data type
        
        if logger is not None:
            logger.info('Total features with missing values (after impute): {}'.format( sum(fea_data_imputed.isna().sum(axis=0) > 0)) )

        # Concat features (xdata_imputed) and other cols (other_data)
        data = pd.concat([other_data, non_num_data, fea_data_imputed], axis=1)
        
    return data

