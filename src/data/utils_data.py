import os
import sys
import logging
import numpy as np
import pandas as pd

import matplotlib
# matplotlib.use('TkAgg')
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import seaborn as sns
from scipy.stats import norm, skew


# TODO: fix this similar to CANDLE
if sys.platform == 'darwin':
    DATADIR = '/Users/apartin/work/jdacs/Benchmarks/Data/Pilot1'
else:
    DATADIR = '/vol/ml/apartin/Benchmarks/Data/Pilot1'
CELLMETA_FILENAME = 'combined_metadata_2018May.txt'


def setup_logger(logfilename='logfile.log'):
    """ Create logger. Output to file and console. """
    # Create file handler
    # "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    # "[%(asctime)s %(process)d] %(message)s"
    # fileFormatter = logging.Formatter("%(asctime)s : %(threadName)-12.12s : %(levelname)-5.5s : %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fileFormatter = logging.Formatter("%(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fileHandler = logging.FileHandler(filename=logfilename)
    fileHandler.setFormatter(fileFormatter)
    fileHandler.setLevel(logging.INFO)

    # Create console handler
    # consoleFormatter = logging.Formatter("%(name)-12s : %(levelname)-8s : %(message)s")
    consoleFormatter = logging.Formatter("%(message)s")
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(consoleFormatter)
    consoleHandler.setLevel(logging.INFO)

    # Create logger and add handlers
    # logger = logging.getLogger(__name__)
    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    logger.addHandler(fileHandler)
    logger.addHandler(consoleHandler)

    # from combo (when use candle)
    # for log in [logger, uno_data.logger]:
    #     log.setLevel(logging.DEBUG)
    #     log.addHandler(fh)
    #     log.addHandler(sh)

    logger.info('{}'.format('-'*90))
    return logger


def split_features_and_other_cols(data, fea_prfx_dict):
    """ Extract two dfs from `data`: fea_data and other_data.
    TODO: this script is also in src/models/utils_tidy (put in a single place)
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


def impute_values(data, fea_prfx_dict, logger):
    """ Impute missing values.
    TODO: this script is also in src/models/utils_tidy (put in a single place)
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
    logger.info('\nImpute missing features ... ({})'.format(list(fea_prfx_dict.keys())))

    # Extract df that contains only features (no meta or response)
    fea_data, other_data = split_features_and_other_cols(data=data, fea_prfx_dict=fea_prfx_dict)
    tot_miss_feas = sum(fea_data.isna().sum() > 1)
    logger.info('Total features with missing values (before imputation): {}'.format(tot_miss_feas))

    if tot_miss_feas > 0:
        colnames = fea_data.columns

        # Impute missing values
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean', verbose=1)
        dtypes_dict = fea_data.dtypes # keep the original dtypes because fit_transform casts to np.float64
        fea_data_imputed = imputer.fit_transform(fea_data)
        fea_data_imputed = pd.DataFrame(fea_data_imputed, columns=colnames)
        fea_data_imputed = fea_data_imputed.astype(dtypes_dict) # cast back to the original data type

        logger.info('Total features with missing values (after impute): {}'.format(sum(fea_data_imputed.isna().sum() > 1)))

        # Concat features (xdata_imputed) and other cols (other_data)
        data = pd.concat([other_data, fea_data_imputed], axis=1)
        
    return data


def plot_rsp_dists(rsp, rsp_cols, savepath=None):
    """ Plot distributions of response variables.
    Args:
        rsp : df of response values
        rsp_cols : list of col names
        savepath : full path to save the image
    """
    ncols = 4
    nrows = int(np.ceil(len(rsp_cols)/ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=False, sharey=False)
    for i, ax in enumerate(axes.ravel()):
        if i >= len(rsp_cols):
            fig.delaxes(ax) # delete un-used ax
        else:
            target_name = rsp_cols[i]
            x = rsp[target_name].copy()
            x = x[~x.isna()].values
            sns.distplot(x, bins=100, kde=True, ax=ax, label=target_name, # fit=norm, 
                        kde_kws={'color': 'k', 'lw': 0.4, 'alpha': 0.8},
                        hist_kws={'color': 'b', 'lw': 0.4, 'alpha': 0.5})
            ax.tick_params(axis='both', which='major', labelsize=7)
            txt = ax.yaxis.get_offset_text(); txt.set_size(7) # adjust exponent fontsize in xticks
            txt = ax.xaxis.get_offset_text(); txt.set_size(7)
            ax.legend(fontsize=5, loc='best')
            ax.grid(True)

    plt.tight_layout()
    # plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
    if savepath is not None:
        plt.savefig(savepath, bbox_inches='tight', dpi=300)
    else:
        plt.savefig('rsp_dists.png', bbox_inches='tight', dpi=300)


def plot_dsc_na_dist(dsc, savepath=None):
    """ Plot distbirution of na values in drug descriptors. """
    fig, ax = plt.subplots()
    sns.distplot(dsc.isna().sum(axis=0)/dsc.shape[0], bins=100, kde=False, hist_kws={'alpha': 0.7})
    plt.xlabel('Ratio of NA values')
    plt.title('Histogram of descriptors')
    plt.grid(True)
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath, bbox_inches='tight', dpi=300)
    else:
        plt.savefig('dsc_hist_ratio_of_na.png', bbox_inches='tight', dpi=200)


def dropna(df, axis=0, th=0.4):
    """ Drop rows or cols based on the ratio of NA values along the axis.
    Args:
        df : input df
        th (float) : if the ratio of NA values along the axis is larger that th, then drop all the values
        axis (int) : 0 to drop rows; 1 to drop cols
    Returns:
        df : updated df
    """
    df = df.copy()
    axis = 0 if axis==1 else 1
    col_idx = df.isna().sum(axis=axis)/df.shape[axis] <= th
    df = df.iloc[:, col_idx.values]
    return df


# --------------------------------------------------------------------------------------------------------------
class CombinedRNASeqLINCS():
    """ Combined LINCS dataset. """
    def __init__(self, datadir=DATADIR, dataset='raw', cellmeta_filename=CELLMETA_FILENAME, sources=[],
                 na_values=['na', '-', ''], verbose=True):
        """ Note that df_rna file must have the following structure:
        df_rna.columns[0] --> 'Sample'
        df_rna.columns[1:] --> gene names
        df_rna.iloc[:, 0] --> strings of sample names
        df_rna.iloc[:, 1:] --> gene expression values
        
        Example:
            DATADIR = '/Users/apartin/work/jdacs/Benchmarks/Data/Pilot1'
            CELLMETA_FILENAME = 'combined_metadata_2018May.txt'
            lincs = CombinedLINCS(dataset='combat', datadir=DATADIR, cellmeta_filename=CELLMETA_FILENAME)
        """
        if dataset == 'raw':
            DATASET = 'combined_rnaseq_data_lincs1000'
        elif dataset == 'source_scale':
            DATASET = 'combined_rnaseq_data_lincs1000_source_scale'
        elif dataset == 'combat':
            DATASET = 'combined_rnaseq_data_lincs1000_combat'
        else:
            raise ValueError(f'The passed dataset ({DATASET}) is not supported.')
            
        data_type = np.float32

        # Load RNA-Seq
        path = os.path.join(datadir, DATASET)
        cols = pd.read_table(path, nrows=0, sep='\t')
        dtype_dict = {c: data_type for c in cols.columns[1:]}
        df_rna = pd.read_table(path, dtype=dtype_dict, sep='\t', na_values=na_values, warn_bad_lines=True)
        df_rna = self._keep_sources(df_rna, sources=sources) 

        # Load metadata
        meta = pd.read_table(os.path.join(datadir, cellmeta_filename), sep='\t')
        meta = self._update_metadata_comb_may2018(meta)
        
        # Merge df_rna and meta
        df_rna, meta = self._update_df_and_meta(df_rna, meta, on='Sample')

        if verbose:
            print(f'\nDataset: {DATASET}')
            print(f'df_rna {df_rna.shape}')
            if meta is not None:
                print(f'meta   {meta.shape}')
            print(df_rna['Sample'].map(lambda s: s.split('.')[0]).value_counts())
            
        self._df_rna, self._meta = df_rna, meta


    def _keep_sources(self, df_rna, sources=[]):
        """ Keep specific data sources.
		Args:
			sources (list) : list of strings indicating the sources/studies to extract.
                (e.g., source=['ccle', 'ctrp'])
		"""
        if len(sources) == 0:
            return df_rna

        if isinstance(sources, str):
            sources = [sources]
            
        if len(sources) > 0:
            sources = [s.lower() for s in sources]
            df_rna = df_rna.loc[df_rna['Sample'].map(lambda s: s.split('.')[0].lower() in sources), :].reset_index(drop=True)
        else:
            print('Empty list was passed to the arg `sources`. Returns the same dataframe.')

        return df_rna  


    def _update_metadata_comb_may2018(self, meta):
        """ Update the metadata of the combined RNA-Seq (Judith metadata):
        /nfs/nciftp/private/tmp/jcohn/metadataForRNASeq2018Apr/combined_metadata_2018May.txt
        Remove "unnecessary" columns.
        Use Argonne naming conventions (e.g. GDC rather than TCGA).
        """
        # Rename columns
        meta = meta.rename(columns={'sample_name': 'Sample',
                                    'dataset': 'source',
                                    'sample_category': 'category',
                                    'sample_descr': 'descr',
                                    'tumor_site_from_data_src': 'csite',
                                    'tumor_type_from_data_src': 'ctype',
                                    'simplified_tumor_site': 'simplified_csite',
                                    'simplified_tumor_type': 'simplified_ctype'
                                    })

        meta['source'] = meta['source'].map(lambda x: x.lower())
        meta['csite'] = meta['csite'].map(lambda x: x.strip())
        meta['ctype'] = meta['ctype'].map(lambda x: x.strip())
        meta['source'] = meta['source'].map(lambda x: 'gdc' if x=='tcga' else x)
        return meta

    
    def _update_df_and_meta(self, df_rna, meta, on='Sample'):
        """ Merge df_rna and meta on a column specified by `on`.
        Args:
            df_rna (df) : df rna
            meta (df) : df meta
        Returns:
            df_rna (df) : df rna updated
            meta (df) : df meta updated
        """
        df_rna = df_rna.copy()
        meta = meta.copy()
        df = pd.merge(meta, df_rna, how='inner', on=on).reset_index(drop=True)

        df_rna = df[['Sample'] + df_rna.columns[1:].tolist()]
        meta = df.drop(columns=df_rna.columns[1:].tolist())
        return df_rna, meta
    

    def df_rna(self):
        """ df_rna getter. """
        df_rna = self._df_rna.copy()
        return df_rna
    
    
    def meta(self):
        """ meta getter. """
        meta = self._meta.copy()
        return meta
    
    
    # def extract_specific_datasets(self, sources=[]):
    def get_subset(self, sources=[]):
        """ Get samples of the specified data sources (this is a getter method).
        Args:
            sources (list) : list of strings indicating the sources/studies to extract
        Returns:
            df_rna (df) : df rna for the data sources specified by `sources`
            meta (df) : df meta for the data sources specified by `sources`
        Example:
            cells_rna, cells_meta = lincs.get_subset(sources=['ccle','nci60'])
        """
        df_rna = self._df_rna.copy()
        meta = self._meta.copy()

        if len(sources) > 0:
            sources = [s.lower() for s in sources]
            df_rna = df_rna.loc[df_rna['Sample'].map(lambda s: s.split('.')[0].lower() in sources), :].reset_index(drop=True)
            df_rna, meta = self._update_df_and_meta(df_rna, meta, on='Sample')
        else:
            print('Empty list was passed to the arg `sources`. Returns the same dataframe.')

        return df_rna, meta
# --------------------------------------------------------------------------------------------------------------