"""
TODO:
- add chemparter data to the tidy df.
"""
from __future__ import division
from __future__ import print_function

import warnings
warnings.filterwarnings('ignore')

import os
import sys
import platform
import time
import datetime
import argparse
import psutil
import numpy as np
import pandas as pd

# https://github.com/MTG/sms-tools/issues/36
import matplotlib
# matplotlib.use('TkAgg')
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import seaborn as sns

# Utils
# file_path = os.getcwd()
# file_path = os.path.join(file_path, 'src/data')
# os.chdir(file_path)
# sys.path.append(file_path)

file_path = os.path.dirname(os.path.realpath(__file__))  # os.path.dirname(os.path.abspath(__file__))
##utils_path = os.path.abspath(os.path.join(file_path, 'utils'))
##sys.path.append(utils_path)
import utils_data as utils

# TODO: fix this similar to CANDLE (i.e. download file server)
if sys.platform == 'darwin':  # my mac
    DATADIR = '/Users/apartin/work/jdacs/Benchmarks/Data/Pilot1'
elif 'theta' in platform.uname()[1]:
    DATADIR = None # TODO: set this
else:
    DATADIR = '/vol/ml/apartin/Benchmarks/Data/Pilot1'
OUTDIR = os.path.join(file_path, '../../data/processed/from_combined')
os.makedirs(OUTDIR, exist_ok=True)

t0 = time.time()


# ========================================================================
#       Args TODO: add to argparse
# ========================================================================
# sources = ['ccle', 'gcsi', 'gdsc', 'ctrp']
sources = ['ccle', 'gcsi', 'gdsc', 'ctrp', 'nci60']
drug_features = ['dsc']  # ['dsc', 'fng']
cell_features = ['rna']  # ['rna', 'cnv']
dropna_thres = 0.4

verbose = True
na_values = ['na', '-', '']
tidy_data_format = 'parquet'

# Analysis of fibro samples are implemented in ccle_fibroblast.py and ccle_preproc.R
drop_fibro = True
fibro_names = ['CCLE.HS229T', 'CCLE.HS739T', 'CCLE.HS840T', 'CCLE.HS895T', 'CCLE.RKN',
               'CTRP.Hs-895-T', 'CTRP.RKN', 'GDSC.RKN', 'gCSI.RKN']

# Prefix to add to feature names based on feature types
fea_prfx_dict = {'rna': 'cell_rna.',
                 'cnv': 'cell_cnv.',
                 'dsc': 'drug_dsc.',
                 'fng': 'drug_fng.'}

prfx_dtypes = {'rna': np.float32,
               'cnv': np.int8,
               'dsc': np.float32,
               'fng': np.int8}


# ========================================================================
#       Logger
# ========================================================================
t = datetime.datetime.now()
t = [t.year, '-', t.month, '-', t.day, '_', 'h', t.hour, '-', 'm', t.minute]
t = ''.join([str(i) for i in t])
logfilename = os.path.join(OUTDIR, 'tidy_data_' + t + '.log')
logger = utils.setup_logger(logfilename=logfilename)

logger.info(f'File path: {file_path}')
logger.info(f'Num of system CPUs: {psutil.cpu_count()}')


# ========================================================================
#       Load response data
# ========================================================================
# filename = 'ChemPartner_single_response_agg'
filename = 'combined_single_response_agg'
logger.info(f'\n\nLoading combined response ... {filename}')
rsp_cols = ['AUC', 'AUC1', 'EC50', 'EC50se',
            'R2fit', 'Einf', 'IC50',
            'HS', 'AAC1', 'DSS1']
rsp = pd.read_table(os.path.join(DATADIR, filename), sep='\t',
                    na_values=na_values,
                    dtype={'SOURCE': str, 'CELL': str, 'DRUG': str,
                           'AUC': np.float32, 'IC50': np.float32, 'EC50': np.float32,
                           'EC50se': np.float32, 'R2fit': np.float32, 'Einf': np.float32,
                           'HS': np.float32, 'AAC1': np.float32, 'AUC1': np.float32, 'DSS1': np.float32},
                    warn_bad_lines=True)
rsp['SOURCE'] = rsp['SOURCE'].apply(lambda x: x.lower())
logger.info(f'rsp.shape {rsp.shape}')

# Replace -Inf and Inf with nan
rsp.replace([np.inf, -np.inf], value=np.nan, inplace=True)

if verbose:
    logger.info('rsp memory usage: {:.3f} GB'.format(sys.getsizeof(rsp)/1e9))
    logger.info('')
    logger.info(rsp.groupby('SOURCE').agg({'CELL': 'nunique', 'DRUG': 'nunique'}).reset_index())
    
# Plot distributions of target variables
utils.plot_rsp_dists(rsp=rsp, rsp_cols=rsp_cols, savepath=os.path.join(OUTDIR, 'rsp_dists.png'))

# Plot distribution of a single target
# target_name = 'EC50se'
# fig, ax = plt.subplots()
# x = rsp[target_name].copy()
# x = x[~x.isna()].values
# sns.distplot(x, bins=100, ax=ax)
# plt.savefig(os.path.join(OUTDIR, target_name+'.png'), bbox_inches='tight')


# ========================================================================
#   Load rna (combined_dataset)
# ========================================================================
logger.info('\n\nLoading rna-seq ... ')
lincs = utils.CombinedRNASeqLINCS(datadir=DATADIR, dataset='raw', sources=sources, na_values=na_values, verbose=False)
rna, cmeta = lincs._df_rna, lincs._meta
rna.rename(columns={'Sample': 'CELL'}, inplace=True)
cmeta.rename(columns={'Sample': 'CELL', 'source': 'SOURCE'}, inplace=True)
rna = rna.rename(columns={c: fea_prfx_dict['rna']+c for c in rna.columns[1:] if fea_prfx_dict['rna'] not in c}) # add fea prefix
logger.info(f'rna.shape {rna.shape}')

# Impute missing values
rna = utils.impute_values(data=rna, fea_prfx_dict=fea_prfx_dict, logger=logger)

if verbose:
    logger.info('rna memory usage: {:.3f} GB'.format(sys.getsizeof(rna)/1e9))
    logger.info('')
    logger.info(cmeta.groupby('SOURCE').agg({'CELL': 'nunique', 'ctype': 'nunique', 'csite': 'nunique'}).reset_index())
    # print(cmeta.groupby(['SOURCE', 'csite']).agg({'CELL': 'nunique'}).reset_index())


# ========================================================================
#   Load drug descriptors
# ========================================================================
filename = 'Combined_PubChem_dragon7_descriptors.tsv'
logger.info('\n\nLoading drug descriptors ... {}'.format('Combined_PubChem_dragon7_descriptors.tsv'))
path = os.path.join(DATADIR, filename)
cols = pd.read_table(path, engine='c', nrows=0)
dtype_dict = {c: prfx_dtypes['dsc'] for c in cols.columns[1:]}
dsc = pd.read_table(path, dtype=dtype_dict, na_values=na_values, warn_bad_lines=True)
dsc.rename(columns={'NAME': 'PUBCHEM'}, inplace=True)
dsc = dsc.rename(columns={c: fea_prfx_dict['dsc']+c for c in dsc.columns[1:] if fea_prfx_dict['dsc'] not in c}) # add fea prefix
logger.info(f'dsc.shape {dsc.shape}')


# ------------------
# Filter descriptors
# ------------------
# dsc.nunique(dropna=True).value_counts()
# dsc.nunique(dropna=True).sort_values()

logger.info('Drop descriptors with *lots* of NA values ...')
utils.plot_dsc_na_dist(dsc=dsc, savepath=os.path.join(OUTDIR, 'dsc_hist_ratio_of_na.png'))
dsc = utils.dropna(df=dsc, axis=1, th=dropna_thres)
logger.info(f'dsc.shape {dsc.shape}')
# dsc.isna().sum().sort_values(ascending=False)

# There are descriptors for which there is a single unique value excluding NA (drop those)
logger.info('Drop descriptors that have a single unique value (excluding NAs) ...')
col_idx = dsc.nunique(dropna=True).values==1
dsc = dsc.iloc[:, ~col_idx]
logger.info(f'dsc.shape {dsc.shape}')

# There are still lots of descriptors which have only a few unique values
# We can categorize those values. E.g.: 564 descriptors have only 2 unique vals,
# and 154 descriptors have only 3 unique vals, etc.
# TODO: use utility code from p1h_alex/utils/data_preproc.py that transform those
# features into categorical and also applies an appropriate imputation.
# dsc.nunique(dropna=True).value_counts()[:10]
# dsc.nunique(dropna=True).value_counts().sort_index()[:10]

# Impute missing values
dsc = utils.impute_values(data=dsc, fea_prfx_dict=fea_prfx_dict, logger=logger)

# Drop low var cols
# tmp, idx = utils_all.drop_low_var_cols(df=dsc, skipna=False)

if verbose:
    logger.info('dsc memory usage: {:.3f} GB'.format(sys.getsizeof(dsc)/1e9))
    

# ========================================================================
#   Load drug meta
# ========================================================================
filename = 'drug_info'
logger.info(f'\n\nLoading drug metadata ... {filename}')
dmeta = pd.read_table(os.path.join(DATADIR, filename), dtype=object)
dmeta['PUBCHEM'] = 'PubChem.CID.' + dmeta['PUBCHEM']
dmeta.insert(loc=0, column='SOURCE', value=dmeta['ID'].map(lambda x: x.split('.')[0].lower()))
dmeta.rename(columns={'ID': 'DRUG'}, inplace=True)
logger.info(f'dmeta.shape {dmeta.shape}')

if verbose:
    # Number of unique drugs in each data source
    # TODO: What's going on with CTRP and GDSC? Why counts are not consistent across the fields??
    logger.info('')
    logger.info(dmeta.groupby('SOURCE').agg({'DRUG': 'nunique', 'NAME': 'nunique',
                                             'CLEAN_NAME': 'nunique', 'PUBCHEM': 'nunique'}).reset_index())


# ========================================================================
#   Drop fibroblast
# ========================================================================
if drop_fibro:
    logger.info('\n\nDrop fibroblast samples ...')
    rna = rna[rna['CELL'].map(lambda x: False if x in fibro_names else True)]
    cmeta = cmeta[cmeta['CELL'].map(lambda x: False if x in fibro_names else True)]
    rsp = rsp[rsp['CELL'].map(lambda x: False if x in fibro_names else True)]
    logger.info(f'rsp.shape   {rsp.shape}')
    logger.info(f'rna.shape   {rna.shape}')
    logger.info(f'cmeta.shape {cmeta.shape}')

    tidy_data_name = 'tidy_data_no_fibro'
else:
    tidy_data_name = 'tidy_data'


# ========================================================================
#   Merge the data
# ========================================================================
"""
Data tables: rsp, rna, dsc, cmeta, dmeta
(rsp, rna): on 'CELL'
(rsp, dsc): on pubchem through fields in dmeta
"""
logger.info('\n\n==========================')
logger.info('... Start merging data ...')
logger.info('==========================')

# Update rsp with additional drug field 'PUBCHEM' (this will be used to merge with descriptors)
logger.info('\nMerge response (rsp) with drug metadata (dmeta) on DRUG in order to add PUBCHEM (required for descriptors) ...')
logger.info(f'rsp.shape   {rsp.shape}')
logger.info(f'dmeta.shape {dmeta.shape}')
rsp = pd.merge(rsp, dmeta[['DRUG', 'PUBCHEM']], on='DRUG', how='left')
logger.info(f'rsp.shape   {rsp.shape}')
logger.info('NA values after merging rsp and dmeta: \n{}'.format(rsp[['DRUG', 'PUBCHEM']].isna().sum()))
logger.info('')
logger.info(rsp.groupby('SOURCE').agg({'DRUG': 'nunique', 'PUBCHEM': 'nunique'}).reset_index())


# --------------------
# Merge rsp with cmeta
# --------------------
logger.info('\nMerge response (rsp) and cell metadata (cmeta) ...')
logger.info(f'rsp.shape   {rsp.shape}')
logger.info(f'cmeta.shape {cmeta.shape}')
rsp1 = pd.merge(rsp, cmeta[['CELL', 'core_str', 'csite', 'ctype', 'simplified_csite', 'simplified_ctype']], on='CELL', how='left')
logger.info(f'rsp1.shape  {rsp1.shape}')
logger.info('')
logger.info(rsp1.groupby('SOURCE').agg({'CELL': 'nunique', 'DRUG': 'nunique',
                                        'PUBCHEM': 'nunique'}).reset_index())
del rsp

# --------------
# Merge with rna
# --------------
logger.info('\nMerge with expression (rna) ...')
logger.info(f'rsp1.shape {rsp1.shape}')
logger.info(f'rna.shape  {rna.shape}')
rsp2 = pd.merge(rsp1, rna, on='CELL', how='inner')
logger.info(f'rsp2.shape {rsp2.shape}')
logger.info('')
logger.info(rsp2.groupby('SOURCE').agg({'CELL': 'nunique', 'DRUG': 'nunique',
                                        'PUBCHEM': 'nunique'}).reset_index())
del rsp1

# --------------
# Merge with dsc
# --------------
logger.info('\nMerge with descriptors (dsc) ...')
logger.info(f'rsp2.shape {rsp2.shape}')
logger.info(f'dsc.shape  {dsc.shape}')
data = pd.merge(rsp2, dsc, on='PUBCHEM', how='inner')
logger.info(f'data.shape {data.shape}')
logger.info('')
logger.info(data.groupby('SOURCE').agg({'CELL': 'nunique', 'DRUG': 'nunique',
                                        'PUBCHEM': 'nunique'}).reset_index())
del rsp2


# Summary of memory usage
logger.info('\nTidy dataframe: {:.3f} GB'.format(sys.getsizeof(data)/1e9))
if verbose:
    for prfx in prfx_dtypes.keys():
        cols = [c for c in data.columns if prfx in c]
        tmp = data[cols]
        mem = 0 if tmp.shape[1]==0 else sys.getsizeof(tmp)/1e9
        logger.info("# of '{}' features: {} ({:.2f} GB)".format(prfx, len(cols), mem))


# Cast features
# https://stackoverflow.com/questions/15891038/change-data-type-of-columns-in-pandas
# for fea_prfx, fea_frmt in prfx_dtypes.items():
#     print(f'feature type and format: ({fea_prfx}, {fea_frmt})')
#     dict_types = {c: fea_frmt for c in tmp.columns if fea_prfx in c}
#     if len(dict_types) > 0:
#         data = data.astype(dict_types)

logger.info('\nEnd of data per-processing: {:.2f} mins'.format((time.time()-t0)/60))


# ========================================================================
#   Finally save data
# ========================================================================
logger.info('\nSave tidy dataframe ...')
t0 = time.time()
data.drop(columns='STUDY', inplace=True) # gives error when save in 'parquet' format
if tidy_data_format == 'parquet':
    tidy_filepath = os.path.join(OUTDIR, tidy_data_name+'.parquet')
    data.to_parquet(tidy_filepath, engine='auto', compression='snappy')
else: 
    tidy_filepath = os.path.join(OUTDIR, 'tidy_data')
    data.to_csv(tidy_filepath, sep='\t')
logger.info('Save tidy data to disk: {:.2f} mins'.format((time.time()-t0)/60))

# Check that the saved data is the same as original one
logger.info(f'\nLoad tidy dataframe {tidy_data_format} ...')
t0 = time.time()
if tidy_data_format == 'parquet':
    data_fromfile = pd.read_parquet(tidy_filepath, engine='auto', columns=None)
else:
    data_fromfile = pd.read_table(tidy_filepath, sep='\t')
logger.info('Load tidy data to disk: {:.2f} mins'.format((time.time()-t0)/60))

logger.info(f'\nLoaded data is the same as original: {data.equals(data_fromfile)}')

logger.info('\n=====================================================')
logger.info(f'Tidy data file path:\n{os.path.abspath(tidy_filepath)}')
logger.info('=====================================================')


# ========================================================================
# ========================================================================
# ========================================================================
#  EDA
# ========================================================================
# ========================================================================
# ========================================================================
# rsp = pd.read_csv(os.path.join(DATADIR, 'combined_single_response_agg'), sep='\t')
# rsp = rsp[['SOURCE', 'CELL', 'DRUG', 'AUC', 'AUC1', 'IC50']]
# print(rsp.shape)
# print(rsp[:2])
# print(rsp.groupby('SOURCE').agg({'CELL': 'nunique', 'DRUG': 'nunique'}).reset_index())

# # Drug meta
# dmeta = pd.read_table(os.path.join(DATADIR, 'drug_info'), dtype=object)
# dmeta['PUBCHEM'] = 'PubChem.CID.' + dmeta['PUBCHEM']
# dmeta.insert(loc=0, column='SOURCE', value=dmeta['ID'].map(lambda x: x.split('.')[0]))
# dmeta.rename(columns={'ID': 'DRUG'}, inplace=True)
# print(dmeta.shape)
# print(dmeta[:2])
# print(dmeta.groupby('SOURCE').agg({'DRUG': 'nunique'}).reset_index())

# # Merge rsp with dmeta
# rsp = pd.merge(rsp, dmeta[['DRUG', 'NAME', 'CLEAN_NAME']], on='DRUG', how='inner')
# print(rsp.shape)
# print(rsp[:2])
# print(rsp.groupby('SOURCE').agg({'CELL': 'nunique', 'DRUG': 'nunique'}).reset_index())

# # Load rna and meta
# rna, meta = utils_all.load_lincs1000(dataset='raw', sources=['ccle', 'ctrp', 'gdsc', 'gcsi', 'nci60'])
# meta = meta[['Sample', 'source', 'core_str', 'csite', 'ctype', 'simplified_csite', 'simplified_ctype']]
# meta.rename(columns={'Sample': 'CELL'}, inplace=True)
# print(meta.nunique())

# # Merge rsp and meta
# mm = pd.merge(rsp, meta, on='CELL', how='inner')
# print(mm.shape)
# print(mm[:2])

# mm.groupby('SOURCE').agg({'CELL': 'nunique'}).reset_index()

# df = mm.groupby(['SOURCE', 'simplified_csite']).agg({'CELL': 'nunique'}).reset_index()
# df.pivot(index='simplified_csite', columns='SOURCE', values='CELL')
# # utils_all.contingency_table(mm, cols=['SOURCE', 'simplified_csite'], to_plot=True, figsize=None, title=None, margins=False, normalize=False)



# label = 'simplified_csite'   # meta label/field
# val = 'hematologic/blood'   # label val
# dd = mm[mm['simplified_csite'].isin([val])]
# dd[label].value_counts()
# dd.groupby('SOURCE').agg({'DRUG': 'nunique'})

# dd.groupby(['SOURCE', 'DRUG']).agg({'AUC': ['mean', 'std'], 'AUC1': ['mean', 'std']})
# dd.groupby(['SOURCE', 'NAME']).agg({'AUC': ['mean', 'std', 'size'], 'AUC1': ['mean', 'std']})
# dd.groupby(['NAME']).agg({'AUC': ['mean', 'std', 'size'], 'AUC1': ['mean', 'std', 'size']})

# for i, mtype in enumerate(mm[label].unique()):
#     dd = mm[mm[label].isin(val)]

