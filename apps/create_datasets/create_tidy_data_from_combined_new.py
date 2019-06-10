"""
This "new" version of the code uses a different dataframe of descriptors:
'pan_drugs_dragon7_descriptors.tsv' instead of 'combined_pubchem_dragon7_descriptors.tsv'
"""
from __future__ import division, print_function

import warnings
warnings.filterwarnings('ignore')

import os
import sys
from pathlib import Path
import platform
import argparse
import datetime
from time import time
from pprint import pprint
import psutil

import sklearn
import numpy as np
import pandas as pd

# github.com/mtg/sms-tools/issues/36
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import seaborn as sns

seed = None
t0 = time()


# file path
# file_path = os.path.dirname(os.path.realpath(__file__))
file_path = Path(__file__).resolve().parent


# utils
utils_path = file_path / '../../utils'
sys.path.append(str(utils_path)) 
# import utils_data
import utils_tidy
import utils
from classlogger import Logger
from rna import CombinedRNASeqLINCS


# todo: fix this similar to candle (i.e. download file server)
# if sys.platform == 'darwin':  # my mac
#     datadir = pathlib.path('/users/apartin/work/jdacs/benchmarks/data/pilot1')
# elif 'theta' in platform.uname()[1]:
#     datadir = none # todo: set this
# else:
#    datadir = pathlib.path('/vol/ml/apartin/benchmarks/data/pilot1')
DATADIR = file_path / '../../data/raw'
OUTDIR = file_path / '../../data/processed/from_combined'

RSP_FILENAME = 'combined_single_response_agg'  # reposne data
RSP_FILENAME_CHEM = 'chempartner_single_response_agg'  # reposne data
CELLMETA_FILENAME = 'combined_metadata_2018May.txt'

# DSC_FILENAME = 'combined_pubchem_dragon7_descriptors.tsv'  # drug descriptors data (old)
DSC_FILENAME = 'pan_drugs_dragon7_descriptors.tsv'  # drug descriptors data (new)

DRUG_META_FILENAME = 'drug_info'



# ========================================================================
#       Arg parser
# ========================================================================
# Arg parser
psr = argparse.ArgumentParser(description='Create tidy data. Arg parser.')
psr.add_argument('--drop_fibro', action='store_true', default=False, help='Default: True')
psr.add_argument('--drop_bad', action='store_true', default=False, help='Default: True')
psr.add_argument('--dropna_th', type=float, default=0.4, help='default=0.4')
psr.add_argument('-v', '--verbose', action='store_true', default=False, help='Default: True')
psr.add_argument('--drug_fea', type=str, nargs='+', choices=['dsc', 'fng'], default=['dsc'], help="default=['dsc']")
psr.add_argument('--cell_fea', type=str, nargs='+', choices=['rna', 'cnv'], default=['rna'], help="default=['rna']")

args = vars(psr.parse_args())
pprint(args)


# Args
drop_fibro = args['drop_fibro']
drop_bad = args['drop_bad']
dropna_thres = args['dropna_th']
verbose = args['verbose']
drug_features = args['drug_fea']
cell_features = args['cell_fea']


# Create outdir
if drop_fibro:
    outdir = OUTDIR / 'tidy_drop_fibro'
else:
    outdir = OUTDIR / 'tidy'
os.makedirs(outdir, exist_ok=True)



# ========================================================================
#       Other settings
# ========================================================================
# sources = ['ccle', 'gcsi', 'gdsc', 'ctrp', 'nci60', 'chempartner']
sources = ['ccle', 'gcsi', 'gdsc', 'ctrp', 'nci60']
na_values = ['na', '-', '']
tidy_file_name = 'tidy_data'
tidy_file_format = 'parquet'

# Response columns
rsp_cols = ['AUC', 'AUC1', 'EC50', 'EC50se',
            'R2fit', 'Einf', 'IC50',
            'HS', 'AAC1', 'DSS1']

# Analysis of fibro samples are implemented in ccle_fibroblast.py and ccle_preproc.R
fibro_names = ['CCLE.HS229T', 'CCLE.HS739T', 'CCLE.HS840T', 'CCLE.HS895T', 'CCLE.RKN',
               'CTRP.Hs-895-T', 'CTRP.RKN', 'GDSC.RKN', 'gCSI.RKN']

# Prefix to add to feature names based on feature types
fea_prfx_dict = {'rna': 'cell_rna.', 'cnv': 'cell_cnv.',
                 'dsc': 'drug_dsc.', 'fng': 'drug_fng.',
                 'clb': 'cell_lbl.', 'dlb': 'drug_lbl.'}

prfx_dtypes = {'rna': np.float32,
               'cnv': np.int8,
               'dsc': np.float32,
               'fng': np.int8,
               'clb': str,
               'dlb': str}

# Dump args
args['outdir'] = outdir
args['sources'] = sources
utils.dump_args(args, outdir=outdir)



# ========================================================================
#       logger
# ========================================================================
logfilename = outdir/'tidy_data.log'
lg = Logger(logfilename=logfilename)
lg.logger.info(f'File path: {file_path}')
lg.logger.info(f'Mum of system cpus: {psutil.cpu_count()}')



# ========================================================================
#       Load response data
# ========================================================================
# Combined response
lg.logger.info(f'\n\nLoading combined response from ...\n{DATADIR / RSP_FILENAME}')
rsp = pd.read_table(DATADIR / RSP_FILENAME,
                    sep='\t', na_values=na_values,
                    dtype={'SOURCE': str, 'CELL': str, 'DRUG': str,
                           'AUC': np.float32, 'IC50': np.float32, 'EC50': np.float32,
                           'EC50se': np.float32, 'R2fit': np.float32, 'Einf': np.float32,
                           'HS': np.float32, 'AAC1': np.float32, 'AUC1': np.float32, 'DSS1': np.float32},
                    warn_bad_lines=True)

# Chempartner response
"""
lg.logger.info(f'Loading chempartner response ...\n{DATADIR / RSP_FILENAME_CHEM}')
rsp_chem = pd.read_table(DATADIR / RSP_FILENAME_CHEM,
                         sep='\t', na_values=na_values,
                         dtype={'SOURCE': str, 'CELL': str, 'DRUG': str,
                                'AUC': np.float32, 'IC50': np.float32, 'EC50': np.float32,
                                'EC50se': np.float32, 'R2fit': np.float32, 'Einf': np.float32,
                                'HS': np.float32, 'AAC1': np.float32, 'AUC1': np.float32, 'DSS1': np.float32},
                         warn_bad_lines=True)
rsp_chem['SOURCE'] = rsp_chem['SOURCE'].map(lambda x: x.split('_')[0])
"""

# Merge rsp from combined and chempartner
# rsp = pd.concat([rsp, rsp_chem], axis=0)

rsp['SOURCE'] = rsp['SOURCE'].apply(lambda x: x.lower())
lg.logger.info(f'rsp.shape {rsp.shape}')

# Replace -inf and inf with nan
rsp.replace([np.inf, -np.inf], value=np.nan, inplace=True)

if verbose:
    lg.logger.info('rsp memory usage: {:.3f} GB'.format(sys.getsizeof(rsp)/1e9))
    lg.logger.info('')
    lg.logger.info(rsp.groupby('SOURCE').agg({'CELL': 'nunique', 'DRUG': 'nunique'}).reset_index())
    


# ========================================================================
#   Drop fibroblast
# ========================================================================
if drop_fibro:
    lg.logger.info('\n\nDrop fibroblast samples ...')
    # rna = rna[rna['CELL'].map(lambda x: False if x in fibro_names else True)]
    # cmeta = cmeta[cmeta['CELL'].map(lambda x: False if x in fibro_names else True)]
    # rsp = rsp[rsp['CELL'].map(lambda x: False if x in fibro_names else True)]
    id_drop = rsp['CELL'].map(lambda x: True if x in fibro_names else False)
    rsp = rsp.loc[~id_drop,:]
    lg.logger.info(f'Dropped {sum(id_drop)} rsp data points.')
    lg.logger.info(f'rsp.shape {rsp.shape}')
    # lg.logger.info(f'rna.shape   {rna.shape}')
    # lg.logger.info(f'cmeta.shape {cmeta.shape}')
    
    
    
# ========================================================================
#   Drop 'bad' points (defined by Yitan)
# ========================================================================    
if drop_bad:
    lg.logger.info('\n\nDrop bad samples ...')
    id_drop = (rsp['AUC'] == 0) & (rsp['EC50se'] == 0) & (rsp['R2fit'] == 0)
    rsp = rsp.loc[~id_drop,:]
    lg.logger.info(f'Dropped {sum(id_drop)} rsp data points.')
    lg.logger.info(f'rsp.shape {rsp.shape}')  
    
    
    
# ========================================================================
#   load rna (combined_dataset)
# ========================================================================
lg.logger.info('\n\nLoading rna-seq ...')
lincs = CombinedRNASeqLINCS(
    datadir=DATADIR,
    cellmeta_fname=CELLMETA_FILENAME,
    dataset='raw',
    sources=sources,
    na_values=na_values,
    verbose=False)
rna, cmeta = lincs._df_rna, lincs._meta
rna.rename(columns={'Sample': 'CELL'}, inplace=True)
cmeta.rename(columns={'Sample': 'CELL', 'source': 'SOURCE'}, inplace=True)

# Add prefix to rna features
rna = rna.rename(columns={c: fea_prfx_dict['rna']+c for c in rna.columns[1:] if fea_prfx_dict['rna'] not in c}) # add genes
rna.insert(loc=1, column=fea_prfx_dict['clb']+'CELL', value=rna['CELL']) # add cell label
lg.logger.info(f'rna.shape {rna.shape}')

# Impute missing values
rna = utils_tidy.impute_values(data=rna, fea_prfx_dict=fea_prfx_dict, logger=lg.logger)

if verbose:
    lg.logger.info('rna memory usage: {:.3f} gb'.format(sys.getsizeof(rna)/1e9))
    lg.logger.info('')
    lg.logger.info(cmeta.groupby('SOURCE').agg({'CELL': 'nunique', 'ctype': 'nunique', 'csite': 'nunique'}).reset_index())
    # print(cmeta.groupby(['SOURCE', 'csite']).agg({'CELL': 'nunique'}).reset_index())


# ========================================================================
#   Load drug descriptors
# ========================================================================
lg.logger.info(f'\n\nLoading drug descriptors from ...\n{DATADIR / DSC_FILENAME}')
path = DATADIR / DSC_FILENAME
cols = pd.read_table(path, engine='c', nrows=0)
dtype_dict = {c: prfx_dtypes['dsc'] for c in cols.columns[1:]}
dsc = pd.read_table(path, dtype=dtype_dict, na_values=na_values, warn_bad_lines=True)
# dsc.rename(columns={'NAME': 'PUBCHEM'}, inplace=True)  # used in the old code
dsc.rename(columns={'NAME': 'DRUG'}, inplace=True)

# Add prefix to drug features
dsc = dsc.rename(columns={c: fea_prfx_dict['dsc']+c for c in dsc.columns[1:] if fea_prfx_dict['dsc'] not in c}) # descriptors
dsc.insert(loc=1, column=fea_prfx_dict['dlb']+'DRUG', value=dsc['DRUG']) # add drug label
lg.logger.info(f'dsc.shape {dsc.shape}')


# ------------------
# Filter descriptors
# ------------------
# dsc.nunique(dropna=True).value_counts()
# dsc.nunique(dropna=True).sort_values()

lg.logger.info('Drop descriptors with *lots* of NA values ...')
utils.plot_dsc_na_dist(dsc=dsc, savepath=outdir/'dsc_hist_ratio_of_na.png')
dsc = utils.dropna(df=dsc, axis=1, th=dropna_thres)
lg.logger.info(f'dsc.shape {dsc.shape}')
# dsc.isna().sum().sort_values(ascending=False)

# There are descriptors for which there is a single unique value excluding NA (drop those)
lg.logger.info('Drop descriptors that have a single unique value (excluding NAs) ...')
col_idx = dsc.nunique(dropna=True).values==1
dsc = dsc.iloc[:, ~col_idx]
lg.logger.info(f'dsc.shape {dsc.shape}')


# there are still lots of descriptors which have only a few unique values
# we can categorize those values. e.g.: 564 descriptors have only 2 unique vals,
# and 154 descriptors have only 3 unique vals, etc.
# todo: use utility code from p1h_alex/utils/data_preproc.py that transform those
# features into categorical and also applies an appropriate imputation.
# dsc.nunique(dropna=true).value_counts()[:10]
# dsc.nunique(dropna=true).value_counts().sort_index()[:10]

# Impute missing values
dsc = utils_tidy.impute_values(data=dsc, fea_prfx_dict=fea_prfx_dict, logger=lg.logger)

# Drop low var cols
# tmp, idx = utils_all.drop_low_var_cols(df=dsc, skipna=False)

if verbose:
    lg.logger.info('dsc memory usage: {:.3f} GB'.format(sys.getsizeof(dsc)/1e9))
    

# ========================================================================
#   Load drug meta
# ========================================================================
""" We don't need drug meta with the new descriptors file. """
# logger.info(f'\n\nLoading drug metadata ... {DRUG_META_FILENAME}')
# dmeta = pd.read_table(os.path.join(DATADIR, DRUG_META_FILENAME), dtype=object)
# dmeta['PUBCHEM'] = 'PubChem.CID.' + dmeta['PUBCHEM']
# dmeta.insert(loc=0, column='SOURCE', value=dmeta['ID'].map(lambda x: x.split('.')[0].lower()))
# dmeta.rename(columns={'ID': 'DRUG'}, inplace=True)
# logger.info(f'dmeta.shape {dmeta.shape}')

# if verbose:
#     # Number of unique drugs in each data source
#     # TODO: What's going on with CTRP and GDSC? Why counts are not consistent across the fields??
#     logger.info('')
#     logger.info(dmeta.groupby('SOURCE').agg({'DRUG': 'nunique', 'NAME': 'nunique',
#                                              'CLEAN_NAME': 'nunique', 'PUBCHEM': 'nunique'}).reset_index())



# ========================================================================
#   Merge the data
# ========================================================================
"""
Data tables: rsp, rna, dsc, cmeta, dmeta
(rsp, rna): on 'CELL'
(rsp, dsc): on pubchem through fields in dmeta
"""
lg.logger.info('\n\n{}'.format('-'*40))
lg.logger.info('... Start merging dataframes ...')
lg.logger.info('{}'.format('-'*40))

# Update rsp with additional drug field 'PUBCHEM' (this will be used to merge with descriptors)
""" No need dmeta with the new descriptors file. """
# logger.info('\nMerge response (rsp) with drug metadata (dmeta) on DRUG in order to add PUBCHEM (required for descriptors) ...')
# logger.info(f'rsp.shape   {rsp.shape}')
# logger.info(f'dmeta.shape {dmeta.shape}')
# rsp = pd.merge(rsp, dmeta[['DRUG', 'PUBCHEM']], on='DRUG', how='left')
# logger.info(f'rsp.shape   {rsp.shape}')
# logger.info('NA values after merging rsp and dmeta: \n{}'.format(rsp[['DRUG', 'PUBCHEM']].isna().sum()))
# logger.info('')
# logger.info(rsp.groupby('SOURCE').agg({'DRUG': 'nunique', 'PUBCHEM': 'nunique'}).reset_index())


# --------------------
# Merge rsp with cmeta
# --------------------
lg.logger.info('\nMerge response (rsp) and cell metadata (cmeta) ...')
lg.logger.info(f'rsp.shape   {rsp.shape}')
lg.logger.info(f'cmeta.shape {cmeta.shape}')
rsp1 = pd.merge(rsp, cmeta[['CELL', 'core_str', 'csite', 'ctype', 'simplified_csite', 'simplified_ctype']], on='CELL', how='left')
lg.logger.info(f'rsp1.shape  {rsp1.shape}')
lg.logger.info('')
# logger.info(rsp1.groupby('SOURCE').agg({'CELL': 'nunique', 'DRUG': 'nunique', 'PUBCHEM': 'nunique'}).reset_index())
lg.logger.info(rsp1.groupby('SOURCE').agg({'CELL': 'nunique'}).reset_index())
del rsp

# --------------
# Merge with rna
# --------------
lg.logger.info('\nMerge with expression (rna) ...')
lg.logger.info(f'rsp1.shape {rsp1.shape}')
lg.logger.info(f'rna.shape  {rna.shape}')
rsp2 = pd.merge(rsp1, rna, on='CELL', how='inner')
lg.logger.info(f'rsp2.shape {rsp2.shape}')
lg.logger.info('')
# logger.info(rsp2.groupby('SOURCE').agg({'CELL': 'nunique', 'DRUG': 'nunique', 'PUBCHEM': 'nunique'}).reset_index())
lg.logger.info(rsp2.groupby('SOURCE').agg({'CELL': 'nunique'}).reset_index())
del rsp1

# --------------
# Merge with dsc
# --------------
lg.logger.info('\nMerge with descriptors (dsc) ...')
lg.logger.info(f'rsp2.shape {rsp2.shape}')
lg.logger.info(f'dsc.shape  {dsc.shape}')
# data = pd.merge(rsp2, dsc, on='PUBCHEM', how='inner')
data = pd.merge(rsp2, dsc, on='DRUG', how='inner')
lg.logger.info(f'data.shape {data.shape}')
lg.logger.info('')
# logger.info(data.groupby('SOURCE').agg({'CELL': 'nunique', 'DRUG': 'nunique', 'PUBCHEM': 'nunique'}).reset_index())
lg.logger.info(data.groupby('SOURCE').agg({'CELL': 'nunique', 'DRUG': 'nunique'}).reset_index())
del rsp2


# ----------------------
# Add labels as features
# ----------------------
# data = data.rename(columns={'cell': fea_prfx_dict['clb'], 'drug': fea_prfx_dict['clb'])
# data[fea_prfx_dict['clb']+'CELL'] = data['CELL']
# data[fea_prfx_dict['dlb']+'DRUG'] = data['DRUG']


# Summary of memory usage
lg.logger.info('\nTidy dataframe: {:.3f} GB'.format(sys.getsizeof(data)/1e9))
if verbose:
    for prfx in prfx_dtypes.keys():
        cols = [c for c in data.columns if prfx in c]
        tmp = data[cols]
        mem = 0 if tmp.shape[1]==0 else sys.getsizeof(tmp)/1e9
        lg.logger.info("# of '{}' features: {} ({:.2f} GB)".format(prfx, len(cols), mem))


# Cast features
# https://stackoverflow.com/questions/15891038/change-data-type-of-columns-in-pandas
# for fea_prfx, fea_frmt in prfx_dtypes.items():
#     print(f'feature type and format: ({fea_prfx}, {fea_frmt})')
#     dict_types = {c: fea_frmt for c in tmp.columns if fea_prfx in c}
#     if len(dict_types) > 0:
#         data = data.astype(dict_types)

lg.logger.info('\nEnd of data per-processing: {:.1f} mins'.format( (time()-t0)/60) )



# ========================================================================
#   Plot rsp distributions
# ========================================================================
# Plot distributions of target variables
utils.plot_rsp_dists(rsp=data, rsp_cols=rsp_cols, savepath=outdir/'rsp_dists.png')

# Plot distribution of a single target
# target_name = 'EC50se'
# fig, ax = plt.subplots()
# x = rsp[target_name].copy()
# x = x[~x.isna()].values
# sns.distplot(x, bins=100, ax=ax)
# plt.savefig(os.path.join(OUTDIR, target_name+'.png'), bbox_inches='tight') 



# ========================================================================
#   Finally save data
# ========================================================================
data.drop(columns='STUDY', inplace=True) # gives error when save in 'parquet' format


# Save data
lg.logger.info(f'\nSave tidy dataframe ({tidy_file_format}) ...')
t0 = time()
if tidy_file_format == 'parquet':
    tidy_filepath = outdir / (tidy_file_name + '.parquet')
    data.to_parquet(tidy_filepath, engine='auto', compression='snappy')
else: 
    tidy_filepath = outdir / 'tidy_data'
    data.to_csv(tidy_filepath, sep='\t')
lg.logger.info('Save tidy dataframe to disk: {:.1f} mins'.format( (time()-t0)/60) )


# Load data
lg.logger.info(f'\nLoad tidy dataframe ({tidy_file_format}) ...')
t0 = time()
if tidy_file_format == 'parquet':
    data_fromfile = pd.read_parquet(tidy_filepath, engine='auto', columns=None)
else:
    data_fromfile = pd.read_table(tidy_filepath, sep='\t')
lg.logger.info('Load tidy dataframe: {:.1f} mins'.format( (time()-t0)/60) )


# Check that the saved data is the same as original one
lg.logger.info(f'\nLoaded dataframe is the same as original: {data.equals(data_fromfile)}')


lg.logger.info('\n{}'.format('-'*90))
lg.logger.info(f'Tidy data file path:\n{os.path.abspath(tidy_filepath)}')
lg.logger.info('{}'.format('-'*90))


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

