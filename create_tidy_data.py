import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import collections
import pyarrow.feather as feather

from scipy.stats import norm, skew

import warnings
warnings.filterwarnings('ignore')

# Utils
file_path = os.getcwd()
# os.path.dirname(os.path.relpath(__file__))
utils_path = os.path.abspath(os.path.join(file_path, 'utils_py'))
sys.path.append(utils_path)
import utils_all as utils
# import utils

DATADIR = '/Users/apartin/work/jdacs/Benchmarks/Data/Pilot1'
OUTDIR = os.path.join(file_path, 'tidy_data_from_combined')
os.makedirs(OUTDIR, exist_ok=True)

t0 = time.time()

# args
sources = ['ccle', 'gcsi', 'gdsc', 'ctrp']
drug_features = ['descriptors', 'fingerprints']
cell_features = ['rnaseq']
dropna_thres = 0.4

verbose = True
na_values = ['na', '-', '']
tidy_data_format = 'parquet'

drop_fibro = True
fibro_names = ['CCLE.HS229T', 'CCLE.HS739T', 'CCLE.HS840T', 'CCLE.HS895T', 'CCLE.RKN',
               'CTRP.Hs-895-T', 'CTRP.RKN', 'GDSC.RKN', 'gCSI.RKN']

prefix_dtypes = {'rna': np.float32,
                 'cnv': np.int8,
                 'dsc': np.float32,
                 'fng': np.int8}


# ======================
#   Load response data
# ======================
# filename = 'ChemPartner_single_response_agg'
filename = 'combined_single_response_agg'
print('\nLoading combined response ... {}'.format(filename))
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
print('rsp.shape', rsp.shape)

# Replace -Inf and Inf with nan
rsp.replace([np.inf, -np.inf], value=np.nan, inplace=True)

if verbose:
    # Unique items (cells, drugs) per data source
    print(rsp.groupby('SOURCE').agg({'CELL': 'nunique', 'DRUG': 'nunique'}).reset_index())
    # rsp.memory_usage(deep=True).sum()
    print('rsp memory usage (GB): {:.3f}'.format(sys.getsizeof(rsp)/1e9))

# target_name = 'EC50se'
# fig, ax = plt.subplots()
# x = rsp[target_name].copy()
# x = x[~x.isna()].values
# sns.distplot(x, bins=100, ax=ax)
# plt.savefig(os.path.join(OUTDIR, target_name+'.png'), bbox_inches='tight')

# Plot distributions of target variables
ncols = 4
nrows = int(np.ceil(len(rsp_cols)/ncols))
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=False, sharey=False)
for i, ax in enumerate(axes.ravel()):
    if i >= len(rsp_cols):
        fig.delaxes(ax) # delete un-used ax
    else:
        target_name = rsp_cols[i]
        # print(i, target_name)
        x = rsp[target_name].copy()
        x = x[~x.isna()].values
        sns.distplot(x, bins=100, kde=True, ax=ax, label=target_name, # fit=norm, 
                     kde_kws={'color': 'k', 'lw': 0.4, 'alpha': 0.8},
                     hist_kws={'color': 'b', 'lw': 0.4, 'alpha': 0.5})
        ax.tick_params(axis='both', which='major', labelsize=7)
        txt = ax.yaxis.get_offset_text(); txt.set_size(7) # adjust exponent fontsize in xticks
        txt = ax.xaxis.get_offset_text(); txt.set_size(7)
        ax.legend(fontsize=5, loc='best')

plt.tight_layout()
# plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
plt.savefig(os.path.join(OUTDIR, 'rsp_hist.png'), bbox_inches='tight', dpi=300)


# ====================================
#   Load rna (combined_dataset)
# ====================================
print('\nLoading rna-seq ... ')
lincs = utils.CombinedRNASeqLINCS(dataset='raw', sources=sources, verbose=False)
rna, cellmeta = lincs._df_rna, lincs._meta
rna.rename(columns={'Sample': 'CELL'}, inplace=True)
cellmeta.rename(columns={'Sample': 'CELL', 'source': 'SOURCE'}, inplace=True)
rna_prefix = 'cell_rna.' # 'rna.'
rna = rna.rename(columns={c: rna_prefix+c for c in rna.columns[1:] if rna_prefix not in c})
print('rna.shape', rna.shape)
# print(rna.iloc[:2, :5])

if verbose:
    # Unique items (cells, drugs) per data source
    print(cellmeta.groupby('SOURCE').agg({'CELL': 'nunique', 'ctype': 'nunique', 'csite': 'nunique'}).reset_index())
    # print(cellmeta.groupby(['SOURCE', 'csite']).agg({'CELL': 'nunique'}).reset_index())
    # rna.memory_usage(deep=True).sum()
    print('rna memory usage (GB): {:.3f}'.format(sys.getsizeof(rna)/1e9))


# =========================
#   Load drug descriptors
# =========================
filename = 'Combined_PubChem_dragon7_descriptors.tsv'
print('\nLoading drug descriptors ... {}'.format('Combined_PubChem_dragon7_descriptors.tsv'))
path = os.path.join(DATADIR, filename)
cols = pd.read_table(path, engine='c', nrows=0)
dtype_dict = {c: np.float32 for c in cols.columns[1:]}
dsc = pd.read_table(path, dtype=dtype_dict, na_values=na_values, warn_bad_lines=True)
dsc.rename(columns={'NAME': 'PUBCHEM'}, inplace=True)
dsc_prefix = 'drug_dsc.' # 'dsc.'
dsc = dsc.rename(columns={c: dsc_prefix+c for c in dsc.columns[1:] if dsc_prefix not in c})
print('dsc.shape', dsc.shape)
# print(dsc.iloc[:2, :5])


# ------------------
# Filter descriptors
# ------------------
# dsc.nunique(dropna=True).value_counts()
# dsc.nunique(dropna=True).sort_values()

print('Drop descriptors with *lots* of NA values ...')
# print('dsc.shape', dsc.shape)
fig, ax = plt.subplots()
sns.distplot(dsc.isna().sum(axis=0)/dsc.shape[0], bins=100, kde=False, hist_kws={'alpha': 0.7})
plt.xlabel('Ratio of NA values')
plt.title('Histogram of descriptors')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'dsc_hist_ratio_of_na.png'))
print('dsc.shape', dsc.shape)
dsc = utils.dropna(df=dsc, axis=1, th=dropna_thres)
print('dsc.shape', dsc.shape)
# dsc.isna().sum().sort_values(ascending=False)

# There are descriptors for which there is a single unique value excluding NA
# (drop those)
print('Drop descriptors with that have a single unique value (excluding NAs) ...')
col_idx = dsc.nunique(dropna=True).values==1
# tmp = dsc.iloc[:, col_idx]
print('dsc.shape', dsc.shape)
dsc = dsc.iloc[:, ~col_idx]
print('dsc.shape', dsc.shape)

# There are still lots of descriptors which have only a few unique values
# We can categorize those values. E.g.: 564 descriptors have only 2 unique vals,
# and 154 descriptors have only 3 unique vals, etc.
# TODO: use utility code from p1h_alex/utils/data_preproc.py that transform those
# features into categorical and also applies an appropriate imputation.
# dsc.nunique(dropna=True).value_counts()[:10]
# dsc.nunique(dropna=True).value_counts().sort_index()[:10]

# Impute missing values
# TODO: 

# Drop low var cols
# tmp, idx = utils.drop_low_var_cols(df=dsc, skipna=False)

if verbose:
    # rna.memory_usage(deep=True).sum()
    print('dsc memory usage (GB): {:.3f}'.format(sys.getsizeof(dsc)/1e9))
    

# ==================
#   Load drug meta
# ==================
filename = 'drug_info'
print('\nLoading drug metadata ... {}'.format(filename))
dmeta = pd.read_table(os.path.join(DATADIR, filename), dtype=object)
dmeta['PUBCHEM'] = 'PubChem.CID.' + dmeta['PUBCHEM']
# dmeta['Drug'] = dmeta['PUBCHEM']
dmeta.insert(loc=0, column='SOURCE', value=dmeta['ID'].map(lambda x: x.split('.')[0].lower()))
# dmeta.drop(columns=['Drug'], inplace=True)
dmeta.rename(columns={'ID': 'DRUG'}, inplace=True)
print(dmeta.shape, 'dmeta.shape')
# print(dmeta.iloc[:2, :5])

if verbose:
    # Number of unique drugs in each data source
    # TODO: What's going on with CTRP and GDSC? Why counts are not consistent across the fields??
    print(dmeta.groupby('SOURCE').agg({'DRUG': 'nunique', 'NAME': 'nunique',
                                    'CLEAN_NAME': 'nunique', 'PUBCHEM': 'nunique'}).reset_index())


# ===================
#   Drop fibroblast
# ===================
if drop_fibro:
    print('\nDrop fibroblast samples ...')
    # print('rna.shape', rna.shape)
    # print('cellmeta.shape', cellmeta.shape)
    # print('rsp.shape', rsp.shape)
    rna = rna[rna['CELL'].map(lambda x: False if x in fibro_names else True)]
    cellmeta = cellmeta[cellmeta['CELL'].map(lambda x: False if x in fibro_names else True)]
    rsp = rsp[rsp['CELL'].map(lambda x: False if x in fibro_names else True)]
    # print('rna.shape', rna.shape)
    # print('cellmeta.shape', cellmeta.shape)
    # print('rsp.shape', rsp.shape)


# ==================
#   Merge the data
# ==================
"""
Data tables: rsp, rna, dsc, cellmeta, dmeta
(rsp, rna): on 'CELL'
(rsp, dsc): on pubchem through fields in dmeta
"""
# print('rsp.shape', rsp.shape)
# print('rna.shape', rna.shape)
# print('dsc.shape', dsc.shape)
# print(rsp[:2])
# print(rna.iloc[:2, :5])
# print(dsc.iloc[:2, :5])


# Drop response for those drugs that don't have descriptors
# print("\nNote that some drugs don't have descriptors:")
# print('NA values in dmeta: \n{}'.format(dmeta[['DRUG', 'PUBCHEM']].isna().sum()))
# if 'descriptors' in drug_features:
#     dmeta = dmeta[~(dmeta['PUBCHEM'].isna().astype(bool))]


# Update rsp with additional drug field 'PUBCHEM' (this will be used to merge with descriptors)
print('\nMerge response (rsp) with drug metadata (dmeta) on DRUG in roder to add PUBCHEM (required for descriptors) ...')
print(rsp.shape)
rsp = pd.merge(rsp, dmeta[['DRUG', 'PUBCHEM']], on='DRUG', how='left')
print(rsp.shape)
print('NA values after merging rsp and dmeta: \n{}'.format(rsp[['DRUG', 'PUBCHEM']].isna().sum()))

if verbose:
    print(rsp.groupby('SOURCE').agg({'DRUG': 'nunique', 'PUBCHEM': 'nunique'}).reset_index())


# # Extract response for sources
# rsp = rsp[~(rsp['AUC'].isna() | rsp['AUC1'].isna())]
# rsp = rsp[rsp['SOURCE'].isin(sources)]
# # rsp['CELL'] = rsp['CELL'].map(lambda s: ''.join([x for i, x in enumerate(s.split('.')) if i>0]))
# print(rsp.shape)
# print(rsp.SOURCE.value_counts())

# -----------------------
# Merge rsp with cellmeta
# -----------------------
print('\nMerge response (rsp) and cell metadata (cellmeta) ...')
print(rsp.shape)
rsp1 = pd.merge(rsp, cellmeta[['CELL', 'core_str', 'csite', 'ctype', 'simplified_csite', 'simplified_ctype']], on='CELL', how='left')
print(rsp1.shape)
print(rsp1.iloc[:2, :10])
print(rsp1.groupby('SOURCE').agg({'CELL': 'nunique', 'DRUG': 'nunique',
                                  'PUBCHEM': 'nunique'}).reset_index())
del rsp

# --------------
# Merge with rna
# --------------
print('\nMerge with expression (rna) ...')
print(rsp1.shape)
rsp2 = pd.merge(rsp1, rna, on='CELL', how='inner')
print(rsp2.shape)
print(rsp2.iloc[:2, :10])
print(rsp2.groupby('SOURCE').agg({'CELL': 'nunique', 'DRUG': 'nunique',
                                  'PUBCHEM': 'nunique'}).reset_index())
del rsp1

# --------------
# Merge with dsc
# --------------
print('\nMerge with descriptors (dsc) ...')
print('rsp2.shape', rsp2.shape)
print('rsp2.shape', dsc.shape)
# print(df.iloc[:2, :10])
# print(dsc.iloc[:2, :10])
data = pd.merge(rsp2, dsc, on='PUBCHEM', how='inner')
print(data.shape)
# print(data.iloc[:2, :20])
print(data.groupby('SOURCE').agg({'CELL': 'nunique', 'DRUG': 'nunique',
                                  'PUBCHEM': 'nunique'}).reset_index())
del rsp2


# Summary
print('\nTidy dataframe memory usage (GB): {:.3f}'.format(sys.getsizeof(data)/1e9))
if verbose:
    for prfx in prefix_dtypes.keys():
        cols = [c for c in data.columns if prfx in c]
        tmp = data[cols]
        mem = 0 if tmp.shape[1]==0 else sys.getsizeof(tmp)/1e9
        print("Number of '{}' features: {} ({:.3f} GB memory)".format(prfx, len(cols), mem))


# Cast features
# https://stackoverflow.com/questions/15891038/change-data-type-of-columns-in-pandas
# for fea_prfx, fea_frmt in prefix_dtypes.items():
#     print(f'feature type and format: ({fea_prfx}, {fea_frmt})')
#     dict_types = {c: fea_frmt for c in tmp.columns if fea_prfx in c}
#     if len(dict_types) > 0:
#         data = data.astype(dict_types)

print('\nEnd of data per-processing: {:.2f} mins'.format((time.time()-t0)/60))


# =====================
#   Finally save data
# =====================
# data.to_csv(os.path.join(OUTDIR, 'tidy_data.txt'), sep='\t', float_format=np.float32, index=False)
print('\nSave tidy dataframe ...')
t0 = time.time()
data.drop(columns='STUDY', inplace=True) # gives error when save in 'parquet' format
if tidy_data_format == 'parquet':
    data.to_parquet(os.path.join(OUTDIR, 'tidy_data.prqt'), engine='auto', compression='snappy')
else: 
    data.to_csv(os.path.join(OUTDIR, 'tidy_data'), sep='\t')
# data.to_feather(os.path.join(OUTDIR, 'tidy_data'))
# feather.write_feather(data, os.path.join(OUTDIR, 'tidy_data'))
# data.to_hdf(os.path.join(OUTDIR, 'tidy_data'), key='df')
print('Time to save tidy data to disk: {:.2f} mins'.format((time.time()-t0)/60))

# Check that feather-saved data is the same as the original one
print('\nLoad tidy dataframe (feather) ...')
t0 = time.time()
if tidy_data_format == 'parquet':
    data_fromfile = pd.read_parquet(os.path.join(OUTDIR, 'tidy_data.prqt'), engine='auto', columns=None)
else:
    data_fromfile = pd.read_table(os.path.join(OUTDIR, 'tidy_data'), sep='\t')
# data_fromfile = feather.read_feather(os.path.join(OUTDIR, 'tidy_data'))
# data_fromfile = pd.read_feather(os.path.join(OUTDIR, 'tidy_data'))
# data_fromfile = pd.read_hdf(os.path.join(OUTDIR, 'tidy_data'))
print('Time to load tidy data to disk: {:.2f} mins'.format((time.time()-t0)/60))

print('\nLoaded data is the same as original: ', data.equals(data_fromfile))



# ==========================
#  EDA
# ==========================
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
# rna, meta = utils.load_lincs1000(dataset='raw', sources=['ccle', 'ctrp', 'gdsc', 'gcsi', 'nci60'])
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
# # utils.contingency_table(mm, cols=['SOURCE', 'simplified_csite'], to_plot=True, figsize=None, title=None, margins=False, normalize=False)



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

