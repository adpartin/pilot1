"""
Alex Partin.

This code compares the available CCLE cell lines from the original source (Broad Institute)
and our combined datased (combined_dataset). Specifically, look at the cell line for which
we have both the rna-seq and the drug response.

Results:
- Combined: 474 samples
- Source (Broad): 472 samples
1. In combined but not in source: {'HS895T', 'HS739T', 'HS229T', 'HS840T', 'RKN'} -->
   These 5 missing samples that appear in the combined dataset are fibroblast and have
   been filtered when I merged response and rna from the CCLE source dataset.
2. In source but not in combined: {'COV504', 'OC316', 'COLO699'} -->
   These 3 samples are the new samples that were sequenced (downloaded in Nov 2018).
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import collections

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
OUTDIR = 'ccle_fibroblast'
os.makedirs(OUTDIR, exist_ok=True)


# ==========================================================================================
# Combined dataset (our data)
# ==========================================================================================
# Load response data
filename = 'combined_single_response_agg'
print('\nLoading combined response ... {}'.format(filename))
# rsp_chem = pd.read_csv(os.path.join(DATADIR, 'ChemPartner_single_response_agg'), sep='\t')
rsp1 = pd.read_csv(os.path.join(DATADIR, filename), sep='\t')
print('rsp1.shape', rsp1.shape)
# print(rsp1[:2])

# Load lincs rna from combined_dataset
print('\nLoading combined rna-seq ...')
lincs = utils.CombinedRNASeqLINCS(dataset='raw', sources='CCLE', verbose=False)
df1, meta1 = lincs._df_rna, lincs._meta
df1.rename(columns={'Sample': 'CELL'}, inplace=True)
meta1.rename(columns={'Sample': 'CELL'}, inplace=True)
print('df1.shape', df1.shape)

# Cells for which we have rna and response
print('\nMerging rsp1 and df1 ...')
df1 = pd.merge(df1, rsp1[['CELL']].drop_duplicates(), on='CELL', how='inner')
meta1 = pd.merge(meta1, rsp1[['CELL']].drop_duplicates(), on='CELL', how='inner')
meta1 = meta1[['CELL', 'source', 'core_str', 'csite', 'ctype', 'simplified_csite', 'simplified_ctype']]
print('df1.shape', df1.shape)
print('meta1.shape', meta1.shape)

# Samples screened but not sequenced
# Note that we can see here the 3 missing samples {'COV504', 'OC316', 'COLO699'}
tmp = rsp1[rsp1['SOURCE']=='CCLE'].copy()
print('\nSamples that were drug sceened but not sequenced (as it appears in the combined dataset): {}'.format(
    len(sorted(list(set(tmp['CELL'].unique()).difference(set(df1['CELL'])))))
))
print(sorted(list(set(tmp['CELL'].unique()).difference(set(df1['CELL'])))))

# # Combined meta (I get the same meta from utils.CombinedRNASeqLINCS)
# combined_meta = pd.read_csv(os.path.join(DATADIR, 'combined_cl_metadata'), sep='\t')
# combined_meta = combined_meta[combined_meta['dataset'].isin(['CCLE'])]
# print(combined_meta.shape)
# combined_meta = combined_meta['core_str', 'tumor_site_from_data_src', 'tumor_type_from_data_src', 'simplified_tumor_site', 'simplified_tumor_type']

# Clean the name to match with df1
df1['CELL'] = df1['CELL'].map(lambda x: x.split('.')[1])
meta1['CELL'] = meta1['CELL'].map(lambda x: x.split('.')[1])


# ==========================================================================================
# Load the original CCLE dataset (count data)
# ==========================================================================================
def get_meta_broad(cellnames):
    cell_types = pd.DataFrame(data={'CELL': [c.split('_')[0] for c in cellnames],
                                    'TYPE': ['_'.join(c.split('_')[1:]) for c in cellnames],
                                    'MISC': [''.join(c.split(' ')[-1]) for c in cellnames]})
    cell_types['TYPE'] = cell_types['TYPE'].map(lambda x: x.split(' ')[0])
    return cell_types

# Load response data
print('\nLoading orignal CCLE response ...')
filename = "CCLE_NP24.2009_Drug_data_2015.02.24.csv"
rsp2 = pd.read_csv(os.path.join("/Users/apartin/work/jdacs/cell-line-data/ccle/from_broad_institute/pharmacological_profiling", filename))
colNameMap = {'CCLE Cell Line Name': 'CELL', 'Primary Cell Line Name': 'CellName', 'Compound': 'Drug',
              'Doses (uM)': 'Doses_uM', 'Activity Data (median)': 'ActivityMedian', 'Activity SD': 'ActivitySD',
              'Num Data': 'nDataPoints', 'EC50 (uM)': 'EC50um', 'IC50 (uM)': 'IC50um'}
rsp2.rename(columns=colNameMap, inplace=True)
print('rsp2.shape', rsp2.shape)

# Load rnaseq
print('\nLoading original CCLE rna-seq ...')
filename = "CCLE_DepMap_18q3_RNAseq_reads_20180718.gct.txt"
df2 = pd.read_csv(os.path.join("/Users/apartin/work/jdacs/cell-line-data/ccle/from_broad_institute/current_data_11-08-2018",
                               filename), header=2, sep='\t')

# Rename cell line names (cols) and gene names
df2 = df2.rename(columns={c: c.split(' ')[0] for c in df2.columns[2:]})
df2 = df2.rename(columns={'Name': 'ENSGName', 'Description': 'GeneName'})
df2.drop(columns=['GeneName'], inplace=True)
df2['ENSGName'] = df2['ENSGName'].map(lambda x: x.split('.')[0])
df2 = df2.T
df2.columns = df2.iloc[0, :]
df2 = df2.iloc[1:, :]
df2.columns.name = None
df2.reset_index(inplace=True)
df2.rename(columns={'index': 'CELL'}, inplace=True)
df2 = df2.sort_values('CELL').reset_index(drop=True)
print('df2.shape', df2.shape)

# Extract fibroblast cell lines
fibro_rna = df2[[True if 'fibro' in c.lower() else False for c in df2.CELL]]
print('fibro_rna.shape', fibro_rna.shape)

# Cells for which we have rna and response
print('\nMerging rsp2 and df2 ...')
df2 = pd.merge(df2, rsp2[['CELL']].drop_duplicates(), on='CELL', how='inner')
print('df2.shape', df2.shape)

# Samples screened but not sequenced
print('\nSamples that were drug sceened but not sequenced (as of the data download date): {}'.format(
    len(sorted(list(set(rsp2['CELL'].unique()).difference(set(df2['CELL'])))))
))
print(sorted(list(set(rsp2['CELL'].unique()).difference(set(df2['CELL'])))))

# Clean the name the match with df1
meta2 = get_meta_broad(cellnames=df2['CELL'].tolist())
df2['CELL'] = df2['CELL'].map(lambda x: x.split('_')[0])

# This is to understand how we got included the fibroblast samples in our combined_dataset
# df2_ = df2.copy()
# rsp2_ = rsp2.copy()
# df2_['CELL'] = df2_['CELL'].map(lambda x: x.split('_')[0])
# rsp2_[['CELL']] = rsp2_['CELL'].map(lambda x: x.split('_')[0])
# df2_ = pd.merge(df2_, rsp2_[['CELL']].drop_duplicates(), on='CELL', how='inner')
# print('df2_.shape', df2_.shape)


# ==========================================================================================
# Now compare what's missing
# ==========================================================================================
# Samples that appear in our data but not in original
cells1 = sorted(set(df1.CELL.tolist()).difference(set(df2.CELL.tolist())))
print('\nCells that are in combined but not in source: \n{}'.format(cells1))

# Samples that appear in original but not in our data
cells2 = sorted(set(df2.CELL.tolist()).difference(set(df1.CELL.tolist())))
print('\nCells that are in source but not in combined: \n{}'.format(cells2))

dd1 = meta1[meta1['CELL'].isin(cells1)].reset_index(drop=True)
dd2 = meta2[meta2['CELL'].isin(cells2)].reset_index(drop=True)

print('\nMissing in source (these are fibroblast) ...')
print('{}'.format(dd1))

print('\nMissing in combined_dataset ...')
print('{}'.format(dd2))


# ==========================================================================================
# Get the list of fibroblast samples and their mappings to other data sources
# ==========================================================================================
cl_mapping = pd.read_csv(os.path.join(DATADIR, 'cl_mapping'), header=None, sep='\t')
cl_mapping.columns = ['src', 'dst']
print('\ncl_mapping.shape', cl_mapping.shape)
cells1 = ['CCLE.'+c for c in cells1]
fibro = cl_mapping[cl_mapping['src'].map(lambda s: True if s in cells1 else False)]
fibro = fibro.values.ravel().tolist()
fibro_names_all = sorted(list(set(fibro).union(set(cells1))))
print('\nFibroblast samples and their mappings: \n{}\n'.format(fibro_names_all))

rsp_fibro_samples = rsp1[rsp1['CELL'].isin(fibro_names_all)]
print('rsp_fibro_samples.shape', rsp_fibro_samples.shape)
rsp_clean_samples = rsp1[rsp1['CELL'].map(lambda s: True if s not in fibro_names_all else False)]
print('rsp_clean_samples.shape', rsp_clean_samples.shape)

plt.ax = plt.subplots()
# sns.kdeplot(rsp_fibro['AUC'], shade=True, label='AUC', legend=True)
# sns.kdeplot(rsp_fibro['AUC1'], shade=True, label='AUC', legend=True)
sns.distplot(rsp_fibro_samples['AUC'], bins=100, kde=True, label='AUC')
sns.distplot(rsp_fibro_samples['AUC1'], bins=100, kde=True, label='AUC1')
plt.title('Fibroblast; total samples: {}'.format(len(rsp_fibro_samples)))
plt.legend()
plt.grid()
plt.savefig(os.path.join(OUTDIR, 'fibroblast_samples_response.png'))


# ==========================================================================================
# Response for the newly sequenced samples
# ==========================================================================================
lst = dd2['CELL'].map(lambda s: 'CCLE.'+s).values.ravel().tolist()
rsp_new_samples = rsp1[rsp1['CELL'].isin(lst)]
print('rsp_new_samples.shape', rsp_new_samples.shape)

plt. ax = plt.subplots()
sns.distplot(rsp_new_samples['AUC'], bins=100, kde=True, label='AUC')
sns.distplot(rsp_new_samples['AUC1'], bins=100, kde=True, label='AUC1')
plt.title('New samples; total samples: {}'.format(len(rsp_new_samples)))
plt.legend()
plt.grid()
plt.savefig(os.path.join(OUTDIR, 'new_samples_response.png'))


# # ==========================================================================================
# # Load the CCLE dataset from original (count data) --> processed in R
# # ==========================================================================================
# # Load DESeq2 ccle data
# dfr = pd.read_csv(os.path.join(file_path, 'ccle_preproc/qdata_ccle_vsd.txt'), sep='\t')
# dfr.columns = [c.split('_')[0] for c in dfr.columns]
# dfr = dfr[sorted(dfr.columns)]
# dfr = dfr.T.reset_index().rename(columns={'index': 'CELL'})
# print(dfr.shape)
# print(dfr[:2])

# # The number of ccle cell lines doens't match from the original source and our data
# # There are some more analysis in R
# print(dfp.shape)
# print(dfr.shape)

# # Samples that appear in our data but not in original
# print(set(dfp.CELL.tolist()).difference(set(dfr.CELL.tolist())))

# # Samples that appear in original but not in our data
# print(set(dfr.CELL.tolist()).difference(set(dfp.CELL.tolist())))

