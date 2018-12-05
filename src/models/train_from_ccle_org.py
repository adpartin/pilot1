"""
Not finished!!
TODO:
- Need to make a fair comparison btw vsd and rpkm/tpm. There are two options:
    1. Compare vsd vs rpkm values provided by ccle. In this case, need to load
       rpkm values and create another tidy dataset (probably in R).
    2. Compare vsd vs tpm values from combined. In this case, need to make sure
       cell and drug mappings are correct in order to use the correct response values.
- Add descriptors to the tidy data.
- Add prefix to features.
- Add distribution plots of target values to R code.
"""
from __future__ import division
from __future__ import print_function

import warnings
warnings.filterwarnings('ignore')

import os
import sys
import time
import datetime
import logging
import argparse
import psutil
from collections import OrderedDict
import numpy as np
import pandas as pd

import matplotlib
# matplotlib.use('TkAgg')
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats
from sklearn.preprocessing import Imputer, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split, learning_curve, KFold, StratifiedKFold

# Utils
file_path = os.getcwd()
file_path = os.path.join(file_path, 'src/models')
os.chdir(file_path)

# file_path = os.path.dirname(os.path.realpath(__file__))  # os.path.dirname(os.path.abspath(__file__))
import utils_models as utils

DATADIR = os.path.join(file_path, '../../data/processed/from_ccle_org')
OUTDIR = os.path.join(file_path, '../../models/from_ccle_org')
FILENAME = 'tidy_data_lincs_ccle_vsd.txt'
os.makedirs(OUTDIR, exist_ok=True)

SEED = 0


# ========================================================================
#       Args TODO: add to argparse
# ========================================================================
# Train and infer data
train_sources = ['ccle']  # ['ccle', 'gcsi', 'gdsc', 'ctrp']
infer_sources = ['ccle']

# Traget (response)
# target_name: 'EC50um', 'IC50um', 'Amax', 'ActArea'
rsp_cols = ['EC50um', 'IC50um', 'Amax', 'ActArea']
target_name = 'ActArea'

# Features
cell_features = ['rnaseq'] # ['rnaseq', cnv', 'rnaseq_latent']
drug_features = ['descriptors'] # [] # ['descriptors', 'fingerprints', 'descriptors_latent', 'fingerprints_latent']
other_features = [] # ['drug_labels'] # ['cell_labels', 'drug_labels', 'ctype', 'csite', 'rna_clusters']

# Models
# ml_models = ['tpot_reg']
ml_models = ['lgb_reg']

trasform_target = False
verbose = True
n_jobs = 4

# Feature prefix
# fea_prefix = {'rnaseq': 'cell_rna',
#               'cnv': 'cell_cnv',
#               'descriptors': 'drug_dsc',
#               'fingerprints': 'drug_fng'}


# ========================================================================
#       Logger
# ========================================================================
t = datetime.datetime.now()
t = [t.year, '-', t.month, '-', t.day, '_', 'h', t.hour, '-', 'm', t.minute]
t = ''.join([str(i) for i in t])
run_outdir = os.path.join(OUTDIR, 'run_'+t)
os.makedirs(run_outdir)
logfilename = os.path.join(run_outdir, 'logfile.log')
logger = utils.setup_logger(logfilename=logfilename)

logger.info(f'File path: {file_path}')
logger.info(f'Num of system CPUs: {psutil.cpu_count()}')
logger.info(f'n_jobs: {n_jobs}')


# ========================================================================
#       Load data and pre-proc
# ========================================================================
datapath = os.path.join(DATADIR, FILENAME)
logger.info(f'\nLoad tidy data ... {datapath}')
data = pd.read_table(datapath, sep='\t')
data.rename(columns={'CellName': 'CELL', 'Drug': 'DRUG'}, inplace=True)
logger.info(f'data.shape {data.shape}')
logger.info('data memory usage (GB): {:.3f}'.format(sys.getsizeof(data)/1e9))
# print(data.groupby('SOURCE').agg({'CELL': 'nunique', 'DRUG': 'nunique', 'PUBCHEM': 'nunique'}).reset_index())
print(data.groupby('tissuetype').agg({'CELL': 'nunique', 'DRUG': 'nunique'}).reset_index())


data.columns[:20]
data[['CCLEName', 'CELL', 'Drug', 'tissuetype']].head()


# ========================================================================
# ========================================================================
def plot_rsp_dists(rsp, rsp_cols, savepath=None):
    """ Plot distributions of response variables.
    Args:
        rsp : df of response values
        rsp_cols : list of col names
        savepath : full path to save the image
    """
    ncols = 2
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
            ax.grid(True)

    plt.tight_layout()
    # plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
    if savepath is not None:
        plt.savefig(savepath, bbox_inches='tight', dpi=300)
    else:
        plt.savefig('rsp_dists.png', bbox_inches='tight', dpi=300)


plot_rsp_dists(rsp=data, rsp_cols=rsp_cols, savepath=os.path.join(OUTDIR, 'rsp_dists.png'))

