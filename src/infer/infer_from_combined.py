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
file_path = os.path.join(file_path, 'src/infer')
os.chdir(file_path)

# file_path = os.path.dirname(os.path.realpath(__file__))  # os.path.dirname(os.path.abspath(__file__))
##utils_path = os.path.abspath(os.path.join(file_path, 'utils'))
##sys.path.append(utils_path)
import utils_models as utils

DATADIR = os.path.join(file_path, '../../data/processed/from_combined')
MODELDIR = os.path.join(file_path, '../../models/from_combined')
OUTDIR = os.path.join(file_path, '../../infer/from_combined')
os.makedirs(OUTDIR, exist_ok=True)

DATA_FILE_NAME = 'tidy_data_no_fibro.parquet'
MODEL_FILE_NAME = 'run_2018-12-5_h17-m42'

SEED = 0


# ========================================================================
#       Args TODO: add to argparse
# ========================================================================
# Train and infer data
train_sources = ['ccle']  # ['ccle', 'gcsi', 'gdsc', 'ctrp']
infer_sources = ['ctrp']

# Traget (response)
# target_name = 'AUC'
target_name = 'AUC1'

# Features
cell_features = ['rnaseq'] # ['rnaseq', cnv', 'rnaseq_latent']
drug_features = ['descriptors'] # [] # ['descriptors', 'fingerprints', 'descriptors_latent', 'fingerprints_latent']
other_features = [] # ['drug_labels'] # ['cell_labels', 'drug_labels', 'ctype', 'csite', 'rna_clusters']

# Models
# ml_models = ['tpot_reg']
ml_models = ['lgb_reg']

trasform_target = False
outlier_remove = False  # IsolationForest
verbose = True
n_jobs = 4

# Feature prefix
fea_prefix = {'rnaseq': 'cell_rna',
              'cnv': 'cell_cnv',
              'descriptors': 'drug_dsc',
              'fingerprints': 'drug_fng'}


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
data = pd.read_parquet(datapath, engine='auto', columns=None)
logger.info(f'data.shape {data.shape}')
logger.info('data memory usage (GB): {:.3f}'.format(sys.getsizeof(data)/1e9))
# print(data.groupby('SOURCE').agg({'CELL': 'nunique', 'DRUG': 'nunique', 'PUBCHEM': 'nunique'}).reset_index())


# Extract sources
logger.info('\nExtract train sources ... {}'.format(train_sources))
data = data[data['SOURCE'].isin(train_sources)].reset_index(drop=True)
logger.info(f'data.shape {data.shape}')
logger.info('data memory usage (GB): {:.3f}'.format(sys.getsizeof(data)/1e9))
# print(data.groupby('SOURCE').agg({'CELL': 'nunique', 'DRUG': 'nunique'}).reset_index())


