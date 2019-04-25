""" 
This script generates the cross-study validation (csv) table.
It launches a script (e.g. trn_from_combined.py) that train ML model(s) using a single source
and makes predictions (inference) on a specified set of other sources.
"""
# python -m pdb src/models/launch_cross_study_val.py
from __future__ import print_function, division

import warnings
warnings.filterwarnings('ignore')

# from comet_ml import Experiment
import os

import sys
from pathlib import Path
import argparse
import datetime
from time import time
from collections import OrderedDict

import sklearn
import numpy as np
import pandas as pd


# File path
# file_path = os.path.dirname(os.path.realpath(__file__))
file_path = Path(__file__).resolve().parent


# Utils
utils_path = file_path / '../../utils'
sys.path.append(str(utils_path))
import trn_from_combined
from classlogger import Logger


# Path
PRJ_NAME = file_path.name 
# OUTDIR = file_path / '../../models/from_combined'
OUTDIR = file_path / '../../out/' / PRJ_NAME


def create_csv(df, outdir='.'):
    """ Creates CSV table for each available metric. """
    # lg.logger.info('Create csv table ...')
    for m in df['metric'].unique():
        csv = df[df['metric']==m].reset_index(drop=True)
        csv.drop(columns=['metric'], inplace=True)

        # Sort rows and cols
        tr_src = csv['train_src']
        csv.drop(columns='train_src', inplace=True)
        csv = csv[sorted(csv.columns)]
        csv = pd.concat([tr_src, csv], axis=1, sort=False)
        csv = csv.sort_values('train_src')

        # save table
        csv = csv.round(3)
        csv.to_csv(outdir/f'csv_{m}.csv', index=False)


def main(args):
    t0 = time()    

    # Create folder to store results
    t = datetime.datetime.now()
    t = [t.year, '-', t.month, '-', t.day, '_', 'h', t.hour, '-', 'm', t.minute]
    t = ''.join([str(i) for i in t])
    dirname = 'csv_' + t
    outdir = OUTDIR/dirname
    os.makedirs(outdir, exist_ok=True)

    # Create logger
    logfilename = outdir/'csv_logfile.log'
    # lg = Logger(logfilename=logfilename)

    # Full set
    cross_study_sets = [
        {'tr_src': ['gcsi'],
         'te_src': ['ctrp', 'gdsc', 'ccle', 'gcsi', 'nci60']},

        {'tr_src': ['ccle'],
         'te_src': ['ctrp', 'gdsc', 'ccle', 'gcsi', 'nci60']},

        {'tr_src': ['gdsc'],
         'te_src': ['ctrp', 'gdsc', 'ccle', 'gcsi', 'nci60']},

        {'tr_src': ['ctrp'],
         'te_src': ['ctrp', 'gdsc', 'ccle', 'gcsi', 'nci60']},

         {'tr_src': ['nci60'],
         'te_src': ['ctrp', 'gdsc', 'ccle', 'gcsi', 'nci60']}
    ]

    # Smaller set
    # cross_study_sets = [
    #     {'tr_src': ['gcsi'],
    #      'te_src': ['ccle', 'gcsi', 'gdsc']},

    #     {'tr_src': ['ccle'],
    #      'te_src': ['ccle', 'gcsi', 'gdsc']},
    # ]


    # Single run
    # idx = 2
    # df_csv_scores, prms = trn_from_combined.main(
    #     ['-tr', *cross_study_sets[idx]['tr_src'],
    #     '-te', *cross_study_sets[idx]['te_src'],
    #     *args])


    # Train using specific data source and predict on others
    # lg.logger.info('Start interrate over training sources ...')
    dfs = []
    for run_id in range(len(cross_study_sets)):
        print('{} Run {} {}'.format('-'*40, run_id+1, '-'*40))
        csv_scores, prms = trn_from_combined.main(
            ['-tr', *cross_study_sets[run_id]['tr_src'],
             '-te', *cross_study_sets[run_id]['te_src'],
             '--outdir', str(outdir),
             *args])
        dfs.append(csv_scores)


    # Combine csv scores from all runs
    csv_all = pd.concat(dfs, axis=0, sort=False)
    csv_all.to_csv(outdir/'csv_all.csv', index=False)


    # Create csv table for each available metric
    create_csv(df=csv_all, outdir=outdir)


    # lg.logger.info('\nTotal CSV runtime {:.3f}\n'.format((time.time()-t0)/60))
    print('Total CSV runtime {:.1f} mins\n'.format( (time()-t0)/60) )


main(sys.argv[1:])

