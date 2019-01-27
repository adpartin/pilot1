""" 
This script generates the cross-study validation (csv) table.
It launches a script (e.g. train_from_combined.py) that train ML model(s) using a single source
and runs predictions (inference) on a specified set of other sources.
"""
# python -m pdb src/models/launch_cross_study_val.py
import os
import sys
import time
import datetime
import argparse
from collections import OrderedDict
import numpy as np
import pandas as pd

import train_from_combined

file_path = os.path.dirname(os.path.realpath(__file__))


def main(args):

    t0 = time.time()
    t = datetime.datetime.now()
    t = [t.year, '-', t.month, '-', t.day, '_', 'h', t.hour, '-', 'm', t.minute]
    t = ''.join([str(i) for i in t])

    # Full set
    # cross_study_sets = [
    #     {'tr_src': ['gcsi'],
    #     'te_src': ['ctrp', 'gdsc', 'ccle', 'gcsi']},

    #     {'tr_src': ['ccle'],
    #     'te_src': ['ctrp', 'gdsc', 'ccle', 'gcsi']},

    #     {'tr_src': ['gdsc'],
    #     'te_src': ['ctrp', 'gdsc', 'ccle', 'gcsi']},

    #     {'tr_src': ['ctrp'],
    #     'te_src': ['ctrp', 'gdsc', 'ccle', 'gcsi']},
    # ]

    # Smaller set
    cross_study_sets = [
        {'tr_src': ['gcsi'],
         'te_src': ['ccle', 'gcsi', 'gdsc']},

        {'tr_src': ['ccle'],
         'te_src': ['ccle', 'gcsi', 'gdsc']},
    ]


    # Single run
    # idx = 2
    # df_csv_scores, params = train_from_combined.main(
    #     ['-tr', *cross_study_sets[idx]['tr_src'],
    #     '-te', *cross_study_sets[idx]['te_src'],
    #     *args])


    # Multiple runs
    dfs = []
    for run_id in range(len(cross_study_sets)):
        print('{} Run {} {}'.format('-'*40, run_id+1, '-'*40))
        csv_scores_all, params = train_from_combined.main(
            ['-tr', *cross_study_sets[run_id]['tr_src'],
             '-te', *cross_study_sets[run_id]['te_src'],
             *args])
        dfs.append(csv_scores_all)

    # Create folder to store results
    t = datetime.datetime.now()
    t = [t.year, '-', t.month, '-', t.day, '_', 'h', t.hour, '-', 'm', t.minute]
    t = ''.join([str(i) for i in t])
    dirname = 'cross_study_val~' + params['cv_method'] + '.' + params['target_name'] + '~' + t
    csv_outdir = os.path.join(params['outdir'], dirname)
    os.makedirs(csv_outdir, exist_ok=True)

    # Create csv table for each available metric 
    df = pd.concat(dfs, axis=0, sort=False)
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
        csv = csv.round(2)
        csv.to_csv(os.path.join(csv_outdir, f'cross-study-val-{m}.csv'), index=False)

    print('\nTotal runtime {:.3f}\n'.format((time.time()-t0)/60))


main(sys.argv[1:])

