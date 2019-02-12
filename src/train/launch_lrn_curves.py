""" 
This script generates multiple learning curves for different training sets.
It launches a script (e.g. trn_lrn_curves.py) that train ML model(s) on various training set sizes.
"""
# python -m pdb src/models/launch_lrn_curves.py
import os
import sys
import time
import datetime
import argparse
from collections import OrderedDict
import numpy as np
import pandas as pd

import trn_lrn_curves


file_path = os.path.dirname(os.path.realpath(__file__))
OUTDIR = os.path.join(file_path, '../../models/from_combined')


def main(args):

    t0 = time.time()

    # Create folder to store results
    t = datetime.datetime.now()
    t = [t.year, '-', t.month, '-', t.day, '_', 'h', t.hour, '-', 'm', t.minute]
    t = ''.join([str(i) for i in t])
    dirname = 'lrn_curves~' + t
    csv_outdir = os.path.join(OUTDIR, dirname)
    os.makedirs(csv_outdir, exist_ok=True)

    # Full set
    cross_study_sets = [
        {'tr_src': ['gcsi']},
        {'tr_src': ['ccle']},
        {'tr_src': ['gdsc']},
        {'tr_src': ['ctrp']},
        {'tr_src': ['nci60']}
    ]


    # Single run
    # idx = 2
    # df_csv_scores, prms = trn_from_combined.main(
    #     ['-tr', *cross_study_sets[idx]['tr_src'],
    #     '-te', *cross_study_sets[idx]['te_src'],
    #     *args])


    # Multiple runs
    dfs = []
    for run_id in range(len(cross_study_sets)):
        print('{} Run {} {}'.format('-'*40, run_id+1, '-'*40))
        lrn_curve_scores, prms = trn_lrn_curves.main(
            ['-tr', *cross_study_sets[run_id]['tr_src'],
             '--outdir', csv_outdir,
             *args])
        src_name = '_'.join( cross_study_sets[run_id]['tr_src'] )
        lrn_curve_scores.insert(loc=0, column='src', value=src_name)
        dfs.append(lrn_curve_scores)

    df = pd.concat(dfs, axis=0, sort=False)
    df.to_csv('lrn_curves_all.csv', index=False)

    # # Create csv table for each available metric 
    # df = pd.concat(dfs, axis=0, sort=False)
    # for m in df['metric'].unique():
    #     csv = df[df['metric']==m].reset_index(drop=True)
    #     csv.drop(columns=['metric'], inplace=True)

    #     # Sort rows and cols
    #     tr_src = csv['train_src']
    #     csv.drop(columns='train_src', inplace=True)
    #     csv = csv[sorted(csv.columns)]
    #     csv = pd.concat([tr_src, csv], axis=1, sort=False)
    #     csv = csv.sort_values('train_src')

    #     # save table
    #     csv = csv.round(2)
    #     csv.to_csv(os.path.join(csv_outdir, f'cross-study-val-{m}.csv'), index=False)

    print('\nTotal runtime {:.3f}\n'.format((time.time()-t0)/60))


main(sys.argv[1:])

