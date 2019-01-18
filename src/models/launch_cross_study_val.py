""" 
This script generates the cross-study validation (csv) table.
It launches a script (e.g. train_from_combined.py) that train ML model(s) using a single source
and runs predictions (inference) on a specified set of other sources.
"""
# python -m pdb src/models/launch_cross_study_val.py
import os
import sys
import argparse
from collections import OrderedDict
import numpy as np
import pandas as pd

import train_from_combined


def main(args):

    # Full set
    cross_study_sets = [
        {'tr_src': ['ctrp'],
        'te_src': ['ctrp', 'gdsc', 'ccle', 'gcsi']},

        {'tr_src': ['gdsc'],
        'te_src': ['ctrp', 'gdsc', 'ccle', 'gcsi']},

        {'tr_src': ['ccle'],
        'te_src': ['ctrp', 'gdsc', 'ccle', 'gcsi']},

        {'tr_src': ['gcsi'],
        'te_src': ['ctrp', 'gdsc', 'ccle', 'gcsi']},
    ]

    # Smaller set
    # cross_study_sets = [
    #     {'tr_src': ['ccle'],
    #      'te_src': ['ccle', 'gcsi']},

    #     {'tr_src': ['gcsi'],
    #      'te_src': ['ccle', 'gcsi']},
    # ]


    # Single run
    # idx = 2
    # train_from_combined.main(['-tr', *cross_study_sets[idx]['tr_src'],
    #                           '-te', *cross_study_sets[idx]['te_src']])

    # Multiple runs
    dfs = []
    for i in range(len(cross_study_sets)):
        print('{} Run {} {}'.format('-'*30, i+1, '-'*30))
        df_csv_scores, outdir = train_from_combined.main(
            ['-tr', *cross_study_sets[i]['tr_src'],
             '-te', *cross_study_sets[i]['te_src'],
             *args])   
        dfs.append(df_csv_scores)


    # Create csv table for each available metric 
    df = pd.concat(dfs, axis=0)
    for m in df['metric'].unique():
        csv = df[df['metric']==m].reset_index(drop=True)
        csv.drop(columns=['metric'], inplace=True)

        # Sort rows and cols
        csv = csv.sort_values('train_src')
        csv = csv[[csv.columns[0]] + sorted(csv.columns[1:])]
        
        # save table
        csv.to_csv(os.path.join(outdir, f'cross-study-val-{m}.csv'), index=False)


main(sys.argv[1:])
