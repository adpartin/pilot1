""" 
This script generates multiple learning curves for different training sets.
It launches a script (e.g. trn_lrn_crv.py) that train ML model(s) on various training set sizes.

TODO: update the function trn_lrn_crv.py so that it will accept a df instead of the source names!
"""
# python -m pdb apps/lrn_crv/launch_lrn_crv.py
from __future__ import print_function, division

import warnings
warnings.filterwarnings('ignore')

# from comet_ml import Experiment
import os

import sys
from pathlib import Path
import argparse
from datetime import datetime
from time import time
from collections import OrderedDict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sklearn
import numpy as np
import pandas as pd


# File path
# file_path = os.path.dirname(os.path.realpath(__file__))
file_path = Path(__file__).resolve().parent


# Utils
utils_path = file_path / '../../utils'
sys.path.append(str(utils_path))
import trn_lrn_crv
import argparser


# Path
PRJ_NAME = file_path.name 
OUTDIR = file_path / '../../out/' / PRJ_NAME



def plot_agg_lrn_crv(df, outdir='.'):
    """ Generates learning curve plots for each metric across all cell line sources. """
    # Get the number of cv_folds
    cvf = len([c for c in df.columns.tolist() if c[0]=='f'])

    rs = ['b', 'r', 'k', 'c', 'm']
    title = None

    for i, met_name in enumerate(df['metric'].unique()):
        dfm = df[df['metric']==met_name].reset_index(drop=True)

        y_values = dfm.iloc[:, -cvf:].values
        y_ = y_values.min() * 0.05
        ymin = y_values.min()
        ymax = y_values.max()
        ylim = [ymin - y_, ymax + y_]

        fig = plt.figure(figsize=(14, 7))
        for j, s in enumerate(dfm['src'].unique()):

            dfs = dfm[dfm['src']==s].reset_index(drop=True)
            tr_sizes  = dfs['tr_size'].unique()
            tr_scores = dfs.loc[dfs['tr_set']==True, dfs.columns[-cvf:]]
            te_scores = dfs.loc[dfs['tr_set']==False, dfs.columns[-cvf:]]

            tr_scores_mean = np.mean(tr_scores, axis=1)
            tr_scores_std  = np.std(tr_scores, axis=1)
            te_scores_mean = np.mean(te_scores, axis=1)
            te_scores_std  = np.std(te_scores, axis=1)

            plt.plot(tr_sizes, tr_scores_mean, '.-', color=colors[j], label=s+'_tr')
            plt.plot(tr_sizes, te_scores_mean, '.--', color=colors[j], label=s+'_val')

            plt.fill_between(tr_sizes, tr_scores_mean - tr_scores_std, tr_scores_mean + tr_scores_std, alpha=0.1, color=colors[j])
            plt.fill_between(tr_sizes, te_scores_mean - te_scores_std, te_scores_mean + te_scores_std, alpha=0.1, color=colors[j])

            if title is not None:
                plt.title(title)
            else:
                plt.title('Learning curve (' + met_name + ')')
            plt.xlabel('Train set size')
            plt.ylabel(met_name)
            plt.legend(bbox_to_anchor=(1.1, 1), loc='upper right', ncol=1)
            # plt.legend(loc='best')
            plt.grid(True)
            plt.tight_layout()
        
        plt.ylim(ylim) 
        plt.savefig( Path(outdir) / ('lrn_crv_' + met_name + '.png') )


def main(args):
    t0 = time()

    # Create folder to store results
    t = datetime.now()
    t = [t.year, '-', t.month, '-', t.day, '_', 'h', t.hour, '-', 'm', t.minute]
    t = ''.join([str(i) for i in t])
    dirname = 'lrn_crv_' + t
    #outdir = OUTDIR/dirname
    #os.makedirs(outdir, exist_ok=True)
    
    # new
    CONFIGFILENAME = 'config_prms.txt'
    config_fname = file_path / CONFIGFILENAME
    tmp_args = argparser.get_args(args=args, config_fname=config_fname)    
    tmp_args = vars(tmp_args)
    
    if ('nn' in tmp_args['model_name']) and (tmp_args['attn'] is True): 
        name_sffx = '.'.join( [tmp_args['model_name']] + ['attn'] + \
                             [tmp_args['cv_method']] + [('cvf'+str(tmp_args['cv_folds']))] + tmp_args['cell_features'] + \
                             tmp_args['drug_features'] + [tmp_args['target_name']] )
            
    elif ('nn' in tmp_args['model_name']) and (tmp_args['attn'] is False): 
        name_sffx = '.'.join( [tmp_args['model_name']] + ['fc'] + \
                             [tmp_args['cv_method']] + [('cvf'+str(tmp_args['cv_folds']))] + tmp_args['cell_features'] + \
                             tmp_args['drug_features'] + [tmp_args['target_name']] )
        
    else:
        name_sffx = '.'.join( [tmp_args['model_name']] + \
                             [tmp_args['cv_method']] + [('cvf'+str(tmp_args['cv_folds']))] + tmp_args['cell_features'] + \
                             tmp_args['drug_features'] + [tmp_args['target_name']] )
        
    run_outdir = Path(outdir) / (name_sffx + '_' + t)
    os.makedirs(run_outdir)
    
    
    # Full set
    csv_sets = [
        {'tr_src': ['gcsi']},
        {'tr_src': ['ccle']},
        {'tr_src': ['gdsc']},
        {'tr_src': ['ctrp']},
        {'tr_src': ['nci60']}
    ]


    # Single run
    # idx = 2
    # df_csv_scores, prms = trn_from_combined.main(
    #     ['-tr', *csv_sets[idx]['tr_src'],
    #     '-te', *csv_sets[idx]['te_src'],
    #     *args])


    # Multiple runs
    dfs = []
    for run_id in range(len(csv_sets)):
        print('{} Run {} {}'.format('-'*40, run_id+1, '-'*40))
        lrn_crv_scores, prms = trn_lrn_crv.main(
            ['-tr', *csv_sets[run_id]['tr_src'],
             '--outdir', str(outdir),
             *args])
        src_name = '_'.join( csv_sets[run_id]['tr_src'] )
        lrn_crv_scores.insert(loc=0, column='src', value=src_name)
        dfs.append(lrn_crv_scores)

    df = pd.concat(dfs, axis=0, sort=False)
    df.to_csv(outdir/'lrn_crv_all.csv', index=False)

    plot_agg_lrn_crv(df, outdir)

    print('Total runtime {:.1f}\n'.format( (time()-t0)/60) )


main(sys.argv[1:])

