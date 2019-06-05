""" This is the main script. """
from __future__ import print_function, division

import warnings
warnings.filterwarnings('ignore')

# from comet_ml import Experiment
import os

import sys
from pathlib import Path
import psutil
import argparse
from datetime import datetime
from time import time
from pprint import pprint
from collections import OrderedDict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sklearn
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split

SEED = None


# File path
# file_path = os.path.dirname(os.path.realpath(__file__))
file_path = Path(__file__).resolve().parent


# Utils
utils_path = file_path / '../../utils'
sys.path.append(str(utils_path))
import utils
from utils_tidy import load_tidy_combined, get_data_by_src, break_src_data 
import argparser
from classlogger import Logger
import ml_models
from cv_splitter import cv_splitter, plot_ytr_yvl_dist
from lrn_crv import my_learning_curve


# Path
PRJ_NAME = file_path.name 
OUTDIR = file_path / '../../out/' / PRJ_NAME
CONFIGFILENAME = 'config_prms.txt'


def create_outdir(outdir, args, src):
    t = datetime.now()
    t = [t.year, '-', t.month, '-', t.day, '_', 'h', t.hour, '-', 'm', t.minute]
    t = ''.join([str(i) for i in t])
    
    l = [args['opt']] + [args['cv_method']] + [('cvf'+str(args['cv_folds']))] + args['cell_features'] + args['drug_features'] + [args['target_name']]

    if ('nn' in args['model_name']) and (args['attn'] is True): 
        name_sffx = '.'.join( [src] + [args['model_name']] + ['attn'] + l )
                             
    elif ('nn' in args['model_name']) and (args['attn'] is False): 
        name_sffx = '.'.join( [src] + [args['model_name']] + ['fc'] + l )
        
    else:
        name_sffx = '.'.join( [src] + [args['model_name']] + l )

    outdir = Path(outdir) / (name_sffx + '_' + t)
    os.makedirs(outdir)
    return outdir







def main(args):
    config_fname = file_path / CONFIGFILENAME
    args = argparser.get_args(args=args, config_fname=config_fname)
    ## pprint(vars(args))
    args = vars(args)
    if args['outdir'] is None:
        args['outdir'] = OUTDIR
    
    #args = None
    ret = run(args)


if __name__ == '__main__':
    main(sys.argv[1:])
 
