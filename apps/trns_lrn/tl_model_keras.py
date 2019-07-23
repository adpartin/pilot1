"""
Train base model. This model will be used to continue training.
"""
from __future__ import print_function, division

import warnings
warnings.filterwarnings('ignore')

import os
import sys
import platform
from pathlib import Path
import psutil
import argparse
from datetime import datetime
from time import time
from pprint import pprint, pformat
from collections import OrderedDict
from glob import glob

import sklearn
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.externals import joblib
from sklearn.metrics import r2_score, mean_absolute_error, median_absolute_error, explained_variance_score

import keras as ke
from keras import backend as K

from keras.models import Sequential, Model, model_from_json, model_from_yaml, load_model
from keras.optimizers import SGD, Adam, RMSprop, Adadelta
from keras.utils import np_utils, multi_gpu_model, plot_model
from keras.callbacks import Callback, ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping

import lightgbm as lgb

SEED = 42
t_start = time()

# File path
file_path = Path(__file__).resolve().parent

# Utils
utils_path = file_path / '../../utils'
sys.path.append(str(utils_path))
import utils
from classlogger import Logger
import ml_models
from ml_models import r2_krs

# Import custom callbacks
keras_contrib = '/vol/ml/apartin/projects/keras-contrib/keras_contrib/callbacks'
sys.path.append(keras_contrib)
from cyclical_learning_rate import CyclicLR

# Path
PRJ_NAME = Path(file_path.name)/'transfer_learn_keras'
OUTDIR = file_path / '../../out/' / PRJ_NAME


def parse_args(args):
    parser = argparse.ArgumentParser(description="Transfer learning.")

    # Input data
    parser.add_argument('-bs_dir', '--base_model_dir', default=None, type=str, help='Path to base (pre-trained) model dir.')
    parser.add_argument('-tl_dir', '--tl_data_dir', default=None, type=str, help='Path to data dir for transfer learning.')
    parser.add_argument('--tl_method', default='fe', type=str, choices=['ft', 'fe'],
                        help="Tranfer learning method. Choices: 'ft': finetune, 'fe': feature extractor, 'pred': just inference (not transfer learning)")

    # parser.add_argument('--dname', default='ytn', type=str, choices=['top6', 'ytn'], help='Dataset name (default: ytn).')
    # parser.add_argument('--frm', default='trch', type=str, choices=['krs', 'trch'], help='DL framework (default: trch).')
    # parser.add_argument('--src', default='GDSC', type=str, help='Data source (default: GDSC).')

    parser.add_argument('-ml', '--model_name', type=str, default='nn_reg0', help="ML model to use (default: 'nn_reg0').")
    parser.add_argument('-ep', '--epochs', default=50, type=int, help='Epochs (default: 50).')
    parser.add_argument('-b', '--batch_size', default=32, type=float, help='Batch size (default: 32).')
    parser.add_argument('--dr_rate', default=0.2, type=float, help='Dropout rate (default: 0.2).')

    parser.add_argument('--opt', default='clr_exp', type=str, choices=['sgd', 'adam', 'clr_trng1', 'clr_trng2', 'clr_exp'], help="Optimizer name (default: 'clr_exp').")
    parser.add_argument('--base_lr', type=float, default=1e-4, help='Base lr for cycle lr.')
    parser.add_argument('--max_lr', type=float, default=1e-3, help='Max lr for cycle lr.')
    parser.add_argument('--gamma', type=float, default=0.999994, help='Gamma parameter for learning cycle LR.')
    parser.add_argument('--skp_ep', type=int, default=3, help='Number of epochs to skip when plotting training curves.')

    parser.add_argument('--n_jobs', default=4, type=int, help='Number of cpu workers (default: 4).')
    args = parser.parse_args(args)
    return args


def create_outdir(outdir, args):
    t = datetime.now()
    t = [t.year, '-', t.month, '-', t.day, '_', 'h', t.hour, '-', 'm', t.minute]
    t = ''.join([str(i) for i in t])
    
    l = [args['src_bs'] + '_to_'+args['src_tl']] + [args['tl_method']] + [args['opt']] +  ['ep'+str(args['epochs'])] + ['drp'+str(args['dr_rate'])]
        
    name_sffx = '.'.join( l )
    outdir = Path(outdir) / (name_sffx + '_' + t)
    os.makedirs(outdir)
    return outdir


# Functions for transfer leaering with Keras -----------------
def print_trainable_layers(model, print_all=False):
    """ Print the trainable state of layers. """
    print('Trainable layers:')
    for layer in model.layers:
        if layer.trainable:
            print(layer.name, layer.trainable)
        if not layer.trainable and print_all:
            print(layer.name, layer.trainable)

            
def freeze_layers(model, layers_ids='all'):
    # freeze_layers = ['1', '2', '3', '4']
    if layers_ids=='all':
        for layer in model.layers:
            layer.trainable = False

    for layer in model.layers:
        if any([True for i in layers_ids if i in layer.name]):
            layer.trainable = False

            
def pop_layers(model, layers_ids):
    # pop_layers = ['4', '5', 'outputs']
    model_layers = model.layers
    for layer in model_layers[::-1]:
        if any([True for i in layers_ids if i in layer.name]):
            model.layers.pop()  
# ------------------------------------------------------------


def run(args):
    base_model_dir = Path(args['base_model_dir'])
    tl_data_dir = Path(args['tl_data_dir'])
    tl_method = args['tl_method']
    
    epochs = args['epochs']
    batch_size = args['batch_size']
    dr_rate = args['dr_rate']
    model_name = args['model_name']
    skp_ep = args['skp_ep']
    opt_name = args['opt']
    base_lr = args['base_lr']
    max_lr = args['max_lr']
    gamma = args['gamma']
    n_jobs = args['n_jobs']

    verbose = True
    # args['src'] = dirpath.name.split('.')[0]
    # src = args['src'] 
    
    # Extract data sources used for base model training and transfer learning
    src_bs = str(base_model_dir).split('/')[-1].split('.')[0]
    src_tl = str(tl_data_dir).split('/')[-1].split('.')[0]
    args['src_bs'] = src_bs
    args['src_tl'] = src_tl
    
    # Split of the dataset used for TRANSFER LEARNING
    fold_base = 0
    
    # Split of the dataset used for for taining the BASE MODEL
    fold_tl = 0

    
    # =====================================================
    #       Logger
    # =====================================================
    run_outdir = create_outdir(OUTDIR, args)
    logfilename = run_outdir/'logfile.log'
    lg = Logger(logfilename)
    lg.logger.info(datetime.now())
    lg.logger.info(f'\nFile path: {file_path}')
    lg.logger.info(f'Machine: {platform.node()} ({platform.system()}, {psutil.cpu_count()} CPUs)')
    lg.logger.info(f'\n{pformat(args)}')

    # Dump args to file
    utils.dump_args(args, run_outdir)      


    # =====================================================
    #       Load data for Transfer Learning
    # =====================================================
    def extract_subset_fea(df, fea_list, fea_sep='_'):
        """ Extract features based feature prefix name. """
        fea = [c for c in df.columns if (c.split(fea_sep)[0]) in fea_list]
        return df[fea]

    def extract_data(df, fea_list):
        """ ... """
        X = extract_subset_fea(df, fea_list=fea_list, fea_sep='_')
        Y = df[['auc']]
        meta = df.drop(columns=X.columns)
        meta = meta.drop(columns=['auc'])
        return X, Y, meta

    def load_data(dirpath, src, fold=0, ccl_fea_list=['geneGE'], drg_fea_list=['DD']):
        """ Load data (Yitan's data and splits) """
        data_fpath = Path((glob(str(dirpath/'*data.parquet')))[0])
        assert data_fpath.is_file(), '*data.parquet file was not found.'
        data = pd.read_parquet( data_fpath )
        lg.logger.info('\ndata {}'.format(data.shape))

        # Path to data splits
        datadir = Path(file_path/'../../data/yitan/Data')
        ccl_folds_dir = Path(file_path/'../../data/yitan/CCL_10Fold_Partition')
        pdm_folds_dir = Path(file_path/'../../data/yitan/PDM_10Fold_Partition')

        ids_path = ccl_folds_dir/f'{src}/cv_{fold_tl}' # 'TestList.txt'
        tr_ids_list = pd.read_csv(ids_path/'TrainList.txt', header=None).squeeze().values
        vl_ids_list = pd.read_csv(ids_path/'ValList.txt', header=None).squeeze().values
        te_ids_list = pd.read_csv(ids_path/'TestList.txt', header=None).squeeze().values

        data_tr = data[ data['cclname'].isin( tr_ids_list ) ]
        data_vl = data[ data['cclname'].isin( vl_ids_list ) ]
        data_te = data[ data['cclname'].isin( te_ids_list ) ]

        return data_tr, data_vl, data_te
    
    # Load data for transfer learning
    data_tr, data_vl, data_te = load_data(dirpath=tl_data_dir, src=src_tl)
    lg.logger.info('data_tr {}'.format(data_tr.shape))
    lg.logger.info('data_vl {}'.format(data_vl.shape))
    lg.logger.info('data_te {}'.format(data_te.shape))
    
    ccl_fea_list = ['geneGE']
    drg_fea_list = ['DD']
    xtr, ytr, mtr = extract_data(data_tr, fea_list = ccl_fea_list+drg_fea_list)
    xvl, yvl, mvl = extract_data(data_vl, fea_list = ccl_fea_list+drg_fea_list)
    xte, yte, mte = extract_data(data_te, fea_list = ccl_fea_list+drg_fea_list)
    
    
    # =====================================================
    #       Scale (CV can start here)
    # =====================================================
    scalerpath = base_model_dir/f'cv{fold_base}'/'scaler.pkl'
    scaler = joblib.load(scalerpath)
    
    cols = xtr.columns
    xtr = pd.DataFrame( scaler.transform(xtr), columns=cols, dtype=np.float32 )
    xvl = pd.DataFrame( scaler.transform(xvl), columns=cols, dtype=np.float32 )
    xte = pd.DataFrame( scaler.transform(xte), columns=cols, dtype=np.float32 )
    # joblib.dump(scaler, outdir/'scaler.pkl')
        
    
    # =====================================================
    #       Load base (pre-trained) model and predict w/o transfer learning
    # =====================================================    
    modelpath = base_model_dir/f'cv{fold_base}'/'final_model.h5'
    model = load_model(str(modelpath), custom_objects={'r2_krs': r2_krs})
    plot_model(model, to_file=run_outdir/f'{model_name}.png')
    
    # Predict
    pred_ytr = model.predict(xtr)
    pred_yvl = model.predict(xvl)
    pred_yte = model.predict(xte)    
    
    # Scores using the base model (as is, w/o transfer learning)
    lg.logger.info('\nScores using base model (simply predictions):')
    pred_scores = {}
    pred_scores['r2_tr'] = r2_score(ytr, pred_ytr)
    pred_scores['r2_vl'] = r2_score(yvl, pred_yvl)
    pred_scores['r2_te'] = r2_score(yte, pred_yte)
    pred_scores['mae_tr'] = mean_absolute_error(ytr, pred_ytr)
    pred_scores['mae_vl'] = mean_absolute_error(yvl, pred_yvl)
    pred_scores['mae_te'] = mean_absolute_error(yte, pred_yte)
    for k, v, in pred_scores.items(): lg.logger.info(f'{k}: {v}')
    utils.dump_dict(pred_scores, outpath=run_outdir/'./pred_scores.txt')
        
        
    # =====================================================
    #       Transfer learning
    # =====================================================            
    # Reset model
    lr_new = 5e-4
    model.compile(loss='mean_squared_error',
                  optimizer=SGD(lr=lr_new, momentum=0.9),
                  metrics=['mae']) 
    
    # CycleLR
    if opt_name == 'clr_trng1':
        clr = CyclicLR(base_lr=base_lr, max_lr=max_lr, mode='triangular')
    elif opt_name == 'clr_trng2':
        clr = CyclicLR(base_lr=base_lr, max_lr=max_lr, mode='triangular2')
    elif opt_name == 'clr_exp':
        clr = CyclicLR(base_lr=base_lr, max_lr=max_lr, mode='exp_range', gamma=gamma) # 0.99994; 0.99999994; 0.999994    
        
        
    # Callbacks
    checkpointer = ModelCheckpoint(str(run_outdir/'model_best.h5'), verbose=0, save_weights_only=False, save_best_only=True)
    csv_logger = CSVLogger(run_outdir/'training.log')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.75, patience=20, verbose=1, mode='auto',
                                  min_delta=0.0001, cooldown=3, min_lr=0.000000001)
    early_stop = EarlyStopping(monitor='val_loss', patience=50, verbose=1, mode='auto')    

    
    # Callbacks list
    callback_list = [checkpointer, csv_logger, early_stop, reduce_lr]
    if 'clr' in opt_name: callback_list = callback_list + [clr]      
    
    
    # Choose transfer learning method    
    if tl_method=='fe':
        # Feature extractor
        print_trainable_layers(model, print_all=True)
        pop_layers(model, layers_ids=['6','7','outputs'])
        print_trainable_layers(model, print_all=True)
        
        # Create feature extractor model
        fea_extractor_model = Model(inputs=model.input, outputs=model.layers[-1].output)

        # Extract featues
        xtr_fea_ext = fea_extractor_model.predict(xtr)
        xvl_fea_ext = fea_extractor_model.predict(xvl)
        xte_fea_ext = fea_extractor_model.predict(xte)
        # print(xvl_fea_ext.shape)
        # print(xte_fea_ext.shape)
        
        # Train final model on extracted features
        lg.logger.info('\n{}'.format('=' * 50))
        lg.logger.info('Train LGBM ...')

        # Model
        init_kwargs = {'objective': 'regression', 'n_estimators': 100, 'n_jobs': n_jobs, 'random_state': SEED}    
        model = lgb.LGBMModel(**init_kwargs)

        # Train
        fit_kwargs = {'verbose': verbose}
        t0 = time()
        model.fit(xtr_fea_ext, ytr, **fit_kwargs)
        fe_runtime = time() - t0
        lg.logger.info('Feature extractor train time: {:.1f} mins'.format(fe_runtime/60))

        # Predict
        pred_ytr = model.predict(xtr_fea_ext)
        pred_yvl = model.predict(xvl_fea_ext)
        pred_yte = model.predict(xte_fea_ext)

        # Dump predictions
        pass        
        
        # Calc scores
        lg.logger.info('\nScores Feature extractor LGBM:')
        fe_scores = {}
        fe_scores['r2_tr'] = r2_score(ytr, pred_ytr)
        fe_scores['r2_vl'] = r2_score(yvl, pred_yvl)
        fe_scores['r2_te'] = r2_score(yte, pred_yte)
        fe_scores['mae_tr'] = mean_absolute_error(ytr, pred_ytr)
        fe_scores['mae_vl'] = mean_absolute_error(yvl, pred_yvl)
        fe_scores['mae_te'] = mean_absolute_error(yte, pred_yte)
        for k, v, in fe_scores.items(): lg.logger.info(f'{k}: {v}')
        utils.dump_dict(fe_scores, outpath=run_outdir/'fe_scores.txt')
        
        
    elif tl_method=='ft':
        # Finetune
        print_trainable_layers(model, print_all=True)
        freeze_layers(model, layers_ids=['1', '2', '3', '4'])
        print_trainable_layers(model, print_all=True)
        
        # Fit params
        fit_kwargs = {'epochs': epochs, 'batch_size': batch_size, 'verbose': 1}
        # fit_kwargs['validation_data'] = (xvl, yvl)
        fit_kwargs['validation_split'] = 0.2
        fit_kwargs['callbacks'] = callback_list  
        
        t0 = time()
        history = model.fit(xvl, yvl, **fit_kwargs)
        ft_runtime = time() - t0
        lg.logger.info('Finetune runtime: {:.1f} mins'.format(ft_runtime/60))
        
        # Predict
        pred_ytr = model.predict(xtr)
        pred_yvl = model.predict(xvl)
        pred_yte = model.predict(xte)
        pred_yvl = model.predict(xvl)
        
        # Dump predictions
        pass
        
        # Scores using finetune method (transfer learning)
        lg.logger.info('\nScores using finetune:')
        ft_scores = {}
        ft_scores['r2_tr'] = r2_score(ytr, pred_ytr)
        ft_scores['r2_vl'] = r2_score(yvl, pred_yvl)
        ft_scores['r2_te'] = r2_score(yte, pred_yte)
        ft_scores['mae_tr'] = mean_absolute_error(ytr, pred_ytr)
        ft_scores['mae_vl'] = mean_absolute_error(yvl, pred_yvl)
        ft_scores['mae_te'] = mean_absolute_error(yte, pred_yte)
        for k, v, in ft_scores.items(): lg.logger.info(f'{k}: {v}')
        utils.dump_dict(tl_scores, outpath=run_outdir/'ft_scores.txt')
        
        
    # Make final plot  # TODO
    pass
    


    # -----------------
    # Summarize results
    # -----------------
    # Plots
    # plts_path = outdir/'plts'
    # os.makedirs(plts_path, exist_ok=True)
    ml_models.plot_prfrm_metrics(history, title=f'Train base model {model_name}',
                                 skp_ep=skp_ep, add_lr=True, outdir=out_nn_model)

    # Save keras history
    ml_models.save_krs_history(history, out_nn_model)

    lg.kill_logger()


def main(args):
    args = parse_args(args)
    args = vars(args)
    ret = run(args)
    

if __name__ == '__main__':
    """ __name__ == '__main__' explained: www.youtube.com/watch?v=sugvnHA7ElY """
    main(sys.argv[1:])


