"""
Train base model. This model will be used to continue training.
"""
from __future__ import print_function, division

import warnings
warnings.filterwarnings('ignore')

import os
import sys
from pathlib import Path
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
import utils

utils_path = file_path / '../../utils'
sys.path.append(str(utils_path))
#import utils
from classlogger import Logger
import ml_models

# Import custom callbacks
keras_contrib = '/vol/ml/apartin/projects/keras-contrib/keras_contrib/callbacks'
sys.path.append(keras_contrib)
from cyclical_learning_rate import CyclicLR

# Path
PRJ_NAME = Path(file_path.name)/'base_model_keras'
OUTDIR = file_path / '../../out/' / PRJ_NAME


def parse_args(args):
    parser = argparse.ArgumentParser(description="Train base NN model.")

    # Input data
    parser.add_argument('--src', default='GDSC', type=str, choices=['CCLE', 'CTRP', 'gCSI', 'GDSC', 'NCI60', 'all'], 
                        help='Data source (default: GDSC).')

    parser.add_argument('--fold', default=0, type=int, help='Split of the dataset used for taining the BASE MODEL (default: 0).')

    parser.add_argument('-cf', '--ccl_fea', nargs='+', default=['geneGE'], choices=['geneGE'], help='Cell line features (default: `geneGE`).')
    parser.add_argument('-df', '--drg_fea', nargs='+', default=['DD'], choices=['DD', 'lbl'], help='Drug features (default: `DD`).')

    parser.add_argument('--drg_subset', default='all', choices=['pdm', 'common', 'all'], help='Drug subset to use for training (default: all).')

    parser.add_argument('-ml', '--model_name', type=str, choices=['nn_reg0', 'nn_reg3', 'nn_reg4', 'nn_reg5', 'nn_reg6'], default='nn_reg0')
    parser.add_argument('-ep', '--epochs', default=250, type=int, help='Epochs (default: 250).')
    parser.add_argument('-b', '--batch_size', default=32, type=float, help='Batch size (default: 32).')
    parser.add_argument('--dr_rate', default=0.2, type=float, help='Dropout rate (default: 0.2).')

    parser.add_argument('--opt', default='clr_trng1', type=str, choices=['sgd', 'adam', 'clr_trng1', 'clr_trng2', 'clr_exp'], help='Optimizer name (default: `sgd`).')
    parser.add_argument('--base_lr', type=float, default=1e-4, help='Base lr for cycle lr.')
    parser.add_argument('--max_lr', type=float, default=1e-3, help='Max lr for cycle lr.')
    parser.add_argument('--gamma', type=float, default=0.999994, help='Gamma parameter for learning cycle LR.')

    parser.add_argument('--skp_ep', type=int, default=10, help='Number of epochs to skip when plotting training curves.')
    parser.add_argument('--n_jobs', default=8, type=int, help='Number of cpu workers (default: 4).')
    args = parser.parse_args(args)
    return args


def create_outdir(outdir, args):
    t = datetime.now()
    t = [t.year, '-', t.month, '-', t.day, '_', 'h', t.hour, '-', 'm', t.minute]
    t = ''.join([str(i) for i in t])
    
    l = [args['src']] + ['drg_'+args['drg_subset']] + [args['model_name']] + [args['opt']] +  ['ep'+str(args['epochs'])] + ['drp'+str(args['dr_rate'])]
        
    name_sffx = '.'.join( l )
    outdir = Path(outdir) / (name_sffx + '_' + t)
    os.makedirs(outdir)
    return outdir


def run(args):
    src = args['src']
    fold = args['fold']
    ccl_fea_list = args['ccl_fea']
    drg_fea_list = args['drg_fea']
    drg_subset = args['drg_subset']
    fea_sep = '_'
    
    model_name = args['model_name']
    epochs = args['epochs']
    batch_size = args['batch_size']
    dr_rate = args['dr_rate']

    opt_name = args['opt']
    base_lr = args['base_lr']
    max_lr = args['max_lr']
    gamma = args['gamma']

    skp_ep = args['skp_ep']
    n_jobs = args['n_jobs']
    verbose = True
    #args['src'] = dirpath.name.split('.')[0]
    #src = args['src'] 
    

    # =====================================================
    #       Logger
    # =====================================================
    run_outdir = create_outdir(OUTDIR, args)
    lg = Logger(run_outdir/'logfile.log')
    lg.logger.info(f'File path: {file_path}')
    lg.logger.info(f'\n{pformat(args)}')

    # Dump args to file
    utils.dump_dict(args, run_outdir/'args.txt')      


    # =====================================================
    #       Load data
    # =====================================================
    data = utils.load_data(file_path, src=src,
            ccl_fea_list=ccl_fea_list, drg_fea_list=drg_fea_list,
            drg_subset=drg_subset,
            fea_sep=fea_sep, logger=lg.logger)

    ccl_folds_dir = Path(file_path/f'../../data/yitan/CCL_10Fold_Partition/{src}')
    #pdm_folds_dir = Path(file_path/'../../data/yitan/PDM_10Fold_Partition')
    ids_path = ccl_folds_dir/f'cv_{fold}' 

    data_tr, data_vl, data_te = utils.get_splits_per_fold(data, ids_path, logger=lg.logger)

    xtr, ytr, mtr = utils.extract_data(data_tr, fea_list = ccl_fea_list+drg_fea_list)
    xvl, yvl, mvl = utils.extract_data(data_vl, fea_list = ccl_fea_list+drg_fea_list)
    xte, yte, mte = utils.extract_data(data_te, fea_list = ccl_fea_list+drg_fea_list)

    # Create output dir
    fold_outdir = run_outdir / ('cv'+str(fold))
    os.makedirs(fold_outdir, exist_ok=False)
    
    # Dump y values to file (this is used to scale the values of PDM)
    ytr.to_csv(fold_outdir/'ytr.csv', index=False)
    yvl.to_csv(fold_outdir/'yvl.csv', index=False)
    yte.to_csv(fold_outdir/'yte.csv', index=False)


    # =====================================================
    #       Scale (CV can start here) --> TODO: handle categorical features!
    # =====================================================
    cols = xtr.columns
    # scaler = MinMaxScaler()
    scaler = StandardScaler()

    cols = xtr.columns
    xtr = pd.DataFrame(scaler.fit_transform(xtr), columns=cols, dtype=np.float32)
    xvl = pd.DataFrame(scaler.transform(xvl), columns=cols, dtype=np.float32)
    xte = pd.DataFrame(scaler.transform(xte), columns=cols, dtype=np.float32)

    # Dump scaler
    joblib.dump(scaler, fold_outdir/'scaler.pkl')
                

    # =====================================================
    #       Train NN keras
    # =====================================================
    # Params
    if model_name == 'nn_reg3':
        xtr_rna = utils.extract_subset_fea(xtr, fea_list=ccl_fea_list, fea_sep=fea_sep)
        xvl_rna = utils.extract_subset_fea(xvl, fea_list=ccl_fea_list, fea_sep=fea_sep)
        xtr_dsc = utils.extract_subset_fea(xtr, fea_list=drg_fea_list, fea_sep=fea_sep)
        xvl_dsc = utils.extract_subset_fea(xvl, fea_list=drg_fea_list, fea_sep=fea_sep)
        xtr_dct = {'in_rna': xtr_rna, 'in_dsc': xtr_dsc} 
        xvl_dct = {'in_rna': xvl_rna, 'in_dsc': xvl_dsc} 
        init_kwargs = {'in_dim_rna': xtr_rna.shape[1], 'in_dim_dsc': xtr_dsc.shape[1], 'dr_rate': dr_rate, 'opt_name': opt_name, 'logger': lg.logger}
    elif model_name == 'nn_reg6':
        # TODO: uses RNA-seq and categorical drug embeddings
        pass
    else:
        xtr_dct = {'inputs': xtr} 
        xvl_dct = {'inputs': xvl} 
        init_kwargs = {'input_dim': xtr.shape[1], 'dr_rate': dr_rate, 'opt_name': opt_name, 'logger': lg.logger}
    fit_kwargs = {'batch_size': batch_size, 'epochs': epochs, 'verbose': 1}

    # Get NN model
    estimator = ml_models.get_model(model_name, init_kwargs=init_kwargs)
    model = estimator.model
    plot_model(model, to_file=run_outdir/f'{model_name}.png')
    model.summary(print_fn=lg.logger.info)

    # CycleLR
    lg.logger.info('\nIterations per epoch: {:.1f}'.format( xtr.shape[0]/batch_size ))
    if opt_name == 'clr_trng1':
        clr = CyclicLR(base_lr=base_lr, max_lr=max_lr, mode='triangular')
    elif opt_name == 'clr_trng2':
        clr = CyclicLR(base_lr=base_lr, max_lr=max_lr, mode='triangular2')
    elif opt_name == 'clr_exp':
        clr = CyclicLR(base_lr=base_lr, max_lr=max_lr, mode='exp_range', gamma=gamma) # 0.99994; 0.99999994; 0.999994
                
    # Checkpointer
    #model_checkpoint_dir = fold_outdir/'models'
    #os.makedirs(model_checkpoint_dir, exist_ok=True)
    #checkpointer = ModelCheckpoint(str(model_checkpoint_dir/'model.ep_{epoch:d}-val_loss_{val_loss:.5f}.h5'), save_best_only=False)
    checkpointer = ModelCheckpoint(str(fold_outdir/'best_model.h5'), monitor='mae', save_best_only=True)

    # Callbacks
    csv_logger = CSVLogger(fold_outdir/'training.log')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.75, patience=20, verbose=1, mode='auto',
                                  min_delta=0.0001, cooldown=3, min_lr=0.000000001)
    early_stop = EarlyStopping(monitor='val_loss', patience=100, verbose=1, mode='auto')
                
    # Callbacks list
    callback_list = [checkpointer, csv_logger, early_stop, reduce_lr]
    if 'clr' in opt_name: callback_list = callback_list + [clr]

    # Fit params
    fit_kwargs['validation_data'] = (xvl_dct, yvl)
    fit_kwargs['callbacks'] = callback_list

    # Train
    t0 = time()
    history = model.fit(xtr_dct, ytr, **fit_kwargs)
    lg.logger.info('Train runtime: {:.1f} mins'.format( (time()-t0)/60 ))

    # Dump model
    model.save( str(fold_outdir/'final_model.h5') )

    # Predict
    pred_ytr = model.predict(xtr)
    pred_yvl = model.predict(xvl)
    pred_yte = model.predict(xte)

    # Calc scores
    lg.logger.info(f'\nScores {model_name}:')
    scores = utils.calc_scores(ytr, pred_ytr, yvl, pred_yvl, yte, pred_yte, logger=lg.logger)
    utils.dump_dict(scores, outpath=fold_outdir/'nn_scores.txt')

    # Plots
    ml_models.plot_prfrm_metrics(history, title=f'Train base model {model_name}',
                                 skp_ep=skp_ep, add_lr=True, outdir=fold_outdir)

    ml_models.save_krs_history(history, fold_outdir)


    #tr_scores = utils.calc_scores_(ytr, pred_ytr)
    #vl_scores = utils.calc_scores_(yvl, pred_yvl)
    #te_scores = utils.calc_scores_(yte, pred_yte)
    #scores = []
    #scores.append( pd.DataFrame([utils.calc_scores_(ytr, pred_ytr)], index='tr').T )
    #scores.append( pd.DataFrame([utils.calc_scores_(yvl, pred_yvl)], index='vl').T )
    #scores.append( pd.DataFrame([utils.calc_scores_(yte, pred_yte)], index='te').T )

    # CSV scores
    te_scores = utils.calc_scores_(yte, pred_yte)
    csv = utils.scores_to_csv(model, scaler, args, te_scores, file_path)
    csv.to_csv( run_outdir/(f'csv_scores_nn_{src}.csv'), index=False )


    # =====================================================
    #       Train LGBM
    # =====================================================
    lg.logger.info('\n{}'.format('=' * 50))
    lg.logger.info('Train LGBM ...')
    
    # Define and train model
    init_kwargs = {'objective': 'regression', 'n_estimators': 100, 'n_jobs': n_jobs, 'random_state': SEED}    
    model = lgb.LGBMModel(**init_kwargs)

    t0 = time()
    fit_kwargs = {'eval_set': (xvl, yvl), 'early_stopping_rounds': 10, 'verbose': False}
    model.fit(xtr, ytr, **fit_kwargs)
    lg.logger.info('Train time: {:.1f} mins'.format( (time()-t0)/60 ))

    # Predict
    pred_ytr = model.predict(xtr)
    pred_yvl = model.predict(xvl)
    pred_yte = model.predict(xte)

    # Calc scores
    lg.logger.info('\nScores LGBM:')
    lgbm_scores = utils.calc_scores(ytr, pred_ytr, yvl, pred_yvl, yte, pred_yte, logger=lg.logger)
    utils.dump_dict(lgbm_scores, outpath=run_outdir/'lgbm_scores.txt')

    # CSV scores
    te_scores = utils.calc_scores_(yte, pred_yte)
    csv = utils.scores_to_csv(model, scaler, args, te_scores, file_path)
    csv.to_csv( run_outdir/(f'csv_scores_lgbm_{src}.csv'), index=False )


    # =====================================================
    #       Train RF
    # =====================================================
    #lg.logger.info('\n{}'.format('-' * 50))
    #lg.logger.info('Train RF Regressor (baseline) ...')

    ## Define and train model
    #init_kwargs = {'n_estimators': 100, 'n_jobs': n_jobs, 'random_state': SEED, 'min_samples_split': 7}    
    #model = RandomForestRegressor(**init_kwargs)

    #t0 = time()
    #model.fit(xtr, ytr)
    #lg.logger.info('Train time: {:.1f} mins'.format( (time()-t0)/60 ))

    ## Predict
    #pred_ytr = model.predict(xtr).squeeze()
    #pred_yvl = model.predict(xvl).squeeze()
    #pred_yte = model.predict(xte).squeeze()

    ## Append scores
    #lg.logger.info('\nScores RF Regressor:')
    #scores = utils.calc_scores(ytr, pred_ytr, yvl, pred_yvl, yte, pred_yte, logger=lg.logger)
    #utils.dump_dict(scores, run_outdir/'rf_scores.txt')


    # Finish and kill logger
    lg.kill_logger()


def main(args):
    args = parse_args(args)
    args = vars(args)
    ret = run(args)
    

if __name__ == '__main__':
    """ __name__ == '__main__' explained: www.youtube.com/watch?v=sugvnHA7ElY """
    main(sys.argv[1:])


