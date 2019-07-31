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

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.externals import joblib
from sklearn.metrics import r2_score, mean_absolute_error, median_absolute_error, explained_variance_score
from scipy.stats import spearmanr

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
    parser.add_argument('--tl_src', default='PDM', type=str, choices=['CCLE', 'CTRP', 'gCSI', 'GDSC', 'NCI60', 'all'],
                        help='Data name for which we apply transfer learning (default: PDM).')
    parser.add_argument('--tl_method', default='fe', type=str, choices=['ft', 'fe'], help="'ft': finetune, 'fe': feature extractor")

    parser.add_argument('--trial', default=0, type=int, help='PDM trial (there are 10 trial, in each trial 10 cv splits; default: 0).')

    parser.add_argument('--drg_subset', default='all', choices=['pdm', 'common', 'all'], help='Drug subset to use for training (default: all).')

    parser.add_argument('-ml', '--model_name', type=str, default='nn_reg0', help="Model to use train baseline NN model (default: 'nn_reg0').")
    parser.add_argument('-ep', '--epochs', default=50, type=int, help='Epochs (default: 50).')
    parser.add_argument('-b', '--batch_size', default=32, type=float, help='Batch size (default: 32).')
    parser.add_argument('--dr_rate', default=0.2, type=float, help='Dropout rate (default: 0.2).')

    parser.add_argument('--opt', default='clr_exp', type=str, choices=['sgd', 'adam', 'clr_trng1', 'clr_trng2', 'clr_exp'], help="Optimizer name (default: 'clr_exp').")
    parser.add_argument('--base_lr', type=float, default=1e-4, help='Base lr for cycle lr.')
    parser.add_argument('--max_lr', type=float, default=1e-3, help='Max lr for cycle lr.')
    parser.add_argument('--gamma', type=float, default=0.999994, help='Gamma parameter for learning cycle LR (default: 0.999994).')

    parser.add_argument('-kp_to', '--keep_up_to', type=str, default='a4', help='Feature extraction (how many layers to keep).')
    parser.add_argument('-frz_to', '--freeze_up_to', type=str, default='drp4', help='Finetuning (until what layer to freeze the network).')

    parser.add_argument('--skp_ep', type=int, default=3, help='Number of epochs to skip when plotting training curves.')
    parser.add_argument('--n_jobs', default=8, type=int, help='Number of cpu workers (default: 4).')
    args = parser.parse_args(args)
    return args


def create_outdir(outdir, args):
    t = datetime.now()
    t = [t.year, '-', t.month, '-', t.day, '_', 'h', t.hour, '-', 'm', t.minute]
    t = ''.join([str(i) for i in t])
    
    l = [args['bs_src'] + '_to_'+args['tl_src']] + [args['tl_method']] + [args['opt']] +  ['ep'+str(args['epochs'])] + ['drp'+str(args['dr_rate'])]
        
    name_sffx = '.'.join( l )
    outdir = Path(outdir) / (name_sffx + '_' + t)
    os.makedirs(outdir)
    return outdir


def run(args):
    base_model_dir = Path(args['base_model_dir'])
    tl_src = args['tl_src']
    tl_method = args['tl_method']
    trial = args['trial']

    ccl_fea_list = ['geneGE']
    drg_fea_list = ['DD']
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
    
    # Extract data sources used for base model training and transfer learning
    bs_src = str(base_model_dir).split('/')[-1].split('.')[0]
    args['bs_src'] = bs_src
    
    # Split of the dataset used for TRANSFER LEARNING
    bs_fold = 0
    
    # Transfer learning parameters
    keep_up_to = args['keep_up_to'] # feature extractor
    freeze_up_to = args['freeze_up_to'] # finetune
    

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
    #       Some global NN configs
    # =====================================================
    if opt_name == 'clr_trng1':
        clr = CyclicLR(base_lr=base_lr, max_lr=max_lr, mode='triangular')
    elif opt_name == 'clr_trng2':
        clr = CyclicLR(base_lr=base_lr, max_lr=max_lr, mode='triangular2')
    elif opt_name == 'clr_exp':
        clr = CyclicLR(base_lr=base_lr, max_lr=max_lr, mode='exp_range', gamma=gamma) # 0.99994; 0.99999994; 0.999994    

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.75, patience=20, verbose=1, mode='auto',
                                  min_delta=0.0001, cooldown=3, min_lr=0.000000001)
    early_stop = EarlyStopping(monitor='val_loss', patience=100, verbose=1, mode='auto')


    # =====================================================
    #       Load data for Transfer Learning
    # =====================================================
    data = utils.load_data(file_path, src=tl_src,
            ccl_fea_list=ccl_fea_list, drg_fea_list=drg_fea_list,
            drg_subset=drg_subset,
            fea_sep=fea_sep, logger=lg.logger)

    if tl_src == 'PDM':
        main_folds_dir = Path(file_path/f'../../data/yitan/PDM_10Fold_Partition/trial_{trial}/')
        matched_to = bs_src
    else:
        main_folds_dir = Path(file_path/f'../../data/yitan/CCL_10Fold_Partition/{tl_src}/')
        matched_to = None

    # These lists contain scores for all folds
    all_bs_lgbm_scores = []
    all_bs_rf_scores = []
    #all_bs_nn_scores = []
    all_tl_scores = []

    # These lists contain preds for all folds
    all_bs_lgbm_preds = []
    all_bs_rf_preds = []
    #all_bs_nn_preds = []
    all_tl_preds = []

    for tl_fold in range(10):
        lg.logger.info('\n{}'.format('=' * 70))
        lg.logger.info('Start fold {}'.format(tl_fold))
        ids_path = main_folds_dir/f'cv_{tl_fold}' 
        data_tr, data_vl, data_te = utils.get_splits_per_fold(data, ids_path, logger=None)

        xtr, ytr, mtr = utils.extract_data(data_tr, fea_list = ccl_fea_list+drg_fea_list, matched_to=matched_to)
        xvl, yvl, mvl = utils.extract_data(data_vl, fea_list = ccl_fea_list+drg_fea_list, matched_to=matched_to)
        xte, yte, mte = utils.extract_data(data_te, fea_list = ccl_fea_list+drg_fea_list, matched_to=matched_to)

        # Create dir to save results for the current fold
        fold_outdir = run_outdir/f'cv{tl_fold}'
        os.makedirs(fold_outdir)
         

        # =====================================================
        #       Scale (CV can start here) --> TODO: don't scale embeddeings
        # =====================================================
        scalerpath = base_model_dir/f'cv{bs_fold}'/'scaler.pkl'
        scaler = joblib.load(scalerpath)
        
        cols = xtr.columns
        xtr = pd.DataFrame( scaler.transform(xtr), columns=cols, dtype=np.float32 )
        xvl = pd.DataFrame( scaler.transform(xvl), columns=cols, dtype=np.float32 )
        xte = pd.DataFrame( scaler.transform(xte), columns=cols, dtype=np.float32 )
        # joblib.dump(scaler, outdir/'scaler.pkl')
            

        # =====================================================
        #      Train baseline (w/o transfer learning) 
        # =====================================================            
        lg.logger.info('\n{}'.format('-' * 50))
        lg.logger.info('Train LGBM (baseline) ...')
        baseline_lgbm_dir = fold_outdir/'baseline_lgbm'
        os.makedirs(baseline_lgbm_dir)

        # Define and train model
        init_kwargs = {'objective': 'regression', 'n_estimators': 100, 'n_jobs': n_jobs, 'random_state': SEED}    
        model = lgb.LGBMModel(**init_kwargs)

        t0 = time()
        fit_kwargs = {'eval_set': (xvl, yvl), 'early_stopping_rounds': 10, 'verbose': False}
        model.fit(xtr, ytr, **fit_kwargs)
        lg.logger.info('Train time: {:.1f} mins'.format( (time()-t0)/60 ))

        # Predict
        pred_ytr = model.predict(xtr).squeeze()
        pred_yvl = model.predict(xvl).squeeze()
        pred_yte = model.predict(xte).squeeze()

        # Append preds
        ptr = mtr.copy(); ptr['fold']=tl_fold; ptr['phase']='tr'; ptr['auc_pred']=pred_ytr
        pvl = mvl.copy(); pvl['fold']=tl_fold; pvl['phase']='vl'; pvl['auc_pred']=pred_yvl
        pte = mte.copy(); pte['fold']=tl_fold; pte['phase']='te'; pte['auc_pred']=pred_yte
        all_bs_lgbm_preds.append( pd.concat([ptr, pvl, pte], axis=0) )
            
        # Append scores
        lg.logger.info('\nBaseline LGBM:')
        scores = utils.calc_scores(ytr, pred_ytr, yvl, pred_yvl, yte, pred_yte, logger=lg.logger)
        utils.dump_dict(scores, baseline_lgbm_dir/'lgbm_baseline_scores.txt')
        all_bs_lgbm_scores.append(scores) # agg scores
        del model, scores, pred_ytr, pred_yvl, pred_yte, ptr, pvl, pte 


        # =====================================================
        #       Load base (pre-trained) model and predict w/o transfer learning
        # =====================================================    
        modelpath = base_model_dir/f'cv{bs_fold}'/'final_model.h5'
        model = load_model(str(modelpath), custom_objects={'r2_krs': r2_krs})
        plot_model(model, to_file=fold_outdir/'nn_model.png')
        

        # =====================================================
        #       Transfer learning
        # =====================================================            
        # Reset model
        lr_new = 5e-4
        model.compile(loss='mean_squared_error', optimizer=SGD(lr=lr_new, momentum=0.9), metrics=['mae']) 
        
        # Callbacks
        checkpointer = ModelCheckpoint(str(run_outdir/'model_best.h5'), verbose=0, save_weights_only=False, save_best_only=True)
        csv_logger = CSVLogger(fold_outdir/'training.log')

        # Callbacks list
        callback_list = [checkpointer, csv_logger, early_stop, reduce_lr]
        if 'clr' in opt_name: callback_list = callback_list + [clr]      
        
        # ------------------------------- 
        # Choose transfer learning method    
        # ------------------------------- 
        if tl_method=='fe':
            # Feature extractor
            # utils.print_trainable_layers(model, print_all=True)
            utils.pop_layers(model, keep_up_to=keep_up_to)
            # utils.print_trainable_layers(model, print_all=True)
            
            # Create feature extractor model
            fea_extractor_model = Model(inputs=model.input, outputs=model.layers[-1].output)

            # Extract featues
            xtr_fea_ext = fea_extractor_model.predict(xtr)
            xvl_fea_ext = fea_extractor_model.predict(xvl)
            xte_fea_ext = fea_extractor_model.predict(xte)
            # print(xvl_fea_ext.shape)
            
            ## Train final model on extracted features
            #lg.logger.info('\n{}'.format('-' * 50))
            #lg.logger.info('Train LGBM feature extractor ...')

            #init_kwargs = {'objective': 'regression', 'n_estimators': 100, 'n_jobs': n_jobs, 'random_state': SEED}    
            #model = lgb.LGBMModel(**init_kwargs)

            #t0 = time()
            #fit_kwargs = {'eval_set': (xvl_fea_ext, yvl), 'early_stopping_rounds': 10, 'verbose': False}
            #model.fit(xtr_fea_ext, ytr, **fit_kwargs)
            #lg.logger.info('Train time: {:.1f} mins'.format( (time()-t0)/60 ))

            # Train final model on extracted features
            lg.logger.info('\n{}'.format('-' * 50))
            lg.logger.info('Train RF feature extractor ...')

            init_kwargs = {'n_estimators': 100, 'n_jobs': n_jobs, 'random_state': SEED, 'min_samples_split': 7}    
            model = RandomForestRegressor(**init_kwargs)
            
            t0 = time()
            model.fit(xtr_fea_ext, ytr)
            lg.logger.info('Train time: {:.1f} mins'.format( (time()-t0)/60 ))

            # Predict
            pred_ytr = model.predict(xtr_fea_ext)
            pred_yvl = model.predict(xvl_fea_ext)
            pred_yte = model.predict(xte_fea_ext)

            # Append preds
            ptr = mtr.copy(); ptr['fold']=tl_fold; ptr['phase']='tr'; ptr['auc_pred'] = pred_ytr
            pvl = mvl.copy(); pvl['fold']=tl_fold; pvl['phase']='vl'; pvl['auc_pred'] = pred_yvl
            pte = mte.copy(); pte['fold']=tl_fold; pte['phase']='te'; pte['auc_pred'] = pred_yte
            all_tl_preds.append( pd.concat([ptr, pvl, pte], axis=0) )
                
            # Append scores
            lg.logger.info('\nScores Feature extractor LGBM:')
            tl_scores = utils.calc_scores(ytr, pred_ytr, yvl, pred_yvl, yte, pred_yte, logger=lg.logger)
            utils.dump_dict(tl_scores, outpath=fold_outdir/'fe_scores.txt')
            
        elif tl_method=='ft':
            # Finetune
            lg.logger.info('\n{}'.format('-' * 50))
            lg.logger.info('Finetune NN ...')

            # utils.print_trainable_layers(model, print_all=True)
            utils.freeze_layers(model, freeze_up_to=freeze_up_to)
            # utils.print_trainable_layers(model, print_all=True)
            
            # Fit params
            fit_kwargs = {'epochs': epochs, 'batch_size': batch_size, 'verbose': 1}
            # fit_kwargs['validation_data'] = (xvl, yvl)
            fit_kwargs['validation_split'] = 0.2
            fit_kwargs['callbacks'] = callback_list  
            
            t0 = time()
            history = model.fit(xtr, ytr, **fit_kwargs)
            
            lg.logger.info('Train time: {:.1f} mins'.format( (time()-t0)/60 ))
            
            # Predict
            pred_ytr = model.predict(xtr).squeeze()
            pred_yvl = model.predict(xvl).squeeze()
            pred_yte = model.predict(xte).squeeze()
            
            # Append preds
            ptr = mtr.copy(); ptr['fold']=tl_fold; ptr['phase']='tr'; ptr['auc_pred']=pred_ytr
            pvl = mvl.copy(); pvl['fold']=tl_fold; pvl['phase']='vl'; pvl['auc_pred']=pred_yvl
            pte = mte.copy(); pte['fold']=tl_fold; pte['phase']='te'; pte['auc_pred']=pred_yte
            all_tl_preds.append( pd.concat([ptr, pvl, pte], axis=0) )
                
            # Append scores (transfer learning)
            lg.logger.info('\nScores using finetune:')
            tl_scores = utils.calc_scores(ytr, pred_ytr, yvl, pred_yvl, yte, pred_yte, logger=lg.logger)
            utils.dump_dict(tl_scores, outpath=fold_outdir/'ft_scores.txt')
             
            ml_models.plot_prfrm_metrics(history, title=f'Train base model {model_name}',
                                         skp_ep=skp_ep, add_lr=True, outdir=fold_outdir)
            ml_models.save_krs_history(history, fold_outdir)

        del model, pred_ytr, pred_yvl, pred_yte, ptr, pvl, pte 
        all_tl_scores.append(tl_scores) # agg scores


    def scores_to_file(scores, outpath, rnd=5):
        """ Dump scores to file. """
        scores = pd.DataFrame.from_dict(scores).T
        scores.columns = ['fold'+str(f+1) for f in range(scores.shape[1])]
        scores.insert(loc=0, column='std', value=scores.std(axis=1))
        scores.insert(loc=0, column='mean', value=scores.iloc[:,1:].mean(axis=1))
        scores.index.name = 'metric'
        scores.reset_index(inplace=True)
        scores.round(rnd).to_csv(outpath, index=False)
    
    # Dump all scores
    scores_to_file(all_bs_lgbm_scores, run_outdir/'folds_bs_lgbm_scores.csv')
    scores_to_file(all_bs_rf_scores, run_outdir/'folds_bs_rf_scores.csv')
    scores_to_file(all_tl_scores, run_outdir/'folds_tl_scores.csv')

    # Dump all preds
    all_bs_lgbm_preds = pd.concat(all_bs_lgbm_preds, axis=0, ignore_index=True)
    all_bs_lgbm_preds.round(5).to_csv(run_outdir/'all_bs_lgbm_preds.csv', index=False)
    all_bs_rf_preds = pd.concat(all_bs_rf_preds, axis=0, ignore_index=True)
    all_bs_rf_preds.round(5).to_csv(run_outdir/'all_bs_rf_preds.csv', index=False)
    all_tl_preds = pd.concat(all_tl_preds, axis=0, ignore_index=True)
    all_tl_preds.round(5).to_csv(run_outdir/'all_tl_preds.csv', index=False)
   
    def preds_to_scores(preds, logger=None):
        scores = {}
        for ph in preds['phase'].unique():
            a = preds[ preds['phase']==ph ] 
            scores[f'r2_{ph}'] = r2_score(a['auc'], a['auc_pred'])
            scores[f'mae_{ph}'] = mean_absolute_error(a['auc'], a['auc_pred'])
            scores[f'spr_rnk_corr_{ph}'] = spearmanr(a['auc'], a['auc_pred'])[0]
        
        scores = {k: scores[k] for k in sorted(scores.keys())}
        for k, v in scores.items(): scores[k] = round(v, 4)
        if logger is not None:
            for k, v, in scores.items(): logger.info(f'{k}: {v}')
        return scores

    lg.logger.info('\nLGBM basline scores:')
    lgbm_scores = preds_to_scores(all_bs_lgbm_preds, logger=lg.logger)
    lg.logger.info('\nRF basline scores:')
    rf_scores = preds_to_scores(all_bs_rf_preds, logger=lg.logger)
    lg.logger.info('\nTransfer learning scores:')
    tl_scores = preds_to_scores(all_tl_preds, logger=lg.logger)

    utils.dump_dict(lgbm_scores, run_outdir/'lgbm_baseline_scores.txt')
    utils.dump_dict(rf_scores, run_outdir/'rf_baseline_scores.txt')
    utils.dump_dict(tl_scores, run_outdir/'transfer_learn_scores.txt')


    lg.kill_logger()


def main(args):
    args = parse_args(args)
    args = vars(args)
    ret = run(args)
    

if __name__ == '__main__':
    """ __name__ == '__main__' explained: www.youtube.com/watch?v=sugvnHA7ElY """
    main(sys.argv[1:])


