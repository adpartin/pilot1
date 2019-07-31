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

from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, median_absolute_error, explained_variance_score
from scipy.stats import spearmanr


def dump_dict(dct, outpath='./dict.txt'):
    """ Dump dict into file. """
    dct = {k: dct[k] for k in sorted(dct.keys())}
    with open( Path(outpath), 'w' ) as file:
        for k, v in dct.items():
            file.write('{}: {}\n'.format(k, v))            
            

def preds_to_scores(preds, logger=None):
    """ preds is a df with true values, predictions, and training phase. """
    scores = {}
    target_name = 'auc'
    for ph in preds['phase'].unique():
        a = preds[ preds['phase']==ph ] 
        scores[f'r2_{ph}'] = r2_score( a[target_name], a[f'{target_name}_pred'] )
        scores[f'mae_{ph}'] = mean_absolute_error( a[target_name], a[f'{target_name}_pred'] )
        scores[f'spr_rnk_corr_{ph}'] = spearmanr( a[target_name], a[f'{target_name}_pred'] )[0]
    
    scores = {k: scores[k] for k in sorted(scores.keys())}
    for k, v in scores.items(): scores[k] = round(v, 4)
    if logger is not None:
        for k, v, in scores.items(): logger.info(f'{k}: {v}')
    return scores
        

def calc_scores(ytr, pred_ytr, yvl, pred_yvl, yte, pred_yte, logger=None):
    """ This func is too specific (instead, use preds_to_scores). """
    scores = {}
    
    #ys = {'ytr': ytr, 'pred_ytr': pred_ytr, 'yvl': yvl, 'pred_yvl': pred_yvl, 'yte': yte, 'pred_yte': pred_yte}
    #ys = {'ytr': ytr, 'yvl': yvl, 'yte': yte}
    #for k, v in ys.items():
    #    if v is not None:
    #        scores[f'r2_{}'] = r2_score(ytr, pred_ytr)
    #        scores[f'mae_{}'] = mean_absolute_error(ytr, pred_ytr)
    #        scores[f'spr_rnk_corr_{}'] = spearmanr(yvl, pred_yvl)[0]
    
    scores['r2_tr'] = r2_score(ytr, pred_ytr)
    scores['r2_vl'] = r2_score(yvl, pred_yvl)
    scores['r2_te'] = r2_score(yte, pred_yte)
    scores['mae_tr'] = mean_absolute_error(ytr, pred_ytr)
    scores['mae_vl'] = mean_absolute_error(yvl, pred_yvl)
    scores['mae_te'] = mean_absolute_error(yte, pred_yte)
    scores['spr_rnk_corr_tr'] = spearmanr(ytr, pred_ytr)[0]
    scores['spr_rnk_corr_vl'] = spearmanr(yvl, pred_yvl)[0]
    scores['spr_rnk_corr_te'] = spearmanr(yte, pred_yte)[0]

    for k, v in scores.items(): scores[k] = round(v, 4)
    if logger is not None:
        for k, v, in scores.items(): logger.info(f'{k}: {v}')
    return scores


def calc_scores_(y_true, y_pred):
    """ This func is too specific (instead, use preds_to_scores). """
    scores = {}
    scores['r2'] = r2_score(y_true, y_pred)
    scores['mae'] = mean_absolute_error(y_true, y_pred)
    scores['spr_rnk_corr'] = spearmanr(y_true, y_pred)[0]
    for k, v in scores.items(): scores[k] = round(v, 4)
    return scores


def scores_to_csv(model, scaler, args, te_scores, file_path):
    csv = []
    all_sources = ['CCLE', 'CTRP', 'gCSI', 'GDSC', 'NCI60']
    #all_sources = ['CCLE', 'gCSI']
    for s in all_sources:
        print(f'Calc scores for {s} ...')
        if args['src'] == s:
            scores = te_scores
        else:
            data = load_data(file_path, src=s,
                    ccl_fea_list=args['ccl_fea'], drg_fea_list=args['drg_fea'],
                    drg_subset=args['drg_subset'])
            x, y, m = extract_data(data, fea_list = args['ccl_fea']+args['drg_fea'])
            x = pd.DataFrame( scaler.transform(x), columns=x.columns ).astype(np.float32) # scale
            y_pred = model.predict(x) # make pred
            scores = calc_scores_(y, y_pred) # calc scores
        
        csv.append( pd.DataFrame([scores], index=[s]).T ) # agg scores

    # Organize scores
    csv = pd.concat(csv, axis=1)
    csv.insert(loc=0, column='tr_src', value=args['src'])
    csv = csv.reset_index().rename(columns={'index': 'metric'})
    csv = csv.round(4)
    return csv


# Functions for transfer leaering with Keras -----------------
def print_trainable_layers(model, print_all=False):
    """ Print the trainable state of layers. """
    print('Trainable layers:')
    for layer in model.layers:
        if layer.trainable:
            print(layer.name, layer.trainable)
        if not layer.trainable and print_all:
            print(layer.name, layer.trainable)

            
def freeze_layers(model, freeze_up_to='all'):
    """ Freeze up to layer freeze_up_to, including! """
    # freeze_layers = ['1', '2', '3', '4']
    if freeze_up_to=='all':
        for layer in model.layers:
            layer.trainable = False

    #for layer in model.layers:
    #    if any([True for i in layers_ids if i in layer.name]):
    #        layer.trainable = False
    for layer in model.layers:
        if freeze_up_to.lower() != layer.name.lower():
            layer.trainable = False
        else:
            layer.trainable = False
            break

def pop_layers(model, keep_up_to):
    # pop_layers = ['4', '5', 'outputs']
    model_layers = model.layers
    #for layer in model_layers[::-1]:
    #    if any([True for i in layers_ids if i in layer.name]):
    #        model.layers.pop()  
    for layer in model_layers[::-1]:
        if keep_up_to.lower() != layer.name.lower():
            model.layers.pop()
        else:
           break
# ------------------------------------------------------------



# Functions for loading Yitan's data -------------------------
def cnt_fea(df, fea_sep='_', verbose=True, logger=None):
    """ Count the number of features per feature type. """
    dct = {}
    unq_prfx = df.columns.map(lambda x: x.split(fea_sep)[0]).unique() # unique feature prefixes
    for prfx in unq_prfx:
        fea_type_cols = [c for c in df.columns if (c.split(fea_sep)[0]) in prfx] # all fea names of specific type
        dct[prfx] = len(fea_type_cols)
    
    if verbose and logger is not None:
        logger.info(pformat(dct))
    elif verbose:
        pprint(dct)
    return dct


def extract_subset_fea(df, fea_list, fea_sep='_'):
    """ Extract features based feature prefix name. """
    fea = [c for c in df.columns if (c.split(fea_sep)[0]) in fea_list]
    return df[fea]


def extract_data(df, fea_list, matched_to=None):
    """ ... """
    X = extract_subset_fea(df, fea_list=fea_list, fea_sep='_')
    if matched_to is not None:
        Y = df[[f'Drug_Response_Matched_To_{matched_to}']]
    else:
        Y = df[['auc']]
    meta = df.drop(columns=X.columns)
    # meta = meta.drop(columns=['auc'])
    return X, Y, meta


def load_data(file_path, src, ccl_fea_list, drg_fea_list, drg_subset='all', fea_sep='_', logger=None):
    """ 
    Args:
        drg_subset : 
            'all' (any drug)
            'common' (intersection of drugs used in PDM and CCL)
            'pdm' (only drugs used in PDM study)
    """
    #""" Load data (Yitan's data and splits) """
    datadir = Path(file_path/'../../data/yitan/Data')
    #ccl_folds_dir = Path(file_path/'../../data/yitan/CCL_10Fold_Partition')
    #pdm_folds_dir = Path(file_path/'../../data/yitan/PDM_10Fold_Partition')
    fea_data_name = 'CCL_PDM_TransferLearningData_rmFactor_0.0_ddNorm_std.pkl'
    
    # Un-pickle files
    import _pickle as cp
    pkl_file = open(datadir/fea_data_name, 'rb')
    res = cp.load(pkl_file)
    ccl = cp.load(pkl_file)
    drg = cp.load(pkl_file)
    pkl_file.close()

    #lg.logger.info('\n{}'.format('=' * 50))
    #lg.logger.info('res: {}'.format(res.shape))
    #lg.logger.info('ccl: {}'.format(ccl.shape))
    #lg.logger.info('drg: {}'.format(drg.shape))
    
    if src == 'PDM':
        res = pd.read_csv(datadir/'../Standardized_PDM_Data.txt', sep='\t')

    # Update resp
    res = res.reset_index()
    res = res.rename(columns={'index': 'idx', 'ccl_name': 'cclname', 'SOURCE': 'src', 'area_under_curve': 'auc'})    
    if logger: logger.info('\n{}'.format( res.groupby('src').agg({'cclname': 'nunique', 'ctrpDrugID': 'nunique'}).reset_index() ))

    # Get subset of drugs
    if drg_subset != 'all':
        pdm_drg_names = res.loc[res['src']=='PDM', 'ctrpDrugID'].unique()
        if drg_subset == 'pdm':
            res = res[res['ctrpDrugID'].isin(pdm_drg_names)]
        elif drg_subset == 'common': # drugs that common for pdm and ccl
            drg_ccl_names = res.loc[res['src']!='PDM', 'ctrpDrugID'].unique()
            common_drgs_names = set(drg_ccl_names).intersection(set(pdm_drg_names))
            res = res[res['ctrpDrugID'].isin(common_drgs_names)]

    # Retain specific source
    if logger: logger.info('\nFull dataset: {}'.format(res.shape))
    if src!='all':  res = res[ res['src'].isin([src]) ]
    if logger: logger.info('Only {}: {}'.format(src, res.shape))

    # Extract subset of cell line features
    if logger: logger.info('\nExtract fea types ...')
    if logger: cnt_fea(ccl, logger=logger);
    ccl = extract_subset_fea(df=ccl, fea_list=ccl_fea_list, fea_sep=fea_sep)
    if logger: cnt_fea(ccl, logger=logger);

    # Extract subset of drug features
    if 'lbl' in drg_fea_list:
        # TODO: didn't finish; need to test
        #drg_enc = drg.copy()
        #drg_enc.insert(loc=0, column='drg_enc', value=LabelEncoder().fit_transform(drg_enc.index), allow_duplicates=False)
        #drg = drg_enc[['drg_enc']]
        
        drg = pd.DataFrame(index=drg.index)
        drg['lbl_drg'] = LabelEncoder().fit_transform(drg.index)
    else:
        if logger: cnt_fea(drg, logger=logger);
        drg = extract_subset_fea(df=drg, fea_list=drg_fea_list, fea_sep=fea_sep)
        if logger: cnt_fea(drg, logger=logger);

    def merge_dfs(res_df, ccl_df, drg_df):
        """ Merge the following dfs: response, ccl fea, drug fea """
        mrg_df = pd.merge(res_df, ccl_df, on='cclname', how='inner')
        mrg_df = pd.merge(mrg_df, drg_df, on='ctrpDrugID', how='inner')
        mrg_df.reset_index(drop=True)
        return mrg_df

    # Bring the labels in, in order to merge on
    ccl = ccl.reset_index().rename(columns={'index': 'cclname'})
    drg = drg.reset_index().rename(columns={'index': 'ctrpDrugID'})

    # Drop categorical vars that comes with the df
    # TODO: not sure if this is better!
    drg_cols = [c for c in drg.columns if c.split('|')[-1]!='int']
    drg = drg[drg_cols]

    if logger: logger.info('\nMerge ...')
    data = merge_dfs(res, ccl, drg)
    if logger: logger.info('data: {}'.format(data.shape))
    return data


def get_splits_per_fold(data, ids_path, logger=None):

    def get_file(fpath):
        return pd.read_csv(fpath, header=None).squeeze().values if fpath.is_file() else None

    """ Load data (Yitan's data and splits) """
    tr_ids_list = get_file(ids_path/'TrainList.txt')
    vl_ids_list = get_file(ids_path/'ValList.txt')
    te_ids_list = get_file(ids_path/'TestList.txt')

    def get_dat_subset(df, colname, lst=None):
        return df[ data[colname].isin( lst ) ].reset_index(drop=True) if lst is not None else None
       
    colname = 'groupID' if 'PDM_10Fold_Partition' in str(ids_path) else 'cclname'
    data_tr = get_dat_subset(data, colname, lst=tr_ids_list)
    data_vl = get_dat_subset(data, colname, lst=vl_ids_list)
    data_te = get_dat_subset(data, colname, lst=te_ids_list)
    
    def log_shape(df, name, logger):
        if df is not None: logger.info('{} {}'.format(name, df.shape))

    if logger is not None:
        log_shape(data_tr, 'data_tr', logger)
        log_shape(data_vl, 'data_vl', logger)
        log_shape(data_te, 'data_te', logger)
    return data_tr, data_vl, data_te
# ------------------------------------------------------------


