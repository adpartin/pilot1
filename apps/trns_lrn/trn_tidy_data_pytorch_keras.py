"""
Train NN drug response predictor with tidy dataset.
Both PyTorch and Keras are supported.
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

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import sklearn
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, median_absolute_error, explained_variance_score

import lightgbm as lgb

import torch
from torch import nn, optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import copy

SEED=42


# File path
file_path = Path(__file__).resolve().parent


# Utils
utils_path = file_path / '../../utils'
sys.path.append(str(utils_path))
import utils
from classlogger import Logger


# Path
PRJ_NAME = file_path.name 
OUTDIR = file_path / '../../out/' / PRJ_NAME


def parse_args(args):
    parser = argparse.ArgumentParser(description="Compare keras and pytorch.")

    # Input data
    parser.add_argument('--dirpath', default=None, type=str, help='Full path to data and split (default: None).')

    parser.add_argument('--dname', default='ytn', type=str, choices=['top6', 'ytn'], help='Dataset name (default: ytn).')
    parser.add_argument('--frm', default='trch', type=str, choices=['krs', 'trch'], help='DL framework (default: trch).')
    # parser.add_argument('--src', default='GDSC', type=str, help='Data source (default: GDSC).')

    parser.add_argument('-ep', '--epochs', default=250, type=int, help='Epochs (default: 250).')
    parser.add_argument('-b', '--batch_size', default=32, type=float, help='Batch size (default: 32).')
    parser.add_argument('--dr_rate', default=0.2, type=float, help='Dropout rate (default: 0.2).')

    parser.add_argument('--n_jobs', default=4, type=int, help='Number of cpu workers (default: 4).')
    args = parser.parse_args(args)
    return args


def create_outdir(outdir, args):
    t = datetime.now()
    t = [t.year, '-', t.month, '-', t.day, '_', 'h', t.hour, '-', 'm', t.minute]
    t = ''.join([str(i) for i in t])
    
    l = [args['src']] + [args['frm']] + ['ep'+str(args['epochs'])] + ['drp'+str(args['dr_rate'])] 
        
    name_sffx = '.'.join( l )
    outdir = Path(outdir) / (name_sffx + '_' + t)
    os.makedirs(outdir)
    return outdir


def r2_torch(y_true, y_pred):
    """ TODO: consider to convert tensors """
    epsilon = 1e-7  # this epsilon value is used in TF
    SS_res = torch.sum( (y_true - y_pred)**2 )
    SS_tot = torch.sum( (y_true - torch.mean(y_true))**2 )
    r2 = 1 - SS_res / (SS_tot + epsilon)
    return r2


def update_scores_reg(pred, true, scores):
    """ Updates score metrics for regression ML predictions.
    The scores are summed for every call of the function (single func call corresponds a single batch).

    Note: these must be implemented with pytroch commands! Otherwise, results gives error:
    RuntimeError: Can't call numpy() on Variable that requires grad. Use var.detach().numpy() instead.   
    pred, true = pred.numpy(), yy.numpy()
    tr_mae += sklearn.metrics.mean_absolute_error(true, pred)
    tr_r2 += sklearn.metrics.r2_score(true, pred)
    """
    for m in scores.keys():
        if 'loss' in m:
            continue

        elif any([True if v in m else False for v in ['mean_abs_err', 'mean_absolute_error']]):
            scores[m] += torch.mean( torch.abs(pred-true) ).item()

        elif any([True if v in m else False for v in ['median_abs_err', 'median_absolute_error']]):
            scores[m] += torch.median( torch.abs(pred-true) ).item()

        elif any([True if v in m else False for v in ['mean_sqrd_err', 'mean_squared_error']]):
            scores[m] += torch.mean( torch.pow(pred-true, 0.5) ).item()  # or torch.mean(torch.sqrt(pred-true))

        elif any([True if v in m else False for v in ['r2', 'r2_score']]):
            scores[m] += r2_torch(y_true=true, y_pred=pred).item()

    return scores      


def proc_batch(x_dct, y, model, loss_fnc, opt=None):
    """ 
    Args:
        opt (torch.optim) : no backprop is performed if optimizer is not provided (for val or test) 
    """
    pred = model(**x_dct)
    pred = pred.type(y.dtype)
    loss = loss_fnc(pred, y)

    # Backward pass
    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()

    return loss, pred      


def fit(model: nn.Module,
        loss_fnc,
        opt: torch.optim,
        tr_dl: torch.utils.data.DataLoader,
        vl_dl: torch.utils.data.DataLoader=None,
        epochs: int=1,
        device: torch.device='cuda:0',
        verbose: bool=True,
        metrics=[]) -> dict:
    """ github.com/stared/livelossplot/blob/master/examples/pytorch.ipynb
    Args:
        metrics (list) : list of metric scores to log
            (available metrics: 'mean_abs_err','median_abs_err', 'mean_sqrd_err', 'r2)
    """ 
    # Create dicts to log scores
    if vl_dl is None:
        logs = OrderedDict({'loss': []})
        logs.update(OrderedDict({m: [] for m in metrics}))        
    else:
        logs = OrderedDict({'loss': [], 'val_loss': []})
        for m in metrics: logs.update(OrderedDict({m: [], 'val_'+m: []}))

    best_score = -np.inf # torch.Tensor([float('inf')])
            
    # Iter over epochs
    phases = ['train', 'val'] if vl_dl is not None else ['train']
    for ep in range(epochs):
        ep_t0 = time()
        # lr_scheduler.step()

        for ph in phases:
            if ph == 'train':
                model.train()
                dl = tr_dl
                scores = {m: 0 for m in logs.keys() if 'val' not in m}
                loss_name = 'loss'
            elif ph == 'val':
                model.eval()
                dl = vl_dl
                scores = {m: 0 for m in logs.keys() if 'val' in m}
                loss_name = 'val_loss'

            # Iter over batches
            for x, y in dl:
                y = y.to(device)
                x = x.to(device)
                x_dct = {'x': x}

                # Process batch
                if ph == 'train':
                    loss, pred = proc_batch(x_dct=x_dct, y=y, model=model, loss_fnc=loss_fnc, opt=opt)
                else:
                    loss, pred = proc_batch(x_dct=x_dct, y=y, model=model, loss_fnc=loss_fnc, opt=None)

                # Compute metrics (running avg)
                scores[loss_name] += loss.item() * x.size(0) # TODO: why should multiply by this value??
                scores = update_scores_reg(pred=pred, true=y, scores=scores)

            # Log scores
            for m in scores.keys():
                logs[m].append(scores[m]/len(dl))

            # Store best model
            if ph=='val' and scores['val_r2'] > best_score:
                best_score = scores['val_r2']
                best_model_wts = copy.deepcopy(model.state_dict())
                
            del y, x, x_dct, loss, pred, scores

        if verbose:
            print(f'Epoch {ep+1}/{epochs}; ',
                  f'{int(time()-ep_t0)}s; ',
                  [f'{k}: {v[-1]:.3f}' for k, v in logs.items()])

        # TODO: log scores into file

    return model, logs


def set_parameter_requires_grad(model, fea_extractor):
    """ This is for transfer learning. """
    if fea_extractor:
        for param in model.parameters():
            param.requires_grad = False


def weight_init_linear(m: nn.Module):
    """
    Pytorch initializes the layers by default (e.g., Linear uses kaiming_uniform_)
    www.deeplearningwizard.com/deep_learning/boosting_models_pytorch/weight_initialization_activation_functions/
    stackoverflow.com/questions/49433936/how-to-initialize-weights-in-pytorch
    github.com/xduan7/UnoPytorch/blob/master/networks/initialization/weight_init.py
    """
    if type(m) == nn.Linear:
        # torch.nn.init.xavier_uniform(m.weight)
        torch.nn.init.kaiming_normal(m.weight) # TODO: try this
        m.bias.data.fill_(0.01)


def get_model_device(model):
    return str(next(model.parameters()).device)


class Dataset_Tidy(Dataset):
    """ ... """
    def __init__(self,
                 xdata: pd.DataFrame,
                 ydata: pd.DataFrame):
        """ ... """
        xdata = pd.DataFrame(xdata).values
        ydata = pd.DataFrame(ydata).values
        self.x = torch.tensor(xdata, dtype=torch.float32)
        self.y = torch.tensor(ydata, dtype=torch.float64)
        self.y = self.y.view(-1, 1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.x[idx, :]
        y = self.y[idx]
        return x, y


class NN_Reg_Tidy(nn.Module):
    """ ... """
    def __init__(self, input_dim, dr_rate=0.2):
        """ ... """
        super().__init__()
        self.input_dim = input_dim
        self.dr_rate = dr_rate
        """
        self.fc1 = nn.Linear(self.input_dim, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 250)
        self.fc4 = nn.Linear(250, 125)
        self.fc5 = nn.Linear(125, 60)
        self.fc6 = nn.Linear(60, 30)
        self.fc7 = nn.Linear(30, 1)
        """
        self.fc1 = nn.Linear(self.input_dim, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 250)
        self.fc4 = nn.Linear(250, 60)
        self.fc5 = nn.Linear(60, 1)
        self.dropout = nn.Dropout(self.dr_rate)
  

    def forward(self, x):
        """
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = self.dropout(F.relu(self.fc4(x)))
        x = F.relu(self.fc5(x)) 
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x)) 
        return x


def run(args):
    dirpath = Path(args['dirpath'])
    dname = args['dname']
    framework = args['frm']
    # src = args['src']
    
    epochs = args['epochs']
    batch_size = args['batch_size']
    dr_rate = args['dr_rate']
    n_jobs = args['n_jobs']

    verbose = True    
    args['src'] = dirpath.name.split('.')[0]
    src = args['src'] 
    
    
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
    #       Load data
    # =====================================================
    """
    xdata_fpath = Path((glob(str(dirpath/'*xdata.parquet')))[0])
    ydata_fpath = Path((glob(str(dirpath/'*ydata.parquet')))[0])
    meta_fpath = Path((glob(str(dirpath/'*meta.parquet')))[0])
    if xdata_fpath.is_file():
        xdata = pd.read_parquet( xdata_fpath, engine='auto', columns=None )
    if ydata_fpath.is_file():
        ydata = pd.read_parquet( ydata_fpath, engine='auto', columns=None )
    if meta_fpath.is_file():
        meta = pd.read_parquet( meta_fpath, engine='auto', columns=None )
    """
    data_fpath = Path((glob(str(dirpath/'*data.parquet')))[0])
    if data_fpath.is_file():
        data = pd.read_parquet( data_fpath, engine='auto', columns=None )
        lg.logger.info('\ndata {}'.format(data.shape))


    if dname=='ytn':
        datadir = Path(file_path/'../../data/yitan/Data')
        ccl_folds_dir = Path(file_path/'../../data/yitan/CCL_10Fold_Partition')
        pdm_folds_dir = Path(file_path/'../../data/yitan/PDM_10Fold_Partition')

        fold = 0
        ccl_fea_list = ['geneGE']
        drg_fea_list = ['DD']
        # ccl_fea_list = ['ccl']
        # drg_fea_list = ['drg']
        fea_sep = '_'

        ids_path = ccl_folds_dir/f'{src}/cv_{fold}' # 'TestList.txt'
        tr_ids_list = pd.read_csv(ids_path/'TrainList.txt', header=None).squeeze().values
        vl_ids_list = pd.read_csv(ids_path/'ValList.txt', header=None).squeeze().values
        # te_ids_list = pd.read_csv(ids_path/'TestList.txt', header=None).squeeze().values

        data_tr = data[ data['cclname'].isin( tr_ids_list ) ]
        data_vl = data[ data['cclname'].isin( vl_ids_list ) ]
        # data_te = data[ data['cclname'].isin( te_ids_list ) ]

        lg.logger.info('data_tr {}'.format(data_tr.shape))
        lg.logger.info('data_vl {}'.format(data_vl.shape))
        # lg.logger.info('data_te {}'.format(data_te.shape))

        def extract_subset_fea(df, fea_list, fea_sep='_'):
            """ Extract features based feature prefix name. """
            fea = [c for c in df.columns if (c.split(fea_sep)[0]) in fea_list]
            df = df[fea]
            return df

        def extract_data(df):
            """ ... """
            X = extract_subset_fea(df, fea_list=ccl_fea_list + drg_fea_list, fea_sep='_')
            Y = df[['auc']]
            meta = df.drop(columns=X.columns)
            meta = meta.drop(columns=['auc'])
            return X, Y, meta

        xtr, ytr, mtr = extract_data(data_tr)
        xvl, yvl, mvl = extract_data(data_vl)
        # xte, yte, mte = extract_data(data_te)
        

    elif dname=='top6':
        datadir = Path(file_path/'../../data/processed/topN/topNcode/')
        datapath = datadir/'top_6.res_reg.cf_rnaseq.dd_dragon7.labled.parquet'
        data = pd.read_parquet(datapath, engine='auto', columns=None)

        # Extract source
        data = data[ data['CELL'].map(lambda x: True if x.split('.')[0]==src else False) ]
        data = data.sample(frac=1.0, axis=0, random_state=SEED).reset_index(drop=True)

        # Drop meta cols
        data.drop(columns=['CELL', 'DRUG'], inplace=True)

        # Split train/test
        df_tr, df_vl = train_test_split(data, test_size=0.2)
        df_tr = df_tr.reset_index(drop=True)
        df_vl = df_vl.reset_index(drop=True)

        # Split features/target
        ytr, xtr = df_tr.iloc[:,0], df_tr.iloc[:,1:]
        yvl, xvl = df_vl.iloc[:,0], df_vl.iloc[:,1:]
        del data, df_tr, df_vl

        # Scale
        columns = xtr.columns
        scaler = StandardScaler()
        xtr = pd.DataFrame( scaler.fit_transform(xtr), columns=columns ).astype(np.float32)
        xvl = pd.DataFrame( scaler.transform(xvl), columns=columns ).astype(np.float32)


    lg.logger.info('\nxtr.shape: {}'.format(xtr.shape))
    lg.logger.info('xvl.shape: {}'.format(xvl.shape))
    lg.logger.info('ytr.shape: {}'.format(ytr.shape))
    lg.logger.info('yvl.shape: {}'.format(yvl.shape))


    # =====================================================
    # Train with LGBM
    # =====================================================
    """
    print('\n{}'.format('=' * 50))
    print('Train with LGBM ...')
    
    # Define model
    init_kwargs = {'objective': 'regression', 'n_estimators': 100,
                   'n_jobs': n_jobs, 'random_state': SEED}    
    model = lgb.LGBMModel(**init_kwargs)

    # Train
    fit_kwargs = {'verbose': verbose}
    t0 = time()
    model.fit(xtr, ytr, **fit_kwargs)
    print('Train time: {:.3f} mins'.format( (time()-t0)/60 ))

    # Predict
    pred_ytr = model.predict(xtr)
    pred_yvl = model.predict(xvl)

    # Calc scores
    print('\nScores with LGBM:')
    lgbm_scores = {}
    lgbm_scores['r2_tr'] = r2_score(ytr, pred_ytr)
    lgbm_scores['r2_vl'] = r2_score(yvl, pred_yvl)
    lgbm_scores['mae_tr'] = mean_absolute_error(ytr, pred_ytr)
    lgbm_scores['mae_vl'] = mean_absolute_error(yvl, pred_yvl)
    for k, v, in lgbm_scores.items(): print(f'{k}: {v}')
    """


    # =====================================================
    # Choose NN framework
    # =====================================================
    if framework == 'trch':
        lg.logger.info('\n{}'.format('=' * 50))
        lg.logger.info('Train with NN Reg ...')
        lg.logger.info('PyTorch version: {}'.format(torch.__version__))
        lg.logger.info('\nCUDA info ...')
        lg.logger.info('device_name:    {}'.format( torch.cuda.get_device_name(0) ))
        lg.logger.info('device_count:   {}'.format( torch.cuda.device_count() ))
        lg.logger.info('current_device: {}'.format( torch.cuda.current_device() ))

        # Create torch datasets
        tr_ds = Dataset_Tidy(xdata=xtr, ydata=ytr)
        vl_ds = Dataset_Tidy(xdata=xvl, ydata=yvl)

        # Create data loaders
        tr_loader_kwargs = {'batch_size': batch_size, 'shuffle': True, 'num_workers': n_jobs}
        vl_loader_kwargs = {'batch_size': 4*batch_size, 'shuffle': False, 'num_workers': n_jobs}

        tr_loader = DataLoader(tr_ds, **tr_loader_kwargs)
        vl_loader = DataLoader(vl_ds, **vl_loader_kwargs)      

        # Choose device
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Create model and move to CUDA device
        model = NN_Reg_Tidy(input_dim=tr_ds.x.shape[1]).to(device) # send model to gpu/cpu device
        # model.apply(weight_init_linear)
        lg.logger.info(f'\n{model}')

        # Query device where the model is located
        print(get_model_device(model))
        print('current_device:', torch.cuda.current_device()) # why current device is 0??      

        # Loss function and optimizer
        loss_fnc = nn.MSELoss(reduction='mean')
        opt = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)  # pytorch.org/docs/stable/optim.html
        # lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer=opt, base_lr=1e-5, max_lr=1e-3, mode='triangular')

        # Define training params
        metrics = ['mean_abs_err', 'r2']
        fit_kwargs = {'epochs': epochs, 'device': device, 'verbose': verbose, 'metrics': metrics}

        # Train
        t0 = time()
        model, logs = fit(
                model=model,
                loss_fnc=loss_fnc,
                opt=opt,
                tr_dl=tr_loader,
                vl_dl=vl_loader,
                **fit_kwargs)
        print('Train time: {:.2f} mins'.format( (time()-t0)/60 ))

        # Dump logs
        logs = pd.DataFrame(logs).reset_index().rename(columns={'index': 'epoch'})
        logs['epoch'] += 1
        logs.to_csv(run_outdir/'logfile.log', index=False)

        # Dump model
        # torch.save(model.state_dict(), run_outdir)
        torch.save(model, run_outdir/'base_model.pt')

        # Predict
        xtr = torch.tensor(xtr.values, dtype=torch.float32).to(device)
        xvl = torch.tensor(xvl.values, dtype=torch.float32).to(device)
        pred_ytr = model(xtr).cpu().detach().numpy()
        pred_yvl = model(xvl).cpu().detach().numpy()

        # Calc scores
        print('\nScores with PyTorch NN:')
        nn_scores = {}
        nn_scores['r2_tr'] = r2_score(ytr, pred_ytr)
        nn_scores['r2_vl'] = r2_score(yvl, pred_yvl)
        nn_scores['mae_tr'] = mean_absolute_error(ytr, pred_ytr)
        nn_scores['mae_vl'] = mean_absolute_error(yvl, pred_yvl)
        lg.logger.info(f'\n{pformat(nn_scores)}')
        # for k, v, in nn_scores.items(): print(f'{k}: {v}')
        utils.dump_dict(nn_scores, outpath=run_outdir/'./torch_scores.txt')


    elif framework == 'krs':
        import tensorflow as tf
        import keras
        from keras import backend as K
        from keras.layers import Input, Dense, Dropout, Activation, BatchNormalization, Lambda, merge
        from keras import optimizers
        from keras.optimizers import SGD, Adam, RMSprop, Adadelta
        from keras.models import Sequential, Model, model_from_json, model_from_yaml
        from keras.utils import np_utils, multi_gpu_model
        from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard      

        def r2_krs(y_true, y_pred):
            SS_res =  K.sum(K.square(y_true - y_pred))
            SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
            return (1 - SS_res/(SS_tot + K.epsilon()))      

        def nn_reg_tidy(input_dim, dr_rate=0.2, opt_name='sgd', logger=None):
            inputs = Input(shape=(input_dim,))
            
            x = Dense(1000, activation='relu')(inputs)
            x = Dropout(dr_rate)(x)
            
            x = Dense(500, activation='relu')(x)
            x = Dropout(dr_rate)(x)
            
            x = Dense(250, activation='relu')(x)
            x = Dropout(dr_rate)(x)
            
            x = Dense(60, activation='relu')(x)
            x = Dropout(dr_rate)(x)
            
#             x = Dense(1000)(inputs)
#             x = BatchNormalization()(x)
#             x = Activation('relu')(x)

#             x = Dense(1000)(x)
#             x = BatchNormalization()(x)
#             x = Activation('relu')(x)
#             x = Dropout(dr_rate)(x)

#             x = Dense(500)(x)
#             x = BatchNormalization()(x)
#             x = Activation('relu')(x)
#             x = Dropout(dr_rate)(x)

#             x = Dense(250)(x)
#             x = BatchNormalization()(x)
#             x = Activation('relu')(x)
#             x = Dropout(dr_rate)(x)

#             x = Dense(125)(x)
#             x = BatchNormalization()(x)
#             x = Activation('relu')(x)
#             x = Dropout(dr_rate)(x)

#             x = Dense(60)(x)
#             x = BatchNormalization()(x)
#             x = Activation('relu')(x)
#             x = Dropout(dr_rate)(x)

#             x = Dense(30)(x)
#             x = BatchNormalization()(x)
#             x = Activation('relu')(x)
#             x = Dropout(dr_rate)(x)

            outputs = Dense(1, activation='relu')(x)
            model = Model(inputs=inputs, outputs=outputs)

            if opt_name == 'sgd':
                opt = SGD(lr=1e-4, momentum=0.9)
            elif opt_name == 'adam':
                opt = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
            else:
                opt = SGD(lr=1e-4, momentum=0.9) # for clr

            model.compile(loss='mean_squared_error',
                          optimizer=opt,
                          metrics=['mae', r2_krs])
            return model

        # Keras callbacks
        # checkpointer = ModelCheckpoint(str(out_nn_model/'autosave.model.h5'), verbose=0,
        #                                save_weights_only=False, save_best_only=True)
        # csv_logger = CSVLogger(out_nn_model/'training.log')
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.75, patience=20, verbose=1, mode='auto',
                                              min_delta=0.0001, cooldown=3, min_lr=0.000000001)
        early_stop = EarlyStopping(monitor='val_loss', patience=100, verbose=1, mode='auto')

        # Callbacks list
        # callback_list = [checkpointer, csv_logger, early_stop, reduce_lr]
        callback_list = [early_stop, reduce_lr]
        # callback_list = callback_list + [clr]

        # Define model
        opt_name = 'sgd'
        init_kwargs = {'input_dim': xtr.shape[1], 'dr_rate': dr_rate, 'opt_name': opt_name}
        fit_kwargs = {'batch_size': batch_size, 'epochs': epochs, 'verbose': verbose}
        fit_kwargs['callbacks'] = callback_list
        # fit_kwargs['validation_split'] = 0.1
        fit_kwargs['validation_data'] = (xvl, yvl)
        model = nn_reg_tidy(**init_kwargs)

        # Train model
        t0 = time()
        history = model.fit(xtr, ytr, **fit_kwargs)
        print('Train time: {:.2f} mins'.format( (time()-t0)/60 ))

        # Predict
        pred_ytr = model.predict(xtr)
        pred_yvl = model.predict(xvl)

        # Calc scores
        print('\nScores with Keras NN:')
        nn_scores = OrderedDict()
        nn_scores['r2_tr'] = r2_score(ytr, pred_ytr)
        nn_scores['r2_vl'] = r2_score(yvl, pred_yvl)
        nn_scores['mae_tr'] = mean_absolute_error(ytr, pred_ytr)
        nn_scores['mae_vl'] = mean_absolute_error(yvl, pred_yvl)
        lg.logger.info(f'\n{pformat(nn_scores)}')
        # for k, v, in nn_scores.items(): print(f'{k}: {v}')
        utils.dump_dict(nn_scores, outpath=run_outdir/'./krs_scores.txt')


    print('Done.')
            
            
def main(args):
    args = parse_args(args)


    args = vars(args)
    ret = run(args)
    

if __name__ == '__main__':
    """ __name__ == '__main__' explained: www.youtube.com/watch?v=sugvnHA7ElY """
    main(sys.argv[1:])
