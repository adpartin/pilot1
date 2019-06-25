""" This is the main script.
Refs:
github.com/stared/livelossplot/blob/master/examples/pytorch.ipynb
pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
pytorch.org/tutorials/beginner/nn_tutorial.html
www.datascience.com/blog/transfer-learning-in-pytorch-part-one
"""
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

# Pytorch
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
print("PyTorch Version: ", torch.__version__)

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
    
    if ('nn' in args['model_name']) and (args['attn'] is True): 
        name_sffx = '.'.join( [src] + [args['model_name']] + ['attn'] + \
                             [args['opt']] + [args['cv_method']] + [('cvf'+str(args['cv_folds']))] + args['cell_features'] + \
                             args['drug_features'] + [args['target_name']] )
            
    elif ('nn' in args['model_name']) and (args['attn'] is False): 
        name_sffx = '.'.join( [src] + [args['model_name']] + ['fc'] + \
                             [args['opt']] + [args['cv_method']] + [('cvf'+str(args['cv_folds']))] + args['cell_features'] + \
                             args['drug_features'] + [args['target_name']] )
        
    else:
        name_sffx = '.'.join( [src] + [args['model_name']] + \
                             [args['opt']] + [args['cv_method']] + [('cvf'+str(args['cv_folds']))] + args['cell_features'] + \
                             args['drug_features'] + [args['target_name']] )

    outdir = Path(outdir) / (name_sffx + '_' + t)
    os.makedirs(outdir)
    return outdir


def get_model_device(model):
    return str(next(model.parameters()).device)


def np_to_tensor(a, dtype=torch.float32):
    return torch.tensor(a, dtype=dtype)


def r2_torch(y_true, y_pred):
    epsilon = 1e-7  # this epsilon value used in TF
    SS_res = torch.sum( (y_true - y_pred)**2 )
    SS_tot = torch.sum( (y_true - torch.mean(y_true))**2 )
    r2 = 1 - SS_res / (SS_tot + epsilon)
    return r2


def proc_batch(xb, yb, model, loss_fnc, opt=None):
    pred = model(xb)
    loss = loss_fnc(pred, yb)

    # Backward pass
    if opt is not None:
        # opt.zero_grad()
        loss.backward()
        opt.step()

    return loss, pred


def calc_metrics(pred, yb, scores, val=False, metrics=None):
    pred, yb = pred.numpy(), yb.numpy()
    prfx = 'val_' if val is True else ''

    for m in metrics:
        if m in ['mae', 'mean_absolute_error']:
            scores[prfx + 'mean_abs_err'] = sklearn.metrics.mean_absolute_error(yb, pred)

        elif m in ['r2', 'r2_score']:
            scores[prfx + 'r2'] = sklearn.metrics.r2_score(yb, pred)

        elif m in ['median_absolute_error']:
            scores[prfx + 'median_abs_err'] = sklearn.metrics.median_absolute_error(yb, pred)

        elif m in ['mean_squared_error']:
            scores[prfx + 'mean_squared_error'] = sklearn.metrics.mean_squared_error(yb, pred)

    return scores


def log_metrics(logs):
    prefix = 'val_' if ph=='val' else ''
    logs[prefix + 'loss'] = bt_loss
    logs[prefix + 'mae'] = bt_mae
    logs[prefix + 'r2'] = bt_r2
    return logs

            
class Top6DataReg(Dataset):
    # discuss.pytorch.org/t/data-processing-as-a-batch-way/14154
    # github.com/utkuozbulak/pytorch-custom-dataset-examples#incorporating-pandas
    # nbviewer.jupyter.org/github/FraPochetti/KagglePlaygrounds/blob/master/NYC%20Taxi%20Fares%20Prediction.ipynb
    def __init__(self, xdata, ydata):
        # self.x = xdata.values
        # self.y = ydata.values
        self.x = xdata
        self.y = ydata
        self.y = self.y.view(-1, 1)
                
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.x[idx, :]
        y = self.y[idx]
        return x, y


class TORCH_REGRESSOR(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 250)
        self.fc4 = nn.Linear(250, 60)
        self.fc5 = nn.Linear(60, 1)
        self.dropout = nn.Dropout(0.2)
                                            
    def forward(self, x):
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = self.dropout(F.relu(self.fc4(x)))
        x = F.relu(self.fc5(x))
        return x

    def fit(self):
        pass
        return None
    
    def predict(self):
        pass
        return None

    
def fit(model: nn.Module,
        loss_fnc, 
        opt: torch.optim,
        tr_dl: torch.utils.data.DataLoader,
        vl_dl: torch.utils.data.DataLoader=None,
        epochs: int=1,
        device: torch.device='cuda:0',
        verbose: bool=True,
        metrics=None) -> dict:
    """ ... """
    print(f'\ndevice: {device}')
    model.to(device)  
    
    with torch.cuda.device(device):
        print('current_device:', torch.cuda.current_device())    
    
        # Similar to keras `history`
        logs = OrderedDict()

        for ep in range(epochs):
            ep_t0 = time()
            phases = ['train', 'val'] if vl_dl is not None else ['train']

            for ph in phases:
                if ph == 'train':
                    model.train()
                    dl = tr_dl
                elif ph == 'val':
                    model.eval()
                    dl = vl_dl

                bt_loss = 0
                bt_mae = 0
                bt_r2 = 0

                for xb, yb in dl:
                    xb = xb.to(device)
                    yb = yb.to(device)

                    # Zero parameter gradients
                    opt.zero_grad()

                    with torch.set_grad_enabled(ph=='train'):
                        input_opt = opt if ph=='train' else None
                        loss, pred = proc_batch(xb, yb, model, loss_fnc, opt=input_opt)

                        # Compute metrics
                        # logs = calc_metrics(yb, pred, logs, phase=False, metrics=metrics)
                        bt_loss += loss.item() # item() returns a number from a tensor that contains a single value
                        bt_mae += torch.mean(torch.abs(pred-yb)).item()
                        bt_r2 += r2_torch(y_true=yb, y_pred=pred).item()

                        # Gives error: RuntimeError: Can't call numpy() on Variable that requires grad. Use var.detach().numpy() instead.
                        # pred, yb = pred.numpy(), yb.numpy()
                        # bt_mae += np.mean(np.abs(pred-yb))
                        # bt_r2 += r2_score(y_true=yb, y_pred=pred)

                bt_loss /= len(dl)
                bt_mae /= len(dl)
                bt_r2 /= len(dl)          

                # Log metrics
                # logs = log_metrics(scores, ph)
                prefix = 'val_' if ph=='val' else ''
                logs[prefix + 'loss'] = bt_loss
                logs[prefix + 'mae'] = bt_mae
                logs[prefix + 'r2'] = bt_r2

            if verbose:
#                 print(f'Epoch {ep+1}/{epochs}; ',
#                       f'{int(time()-ep_t0)}s; ',
#                       [f'{k}: {v:.3f}' for k, v in logs.items()])
                l = [f'{k}: {v:.3f}' for k, v in logs.items()]
                print(f'Epoch {ep+1}/{epochs}; ',
                      f'{int(time()-ep_t0)}s; ',
                      *l)                
    return logs

    
def run(args):
    t0 = time()
    
    cell_fea = ['rna']
    drug_fea = ['dsc']
    other_fea = []
    epochs = 300
    dname = 'combined'
    tr_sources = ['gdsc']
    target_name = 'AUC' 
    scaler = 'stnd'
    mltype = 'reg'
    batch_size = 32

    # Feature list
    fea_list = cell_fea + drug_fea + other_fea

    # Flag for feature extracting. When False, we finetune the whole model,
    #   when True we only update the reshaped layer params
    feature_extract = True

    # ========================================================================
    #       Load data and pre-proc
    # ========================================================================
    if dname == 'combined':
        DATADIR = file_path / '../../data/processed/from_combined/tidy_drop_fibro'
        DATAFILENAME = 'tidy_data.parquet'
        datapath = DATADIR / DATAFILENAME
    
        dataset = load_tidy_combined(
                datapath, fea_list=fea_list, random_state=SEED) # logger=lg.logger

        print(f'\n{tr_sources} ...')
        tr_data = get_data_by_src(
                dataset, src_names=tr_sources) # logger=lg.logger

        xdata, ydata, meta, tr_scaler = break_src_data(
                tr_data, target=target_name,
                scaler=scaler) # logger=lg.logger
            
        del tr_data

    elif dname == 'top6':
        DATADIR = file_path / '../../data/raw/'
        DATAFILENAME = 'uniq.top6.reg.parquet'
        datapath = DATADIR / DATAFILENAME
        
        df = pd.read_parquet(datapath, engine='auto', columns=None)
        df = df.sample(frac=1.0, axis=0, random_state=SEED).reset_index(drop=True)

        scaler = args['scaler']
        if  scaler is not None:
            if scaler == 'stnd':
                scaler = StandardScaler()
            elif scaler == 'minmax':
                scaler = MinMaxScaler()
            elif scaler == 'rbst':
                scaler = RobustScaler()

        xdata = df.iloc[:, 1:]
        ydata = df.iloc[:, 0]
        xdata = scaler.fit_transform(xdata).astype(np.float32)
        
        src = 'top6'
        dfs = {src: (ydata, xdata)}

        del df, xdata, ydata

    # ========================================================================
    #       Define CV split
    # ========================================================================
    #cv = cv_splitter(cv_method=cv_method, cv_folds=1, test_size=0.2,
    #                 mltype=mltype, shuffle=True, random_state=SEED)
    #if cv_method=='simple':
    #    groups = None
    #elif cv_method=='group':
    #    groups = tr_data['CELL'].copy()
        
    data = pd.concat([ydata, xdata], axis=1)
    df_tr, df_te = train_test_split(data, test_size=0.2)
    df_tr = df_tr.reset_index(drop=True)
    df_te = df_te.reset_index(drop=True)
    print(df_tr.shape)
    print(df_te.shape)

    ytr, xtr = df_tr.iloc[:,0], df_tr.iloc[:,1:]
    yte, xte = df_te.iloc[:,0], df_te.iloc[:,1:]

    # Scale
    col_names = xtr.columns
    scaler = StandardScaler()
    xtr = pd.DataFrame( scaler.fit_transform(xtr) ).astype(np.float32)
    xte = pd.DataFrame( scaler.transform(xte) ).astype(np.float32)
    xtr.columns = col_names
    xte.columns = col_names

    xtr = np_to_tensor(xtr.values)
    ytr = np_to_tensor(ytr.values)
    xte = np_to_tensor(xte.values)
    yte = np_to_tensor(yte.values)
    # print(type(xtr))
    # print(xtr.dtype)

    tr_ds = Top6DataReg(xdata=xtr, ydata=ytr)
    te_ds = Top6DataReg(xdata=xte, ydata=yte)

    # Define data loaders
    num_workers = 1
    tr_loader_prms = {'batch_size': batch_size, 'shuffle': True, 'num_workers': num_workers}
    te_loader_prms = {'batch_size': 4*batch_size, 'shuffle': False, 'num_workers': num_workers}
    tr_loader = DataLoader(tr_ds, **tr_loader_prms)
    te_loader = DataLoader(te_ds, **te_loader_prms)

    # pytorch.org/docs/stable/cuda.html
    # towardsdatascience.com/speed-up-your-algorithms-part-1-pytorch-56d8a4ae7051
    print('is_available:  ', torch.cuda.is_available())
    print('device_name:   ', torch.cuda.get_device_name(0))
    print('device_count:  ', torch.cuda.device_count())
    print('current_device:', torch.cuda.current_device())

    # Define network
    device = torch.device('cuda:3')
    model = TORCH_REGRESSOR(input_dim=tr_ds.x.shape[1]).to(device=device) # send model to gpu/cpu device
    # print(get_model_device(model))
    # print('current_device:', torch.cuda.current_device()) # why current device is 0??


    if device.type == 'cuda':
        print(get_model_device(model))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3, 1), 'GB')
        

    # Define loss function and optimizer
    loss_fnc = nn.MSELoss(reduction='mean')
    opt = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)  # pytorch.org/docs/stable/optim.html

    # ----------------------------------------------------------------------------
    # Train
    # ----------------------------------------------------------------------------
    logs = fit(tr_dl = tr_loader,
               vl_dl = te_loader,
               model = model,
               loss_fnc = loss_fnc,
               opt = opt,
               epochs = epochs,
               device = device,
               metrics=None)     
    
#     tr_loss_list = []
#     tr_mae_list = []
#     tr_r2_list = []

#     te_loss_list = []
#     te_mae_list = []
#     te_r2_list = []

#     # logs = OrderedDict()

#     # Choose cuda device with context manager
#     with torch.cuda.device(device):
#         print('\ncurrent_device:', torch.cuda.current_device())

#         for ep in range(epochs):
#             ep_t0 = time()

#             # Training loop
#             model.train() # turns-on dropout for training
#             tr_loss, tr_mae, tr_r2 = 0, 0, 0

#             for xb, yb in tr_loader:
#                 xb = xb.to(device) # move data to gpu/cpu device
#                 yb = yb.to(device) # move data to gpu/cpu device

#                 # Feedforward
#                 pred = model(xb)
#                 loss = loss_fnc(pred, yb)

#                 # Backprop and optimization
#                 opt.zero_grad()
#                 loss.backward()   # compute loss gradients wrt to model parameters and inputs
#                 opt.step()  # update model parameters;  pytorch.org/docs/stable/optim.html

#                 # Compute metrics
#                 tr_loss += loss.item() # item() returns a number from a tensor that contains a single value
#                 tr_mae += torch.mean(torch.abs(pred-yb))
#                 tr_r2 += r2_torch(y_true=yb, y_pred=pred)

#             tr_loss /= len(tr_loader)
#             tr_mae /= len(tr_loader)
#             tr_r2 /= len(tr_loader)

#             tr_loss_list.append(tr_loss)
#             tr_mae_list.append(tr_mae)
#             tr_r2_list.append(tr_r2)

#             del xb, yb


#             # Validation loop
#             model.eval()  # turn-off dropout in inferenece
#             # with torch.set_grad_enabled(False)  # check this alternative ??!!
#             with torch.no_grad():
#                 te_loss, te_mae, te_r2 = 0, 0, 0

#                 for xb, yb in te_loader:
#                     xb = xb.to(device)
#                     yb = yb.to(device)

#                     # Feedforward
#                     pred = model(xb)
#                     loss = loss_fnc(pred, yb)

#                     # Compute metrics
#                     te_loss += loss.item() # item() returns a number from a tensor that contains a single value
#                     te_mae += torch.mean(torch.abs(pred-yb))
#                     te_r2 += r2_torch(y_true=yb, y_pred=pred)

#                 te_loss /= len(te_loader)
#                 te_mae /= len(te_loader)
#                 te_r2 /= len(te_loader)

#             te_loss_list.append(te_loss)
#             te_mae_list.append(te_mae)
#             te_r2_list.append(te_r2)

#             del xb, yb

#             print(f'Epoch {ep+1}/{epochs}; ',
#                   f'{int(time()-ep_t0)}s; '
#                   f'tr_loss: {tr_loss:.3f}; ',
#                   f'vl_loss: {te_loss:.3f}; ',
#                   f'tr_mae: {tr_mae:.3f}; ',
#                   f'vl_mae: {te_mae:.3f}; ',
#                   f'tr_r2: {tr_r2:.3f}; ',
#                   f'vl_r2: {te_r2:.3f}; ')


    # print(model.f5.in_features)
    # print(model.f5.out_features)

#     fig, ax = plt.subplots((1,1))
#     ax.plot(

    print('Done.')



def main(args):
    #config_fname = file_path / CONFIGFILENAME
    #args = argparser.get_args(args=args, config_fname=config_fname)
    ## pprint(vars(args))
    #args = vars(args)
    #if args['outdir'] is None:
    #    args['outdir'] = OUTDIR
    
    args = None
    ret = run(args)


if __name__ == '__main__':
    main(sys.argv[1:])
    
