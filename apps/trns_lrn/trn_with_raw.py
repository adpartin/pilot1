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

import matplotlib
matplotlib.use('Agg')
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
    # Args
    parser = argparse.ArgumentParser(description="PyTorch with Raw dfs (from Yitan).")
    # parser.add_argument('--frm', default='krs', type=str, choices=['krs', 'trch'], help='DL framework (default: keras).')
    # parser.add_argument('--dname', default='top6', type=str, choices=['top6', 'ytn'], help='Dataset name (default: top6).')
    parser.add_argument('--src', default='GDSC', type=str, help='Data source (default: GDSC).')
    parser.add_argument('--fold', default=0, type=int, help='Fold (default: 0).')
    
    parser.add_argument('-cf', '--ccl_fea', nargs='+', default=['geneGE'], choices=['geneGE'],
                        help='Cell line features (default: `geneGE`).')
    parser.add_argument('-df', '--drg_fea', nargs='+', default=['DD'], choices=['DD'],
                        help='Drug features (default: `DD`).')
    
    parser.add_argument('--ep', default=250, type=int, help='Epochs (default: 250).')
    parser.add_argument('--dr_rate', default=0.2, type=float, help='Dropout rate (default: 0.2).')
    parser.add_argument('--batch_size', default=32, type=float, help='Batch size (default: 32).')

    parser.add_argument('--n_jobs', default=4, type=int, help='Number of cpu workers (default: 4).')
    
    args = parser.parse_args(args)
    args = vars(args)
    return args
    
    
def create_outdir(outdir, args):
    t = datetime.now()
    t = [t.year, '-', t.month, '-', t.day, '_', 'h', t.hour, '-', 'm', t.minute]
    t = ''.join([str(i) for i in t])
    
    l = [args['src']] + args['ccl_fea'] + args['drg_fea'] + ['EP'+str(args['ep'])] + \
        ['DR'+str(args['dr_rate'])] + [('fold'+str(args['fold']))]
        
    # if 'nn' in args['model_name']: l = [args['opt']] + l
                
    name_sffx = '.'.join( l )
    outdir = Path(outdir) / (name_sffx + '_' + t)
    os.makedirs(outdir)
    return outdir


def r2_torch(y_true, y_pred):
    """ TODO: consider to convert tensors"""
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
            for i, (_, _, _, _, auc, ccl_fea, drg_fea) in enumerate(dl):
                y = auc.to(device) # dtype=torch.float32 fixed an error
                ccl_fea = ccl_fea.to(device)
                drg_fea = drg_fea.to(device)
                x_dct = {'ccl_fea': ccl_fea, 'drg_fea': drg_fea}                

                # Process batch
                if ph == 'train':
                    loss, pred = proc_batch(x_dct=x_dct, y=y, model=model, loss_fnc=loss_fnc, opt=opt)
                else:
                    loss, pred = proc_batch(x_dct=x_dct, y=y, model=model, loss_fnc=loss_fnc, opt=None)

                # Compute metrics (running avg)
                scores[loss_name] += loss.item()
                scores = update_scores_reg(pred=pred, true=y, scores=scores)
                
            # Log scores
            for m in scores.keys():
                logs[m].append(scores[m]/len(dl))

            del y, ccl_fea, drg_fea, x_dct, loss, pred, scores

        if verbose:
            print(f'Epoch {ep+1}/{epochs}; ',
                  f'{int(time()-ep_t0)}s; ',
                  [f'{k}: {v[-1]:.3f}' for k, v in logs.items()])

        # TODO: log scores into file

    return logs


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


class Dataset_Raw(Dataset):
    def __init__(self,
                 res_df: pd.DataFrame,
                 ccl_df: pd.DataFrame,
                 drg_df: pd.DataFrame,
                 ccl_folds_dir: str,
                 src: str,
                 fold: int,
                 tr_ph: str,
                 ccl_fea_list: list=None,
                 drg_fea_list: list=None,
                 fea_sep: str='_',
                 fea_float_dtype: type=torch.float32,
                 # target_float_dtype: type=torch.float64,
                 # drg_dsc_preproc: str=None,  # TODO
                 # ccl_rna_preproc: str=None,  # TODO
                 verbose: bool=True):
        """ 
        Define pytorch dataset for drug response prediction.
        Args:
            res_df (pd.DataFrame) : drug response df
            ccl_df (pd.DataFrame) : cell feature df
            drg_df (pd.DataFrame) : drug feature df
            ccl_folds_dir (str) : folder path that contains cv patitions in text files
            src (str) : source name
            fold (int) : fold index
            tr_ph (str) : training phase ('tr', 'vl', 'te')
            ccl_fea_list (list) : list of prefixes of cell features to retain
            drg_fea_list (list) : list of prefixes of drug features to retain
            fea_sep (str) : separator of feature prefix that indicates type and feature name
            fea_float_dtype (type) : float precision for features
            drg_dsc_preproc (str) : TODO: not implemented
            cell_rna_preproc (str) : TODO: not implemented
            verbose : bool=True
        """
        # ============================================
        # Initialize
        # ============================================
        self.ccl_folds_dir = ccl_folds_dir
        self.src = src
        self.fold = fold
        self.tr_ph = tr_ph.lower()
        self.ccl_fea_list = ccl_fea_list
        self.drg_fea_list = drg_fea_list
        self.fea_sep = fea_sep
        self.fea_float_dtype = fea_float_dtype

        # ============================================
        # Get the ccl names
        # ============================================
        if self.tr_ph in ['tr', 'train', 'training']:
            self.ids_fname = 'TrainList.txt'
        elif self.tr_ph in ['vl', 'val', 'validation']:
            self.ids_fname = 'ValList.txt'
        elif self.tr_ph in ['te', 'test', 'testing']:
            self.ids_fname = 'TestList.txt'
        else:
            raise ValueError('Must specify valid `tr_ph` argument.')
            
        self.ids_path = self.ccl_folds_dir/f'{self.src}/cv_{self.fold}'/self.ids_fname # 'TestList.txt'        
        self.ids_list = pd.read_csv(self.ids_path, header=None).squeeze().values
        
        # ============================================
        # Load dfs
        # ============================================
        res_df = res_df[ res_df['src'].isin([src]) ]  # extract responses of specific source
        self.res_df = res_df[ res_df['ccl_name'].isin( self.ids_list ) ]  # extract responses of specific ccl samples
        
        # Extract specific types of features
        self.ccl_df = ccl_df if self.ccl_fea_list is None else self.extract_subset_fea(ccl_df, fea_list=self.ccl_fea_list, fea_sep=self.fea_sep)
        self.drg_df = drg_df if self.drg_fea_list is None else self.extract_subset_fea(drg_df, fea_list=self.drg_fea_list, fea_sep=self.fea_sep)
        
        # ============================================
        # Public attributes
        # ============================================
        self.cells = self.res_df['ccl_name'].unique().tolist()    # unique cells
        self.drugs = self.res_df['ctrpDrugID'].unique().tolist()  # unique drugs
        self.num_records = len(self.res_df)
        self.ccl_dim = self.ccl_df.shape[1]
        self.drg_dim = self.drg_df.shape[1]
        
        self.ccl_fea_cnt = self.cnt_fea(self.ccl_df, fea_sep=self.fea_sep, verbose=False)
        self.drg_fea_cnt = self.cnt_fea(self.drg_df, fea_sep=self.fea_sep, verbose=False)
        
        # ============================================
        # Convert dfs to arrays and dict for faster access
        # ============================================
        # self.res_arr = self.res_df.values
        # self.ccl_fea_dct = {idx: row.values for idx, row in self.ccl_df.iterrows()}
        # self.drg_fea_dct = {idx: row.values for idx, row in self.drg_df.iterrows()}
        # TODO: does the values must be pytorch tensors??
        self.res_arr = self.res_df.values
        self.ccl_fea_dct = {idx: torch.tensor(row.values, dtype=fea_float_dtype) for idx, row in self.ccl_df.iterrows()}
        self.drg_fea_dct = {idx: torch.tensor(row.values, dtype=fea_float_dtype) for idx, row in self.drg_df.iterrows()}

        # ============================================
        # Summary
        # ============================================
        if verbose:
            print('=' * 80)
            print(f'Data source: {self.src}')
            print(f'Phase: {tr_ph}')
            print(f'Data points: {self.num_records}')
            print(f'Unique cells: {len(self.cells)}')
            print(f'Unique drugs: {len(self.drugs)}')
            print(f'ccl_df.shape: {self.ccl_df.shape}')
            print(f'drg_df.shape: {self.drg_df.shape}')
            print(f'Cell features: {self.ccl_fea_cnt}')
            print(f'Drug features: {self.drg_fea_cnt}')
            

    def __len__(self):
        return len(self.res_arr)

    
    def __getitem__(self, index):
        """ 
        Ref: github.com/xduan7/UnoPytorch/blob/master/utils/datasets/drug_resp_dataset.py        
        res indices: [idx, src, ccl_name, ctrpDrugID, auc, groupID]
        """
        res = self.res_arr[index]
        
        idx = res[0]
        src = res[1]
        ccl_id = res[2]
        drg_id = res[3]
        auc = res[4]
        grp_id = res[5]
        
        ccl_fea = self.ccl_fea_dct[ccl_id]
        drg_fea = self.drg_fea_dct[drg_id]
        
        # Cast values
        # ccl_fea = ccl_fea.astype(np.float32)
        # drg_fea = drg_fea.astype(np.float32)
        
        # return ccl_fea, drg_fea, auc
        return idx, src, ccl_id, drg_id, auc, ccl_fea, drg_fea
    
    
    def extract_subset_fea(self, df, fea_list, fea_sep='_'):
        """ Extract features based feature prefix name. """
        fea = [c for c in df.columns if (c.split(fea_sep)[0]) in fea_list]
        df = df[fea]
        return df    
    
    
    def cnt_fea(self, df, fea_sep='_', verbose=True):
        """ Count the number of features per feature type. """
        dct = {}
        unq_prfx = df.columns.map(lambda x: x.split(fea_sep)[0]).unique() # unique feature prefixes
        for prfx in unq_prfx:
            fea_type_cols = [c for c in df.columns if (c.split(fea_sep)[0]) in prfx] # all fea names of specific type
            dct[prfx] = len(fea_type_cols)
        if verbose: print(dct)
        return dct
    
    
class NN_Reg_Raw(nn.Module):
    def __init__(self,
                 ccl_dim: int,
                 drg_dim: int,
                 dr_rate: float=0.2):
        super().__init__()
        self.ccl_dim = ccl_dim
        self.drg_dim = drg_dim
        self.dr_rate = dr_rate

#         self.fc1 = nn.Linear(self.ccl_dim + self.drg_dim, 1000)
#         self.bn1 = nn.BatchNorm1d(1000)
        
#         self.fc2 = nn.Linear(1000, 1000)
#         self.bn2 = nn.BatchNorm1d(1000)
        
#         self.fc3 = nn.Linear(1000, 500)
#         self.bn3 = nn.BatchNorm1d(500)
        
#         self.fc4 = nn.Linear(500, 250)
#         self.bn4 = nn.BatchNorm1d(250)
        
#         self.fc5 = nn.Linear(250, 125)
#         self.bn5 = nn.BatchNorm1d(125)
        
#         self.fc6 = nn.Linear(125, 60)
#         self.bn6 = nn.BatchNorm1d(60)

#         self.fc7 = nn.Linear(60, 30)
#         self.bn7 = nn.BatchNorm1d(30)
        
#         self.fc8 = nn.Linear(30, 1)
        
        self.fc1 = nn.Linear(self.ccl_dim + self.drg_dim, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 250)
        self.fc4 = nn.Linear(250, 60)
        self.fc5 = nn.Linear(60, 1)
        self.dropout = nn.Dropout(self.dr_rate)  
        

    def forward(self, ccl_fea, drg_fea):
        x = torch.cat((ccl_fea, drg_fea), dim=1)
        
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = self.dropout(F.relu(self.fc4(x)))
        x = F.relu(self.fc5(x))
        
#         # Expanded
#         x = self.fc1(x)
#         x = self.bn1(x)
#         x = F.relu(x)
        
#         x = self.fc2(x)
#         x = self.bn2(x)
#         x = F.relu(x)
#         x = F.dropout(x, p=self.dr_rate, training=self.training)

#         x = self.fc3(x)
#         x = self.bn3(x)
#         x = F.relu(x)
#         x = F.dropout(x, p=self.dr_rate, training=self.training)

#         x = self.fc4(x)
#         x = self.bn4(x)
#         x = F.relu(x)
#         x = F.dropout(x, p=self.dr_rate, training=self.training)
        
#         x = self.fc5(x)
#         x = self.bn5(x)
#         x = F.relu(x)
#         x = F.dropout(x, p=self.dr_rate, training=self.training)

#         x = self.fc6(x)
#         x = self.bn6(x)
#         x = F.relu(x)
#         x = F.dropout(x, p=self.dr_rate, training=self.training)

#         x = self.fc7(x)
#         x = self.bn7(x)
#         x = F.relu(x)
#         x = F.dropout(x, p=self.dr_rate, training=self.training)

#         x = self.fc8(x)
#         x = F.relu(x)
        return x 


def run(args):
    # Args
    # framework = args['frm'] # 'pytorch'
    # data_name = args['dname'] # 'top6'
    src = args['src']
    epochs = args['ep']
    fold = args['fold']
    ccl_fea_list = args['ccl_fea'] # ['geneGE']
    drg_fea_list = args['drg_fea'] # ['DD']
    dr_rate = args['dr_rate']
    batch_size = args['batch_size']
    n_jobs = args['n_jobs']
    
    fea_sep = '_'
    verbose = True
    # fea_float_dtype = fea_float_dtype
    
    
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
    # Load data
    # =====================================================
    datadir = Path(file_path/'../../data/yitan/Data')
    ccl_folds_dir = Path(file_path/'../../data/yitan/CCL_10Fold_Partition')
    pdm_folds_dir = Path(file_path/'../../data/yitan/PDM_10Fold_Partition')
    fea_data_name = 'CCL_PDM_TransferLearningData_rmFactor_0.0_ddNorm_std.pkl'

    # Un-pickle files
    import _pickle as cp
    pkl_file = open(datadir/fea_data_name, 'rb')
    res = cp.load(pkl_file)
    ccl = cp.load(pkl_file)
    drg = cp.load(pkl_file)
    pkl_file.close()
    
    lg.logger.info('\n{}'.format('=' * 50))
    lg.logger.info('res: {}'.format(res.shape))
    lg.logger.info('ccl: {}'.format(ccl.shape))
    lg.logger.info('drg: {}'.format(drg.shape))
    
    tmp = res.groupby('SOURCE').agg({'ccl_name': 'nunique', 'ctrpDrugID': 'nunique'}).reset_index()
    lg.logger.info(tmp)
    
    # Update dfs
    res = res.reset_index()
    res = res.rename(columns={'index': 'idx', 'SOURCE': 'src', 'area_under_curve': 'auc'})


    # =====================================================
    # PyTorch
    # =====================================================
    lg.logger.info('\n{}'.format('=' * 50))
    lg.logger.info('Train with NN Reg ...')
    lg.logger.info('PyTorch version: {}'.format(torch.__version__))
    lg.logger.info('\nCUDA info ...')
    lg.logger.info('is_available:   {}'.format( torch.cuda.is_available()) )
    lg.logger.info('device_name:    {}'.format( torch.cuda.get_device_name(0)) )
    lg.logger.info('device_count:   {}'.format( torch.cuda.device_count()) )
    lg.logger.info('current_device: {}'.format( torch.cuda.current_device()) )

    # Create torch datasets
    ds_kwargs = {
        'res_df': res,
        'ccl_df': ccl,
        'drg_df': drg,
        'ccl_folds_dir': ccl_folds_dir,
        'src': src,
        'ccl_fea_list': ccl_fea_list,
        'drg_fea_list': drg_fea_list,
        'fea_sep': fea_sep}

    tr_ds = Dataset_Raw(tr_ph='tr', fold=fold, **ds_kwargs)
    vl_ds = Dataset_Raw(tr_ph='vl', fold=fold, **ds_kwargs)
    te_ds = Dataset_Raw(tr_ph='te', fold=fold, **ds_kwargs)    
    
    # Create data loaders
    tr_loader_kwargs = {'batch_size': batch_size, 'shuffle': True,  'num_workers': n_jobs}
    vl_loader_kwargs = {'batch_size': 4*batch_size, 'shuffle': False, 'num_workers': n_jobs}
    te_loader_kwargs = {'batch_size': 4*batch_size, 'shuffle': False, 'num_workers': n_jobs}

    tr_loader = DataLoader(tr_ds, **tr_loader_kwargs)
    vl_loader = DataLoader(vl_ds, **vl_loader_kwargs)
    te_loader = DataLoader(te_ds, **te_loader_kwargs)    
    
    
    # Choose device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Create model and move to CUDA device
    model = NN_Reg_Raw(
        ccl_dim = tr_ds.ccl_dim,
        drg_dim = tr_ds.drg_dim,
        dr_rate = dr_rate).to(device)
    # model.apply(weight_init_linear)
    print(model)

    # Query device where the model is located
    # print(get_model_device(model))
    # print('current_device:', torch.cuda.current_device()) # why current device is 0??      

    # Loss function and optimizer
    loss_fnc = nn.MSELoss(reduction='mean')
    opt = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, nesterov=False)  # pytorch.org/docs/stable/optim.html
    # lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer=opt, base_lr=1e-5, max_lr=1e-3, mode='triangular')

    # Define training params
    metrics = ['mean_abs_err', 'r2']
    fit_kwargs = {'epochs': epochs, 'device': device, 'verbose': verbose, 'metrics': metrics}

    # Train
    t0 = time()
    logs = fit(model=model,
               loss_fnc=loss_fnc,
               opt=opt,
               tr_dl=tr_loader,
               vl_dl=vl_loader,
               **fit_kwargs)
    lg.logger.info('Train: {:.1f} hrs'.format( (time()-t0)/360) )

    # Predict
    xtr = torch.tensor(xtr.values, dtype=torch.float32).to(device)
    xvl = torch.tensor(xvl.values, dtype=torch.float32).to(device)
    pred_ytr = model(xtr).cpu().detach().numpy()
    pred_yvl = model(xvl).cpu().detach().numpy()

    # Calc scores
    print('\nScores with PyTorch NN:')
    nn_scores = OrderedDict()
    nn_scores['r2_tr'] = r2_score(ytr, pred_ytr)
    nn_scores['r2_vl'] = r2_score(yvl, pred_yvl)
    nn_scores['mae_tr'] = mean_absolute_error(ytr, pred_ytr)
    nn_scores['mae_vl'] = mean_absolute_error(yvl, pred_yvl)
    for k, v, in nn_scores.items(): print(f'{k}: {v}')

    # Dump results
    nn_scores = pd.DataFrame(nn_scores)
    nn_scores.to_csv( run_outdir/('nn_scores.csv'), index=False) 
    lg.logger.info(f'\nnn_scores\n{nn_scores}')
    
    lg.kill_logger()
            
            
def main(args):
    args = parse_args(args)
    ret = run(args)
    

if __name__ == '__main__':
    """ __name__ == '__main__' explained: www.youtube.com/watch?v=sugvnHA7ElY """
    main(sys.argv[1:])
