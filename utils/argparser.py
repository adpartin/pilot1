# Parsing priority: command-line > config-file > defualt params
import argparse
import configparser
from collections import OrderedDict


# Default (arg, value) pairs for argparse object
dflt_args = {
    'target_name': 'AUC',
    'target_transform': False,
    'train_sources': ['ccle'],
    'test_sources': ['ccle'],
    'row_sample': None,
    'col_sample': None,
    'tissue_type': None,
    'cell_features': ['rna'], 
    'drug_features': ['dsc'],
    'other_features': [] ,
    # 'mltype': 'reg',
    'model_name': 'lgb_reg',
    'cv_method': 'simple',
    'cv_folds': 1,
    'retrain': False,
    'n_jobs': 4,
    'outdir': None,

    'dname': 'combined',

    # Learning curve
    'lc_ticks': 5,

    # Neural network
    'epochs': 500,
    'batch_size': 32,
    'dr_rate': 0.2,
    'attn': False,
    'scaler': 'stnd',
    'opt': 'sgd',
}


def get_args(args, config_fname=None):
    """ Main function that extracts arguments from command-line and config file.
    Args:
        args : args from command line.
        config_fname : config file name that contains (arg, value) pair. This will override the dflt_args.
    """
    parser = get_cli_args(args)  # get command line args

    if config_fname is not None:
        config_params = read_config_file(file=config_fname)  # get config file args
        dflt_args_new = override_dflt_with_config(
            dflt_args=dflt_args,
            config_params=config_params)
        parser.set_defaults(**dflt_args_new)
    else:
        parser.set_defaults(**dflt_args)

    #parser.set_defaults(**dflt_args_new)
    args = parser.parse_args(args)
    #args = parser.parse_known_args(args)
    return args


def get_cli_args(args=None):
    """ Extracts args with argparse. """
    # Initialize parser
    parser = argparse.ArgumentParser(description="Cell-drug sensitivity parser.")

    # Select target to predict
    parser.add_argument('-t', '--target_name',
        # default="AUC",
        choices=['AUC', 'AUC1', 'IC50'],
        help='Column name of the target variable.') # target_name = 'AUC1'
    parser.add_argument('-tt', '--target_transform',
        type=str2bool,
        # action='store_true', default=False, help='Default: False'
        help='True: transform target; Flase: do not transform target')

    # Select train and test (inference) sources
    parser.add_argument('-tr', '--train_sources', nargs='+',
        # default=["ccle"],
        choices=['ccle', 'gcsi', 'gdsc', 'ctrp', 'nci60'],
        help='Data sources to use for training.')
    parser.add_argument('-te', '--test_sources', nargs='+',
        # default=["ccle"],
        choices=['ccle', 'gcsi', 'gdsc', 'ctrp', 'nci60'],
        help='Data sources to use for testing.')

    # Keep a subset of row/cols
    parser.add_argument('--row_sample',
        help='Sample a subset of rows (float in the range (0.0, 1.0], or int for the exact num of rows to keep).')
    parser.add_argument('--col_sample',
        help='Sample a subset of cols (float in the range (0.0, 1.0], or int for the exact num of rows to keep).')

    # Select tissue types
    parser.add_argument('-ts', '--tissue_type',
        # default=argparse.SUPPRESS,
        choices=[],
        help='Tissue type to use.')

    # Select feature types
    parser.add_argument('-cf', '--cell_features', nargs='+',
        # default=['rna'],
        choices=['rna', 'cnv', 'clb'],
        help='Cell line features.') # ['rna', cnv', 'rna_latent']
    parser.add_argument('-df', '--drug_features', nargs='+',
        # default=['dsc'],
        choices=['dsc', 'fng', 'dlb'],
        help='Drug features.') # ['dsc', 'fng', 'dsc_latent', 'fng_latent']
    parser.add_argument('-of', '--other_features',
        # default=[],
        choices=[],
        help='Other feature types (derived from cell lines and drugs). E.g.: cancer type, etc).') # ['cell_labels', 'drug_labels', 'ctype', 'csite', 'rna_clusters']

    # Select ML models
    # parser.add_argument('--mltype',
    #     # default=["lgb_reg"],
    #     choices=['reg', 'cls'],
    #     help='Type to ML problem: `reg` or `cls`.')

    # Select ML models
    parser.add_argument('-ml', '--model_name',
        # default=["lgb_reg"],
        choices=['lgb_reg', 'rf_reg', 'nn_reg', 'nn_model0', 'nn_model1', 'nn_model2', 'nn_model3', 'nn_model4'],
        help='ML model to use for training.')

    # Select CV scheme
    parser.add_argument('-cvm', '--cv_method',
        # default="simple",
        choices=['simple', 'group'],
        help='Cross-val split method.')
    parser.add_argument('-cvf', '--cv_folds',
        # default=5,
        type=int,
        help='Number cross-val folds.')

    # Re-train model
    parser.add_argument('--retrain',
        # type=str2bool,
        action='store_true',
        default=False,
        help='Re-train final model using the entire dataset (Default: False).')        

    # Learning curve
    parser.add_argument('--lc_ticks',
        # default=5,
        type=int,
        help='Number of ticks in the learning curve plot.')

    # Take care of utliers
    # parser.add_argument("--outlier", default=False)

    # Define n_jobs
    parser.add_argument('--n_jobs',
        # default=4,
        type=int)

    # Select outdir
    parser.add_argument('--outdir',
        # default=["lgb_reg"],
        type=str,
        help='Output dir.')    

    # Select data name
    parser.add_argument('--dname',
        type=str, choices=['combined', 'top6'],
        help='Data name (combined (default) or top6).')

    # Select NN hyper_params
    parser.add_argument('-ep', '--epochs',
        # default=100,
        type=int,
        help='Number of epochs to train the neural network.')
    parser.add_argument('-b', '--batch_size',
        # default=64,
        type=int,
        help='Batch size for the neural network.')
    parser.add_argument('--dr_rate',
        # default=0.2,
        type=float,
        help='Dropout ratefor the neural network.')
    parser.add_argument('--attn',
        # type=str2bool,
        action='store_true',
        default=False,
        help='Use attention layer (Default: False).')
    parser.add_argument('-sc', '--scaler',
        # default=0.2,
        type=str,
        choices=['stnd', 'minmax', 'rbst'])
    parser.add_argument('--opt',
        # default=["lgb_reg"],
        type=str,
        choices=['sgd', 'adam', 'clr'],
        help='Optimizer name.')    

    return parser


def read_config_file(file):
    """ Read config file params.
    https://github.com/ECP-CANDLE/Benchmarks/blob/release_01/common/default_utils.py
    """
    config = configparser.ConfigParser()
    config.read(file)
    fileparams = {}
    
    for sec in config.sections():
        for param, value in config.items(sec):
            # if arg appear more than once in the file, use first one
            if not param in fileparams:
                fileparams[param] = eval(value)

    # print('\nConfig file params:')
    # print(fileparams)
    return fileparams


def override_dflt_with_config(dflt_args, config_params):
    """ Override default argparse arguments with parameters specified in config file.
    Args:
        dflt_args : dict with default arguments for argparse object
        config_params : dict with params from config file
    """
    for p, v in config_params.items():
        if p in dflt_args.keys():
            dflt_args[p] = v
        else:
            # TODO
            pass
    return dflt_args


# def args_overwrite_config(args, config_params):
#     """ Overwrite configuration parameters with parameters specified via command-line.    
#     Args:
#         args : ArgumentParser object (Parameters specified via command-line)
#         config_params : python dictionary (Parameters read from configuration file)
#     """
#     params = config_params
#     args_dict = vars(args)
#     for arg in args_dict.keys():
#         params[arg] = args_dict[arg]
#     return params


def str2bool(v):
    if v.lower() in ('yes', 'y', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'n', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


