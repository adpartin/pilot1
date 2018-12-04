import os
import logging
import numpy as np
import pandas as pd

import matplotlib
# matplotlib.use('TkAgg')
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import seaborn as sns
from scipy import stats
from sklearn.metrics import r2_score, mean_absolute_error, median_absolute_error, explained_variance_score

DATADIR = '/Users/apartin/work/jdacs/Benchmarks/Data/Pilot1'


def setup_logger(logfilename='logfile.log'):
    """ Create logger. Output to file and console. """
    # Create file handler
    # "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    # "[%(asctime)s %(process)d] %(message)s"
    # fileFormatter = logging.Formatter("%(asctime)s : %(threadName)-12.12s : %(levelname)-5.5s : %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fileFormatter = logging.Formatter("%(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fileHandler = logging.FileHandler(filename=logfilename)
    fileHandler.setFormatter(fileFormatter)
    fileHandler.setLevel(logging.INFO)

    # Create console handler
    # consoleFormatter = logging.Formatter("%(name)-12s : %(levelname)-8s : %(message)s")
    consoleFormatter = logging.Formatter("%(message)s")
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(consoleFormatter)
    consoleHandler.setLevel(logging.INFO)

    # Create logger and add handlers
    # logger = logging.getLogger(__name__)
    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    logger.addHandler(fileHandler)
    logger.addHandler(consoleHandler)

    # from combo (when use candle)
    # for log in [logger, uno_data.logger]:
    #     log.setLevel(logging.DEBUG)
    #     log.addHandler(fh)
    #     log.addHandler(sh)
    return logger


def boxplot_rsp_per_drug(df, target_name, path='boxplot_rsp_per_drug.png'):
    """ Boxplot of response per drug. """
    # https://seaborn.pydata.org/generated/seaborn.catplot.html
    # https://seaborn.pydata.org/generated/seaborn.boxplot.html
    # https://matplotlib.org/api/_as_gen/matplotlib.pyplot.boxplot.html
    # fig, ax = plt.subplots()
    # bp = data.boxplot(column=target_name, by='DRUG', rot=70, fontsize=10, sym='k.', return_type='both') # how to control alpha of fliers?
    g = sns.catplot(data=df, kind='box', x='DRUG', y=target_name, showfliers=True, sym='r.') # 'sym' doesn't affect
    g.set_xticklabels(rotation=70)
    # ax = sns.swarmplot(data=data, x='DRUG', y='AUC', color='0.25') # takes too long
    # ax = sns.catplot(data=data, kind='box', x='DRUG', y='AUC', hue='ctype') # takes too long
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')


def plot_hist_drugs(x, name='drugs_hist', ordered=False, path='hist_drugs.png'):
    """ Plot counts per drug. """
    # x = data['DRUG']
    val_counts = x.value_counts().reset_index().rename(columns={'index': 'drug', 'DRUG': 'count'})
    if ordered:
        val_counts.sort_values(by='count', inplace=True)
    else:
        val_counts.sort_values(by='drug', inplace=True)
    fig, ax = plt.subplots()
    plt.barh(val_counts['drug'], val_counts['count'], color='b', align='center', alpha=0.7)
    # g = sns.barplot(x=val_counts['count'], y=val_counts['drug'], palette='viridis')
    # # g = sns.catplot(y='DRUG', data=data, kind='count', palette='viridis', alpha=0.9)
    # # g = sns.catplot(y='DRUG', data=data, kind='count', order=data['DRUG'].value_counts().index, palette='viridis', alpha=0.9)
    plt.xlabel('count')
    plt.ylabel('drug label')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')


def plot_hist(x, var_name, bins=100, path='hist.png'):
    """ Plot hist of a 1-D array x. """
    (mu, sigma) = stats.norm.fit(x)
    fig, ax = plt.subplots()
    sns.distplot(x, bins=bins, kde=True, fit=stats.norm, 
                 hist_kws={'linewidth': 2, 'alpha': 0.6, 'color': 'b'},
                 kde_kws={'linewidth': 2, 'alpha': 0.6, 'color': 'k'},
                 fit_kws={'linewidth': 2, 'alpha': 0.6, 'color': 'r', 'label': f'norm fit: $\mu$={mu:.2f}, $\sigma$={sigma:.2f}'})
    plt.grid(True)
    plt.legend()
    plt.title(var_name + ' hist')
    plt.savefig(path, bbox_inches='tight')


def plot_qq(x, var_name, path='qq_plot.png'):
    """ Q-Q plot (for response variables). """
    # https://en.wikipedia.org/wiki/Q%E2%80%93Q_plot
    # In Q-Q plot the axes are transformed in order to make a normal distribution appear in a straight line.
    # (a perfectly normal distribution would exactly follow a line with slope = 1 and intercept = 0).
    # The theoretical quantiles are placed along the x-axis. That is, the x-axis is not our data, it's simply
    # an expectation of where our data should have been if it were normal.
    # The actual data is plotted along the y-axis.
    fig, ax = plt.subplots()
    res = stats.probplot(x, dist='norm', fit=True, plot=plt)
    plt.title('Q-Q plot of ' + var_name)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')


def print_scores(model, xdata, ydata, logger=None):
    preds = model.predict(xdata)
    model_r2_score = r2_score(ydata, preds)
    model_mean_abs_error = mean_absolute_error(ydata, preds)
    model_median_abs_error = median_absolute_error(ydata, preds)
    model_explained_variance_score = explained_variance_score(ydata, preds)
    if logger is not None:
        logger.info(f'r2_score: {model_r2_score:.2f}')
        logger.info(f'mean_absolute_error: {model_mean_abs_error:.2f}')
        logger.info(f'median_absolute_error: {model_median_abs_error:.2f}')
        logger.info(f'model_explained_variance_score: {model_explained_variance_score:.2f}')
    else:
        print(f'r2_score: {model_r2_score:.2f}')
        print(f'mean_absolute_error: {model_mean_abs_error:.2f}')
        print(f'median_absolute_error: {model_median_abs_error:.2f}')
        print(f'model_explained_variance_score: {model_explained_variance_score:.2f}')


def dump_preds(model, df_data, xdata, target_name, path, model_name=None):
    """
    Args:
        model : ml model (must have predict() method)
        df : df that contains the cell and drug names, and target value
        xdata : features to make predictions
        target_name : name of the target as it appears in the df (e.g. 'AUC')
    """
    # preds = pd.DataFrame({target_name+'_pred': model.predict(xdata)})
    df1 = df_data[['CELL', 'DRUG', 'csite', 'ctype', 'simplified_csite', 'simplified_ctype', target_name]].copy()

    preds = model.predict(xdata)
    abs_error = abs(df_data[target_name] - preds)
    squared_error = (df_data[target_name] - preds)**2
    df2 = pd.DataFrame({target_name+'_pred': model.predict(xdata),
                        target_name+'_error': abs_error,
                        target_name+'_sq_error': squared_error})

    df_preds = pd.concat([df1, df2], axis=1).reset_index(drop=True)
    df_preds.to_csv(path)


class HyperTunedModel():  # ModelTunedCVSearch
    # https://www.kaggle.com/spektrum/randomsearchcv-to-hyper-tune-1st-level
    pass


# ---------------------------------------------------------------------------------
def plot_boxplot(x, y, figsize=(15, 5), title=None, outpath=None):
    """
    Args:
        y : 1-D array for which to plot the boxplot
        x : 1-D array that contains "groups". Boxplot of y is computed for each group in y.
        figsize : 
        title : (default: None)
        outpath : path to save the figure (default: doesn't save if path not provided)
    
    # explanation for outliers in boxplot: https://en.wikipedia.org/wiki/Interquartile_range
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(x=x, y=y, showfliers=True)
    # sns.swarmplot(x=x, y=y, color='0.25')  # swarmplot takes to long to plot!
    if title:
        ax.set_title(title)
    ax.grid(True)
    if outpath:
        plt.savefig(fname=outpath)
        
        
def plot_density(col, split_by, df, kind='kde', bins=100, figsize=(15, 5), title=None, outpath=None): 
    """
    Args:
        col : the column name in df to plot (e.g. 'GROWTH')
        split_by : by which column to split the plots (e.g., 'SOURCE' --> plot growth for each source)
        df : load_drug_fingerprints
        kind : 
        bins : number of bins in histogram (used when king='hist')
        figsize : 
        title : (default: None)
        outpath : path to save the figure (default: doesn't save if path not provided)
    """
    fig, ax = plt.subplots(figsize=figsize)
    for i, b in enumerate(df[split_by].unique()):
        tmp = df[df[split_by]==b].copy()
        if kind=='kde':
            sns.kdeplot(tmp[col], shade=True, label=b, legend=True)
        elif kind=='hist':
            ax.hist(tmp[col], bins=bins, label=b)
    if title:
        ax.set_title(title)
    ax.grid(True)
    if outpath:
        plt.savefig(fname=outpath) 
        

# ---------------------------------------------------------------------------------
# def impute_and_scale(df, scaling='std', imputing='mean', dropna='all'):
#     """Impute missing values with mean and scale data included in pandas dataframe.
#     Parameters
#     ----------
#     df : pandas dataframe
#         dataframe to impute and scale
#     scaling : 'maxabs' [-1,1], 'minmax' [0,1], 'std', or None, optional (default 'std')
#         type of scaling to apply
#     """
#     from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler, Imputer
    
#     if dropna:
#         df = df.dropna(axis=1, how=dropna)
#     else:
#         empty_cols = df.columns[df.notnull().sum() == 0]
#         df[empty_cols] = 0

#     if imputing is None or imputing.lower() == 'none':
#         mat = df.values
#     else:
#         imputer = Imputer(strategy=imputing, axis=0)
#         mat = imputer.fit_transform(df)

#     if scaling is None or scaling.lower() == 'none':
#         return pd.DataFrame(mat, columns=df.columns)

#     if scaling == 'maxabs':
#         scaler = MaxAbsScaler()
#     elif scaling == 'minmax':
#         scaler = MinMaxScaler()
#     else:
#         scaler = StandardScaler()

#     mat = scaler.fit_transform(mat)
#     df = pd.DataFrame(mat, columns=df.columns)

#     return df


# def load_drug_info():
#     path = os.path.join(DATADIR, 'drug_info')
#     df = pd.read_table(path, dtype=object)
#     df['PUBCHEM'] = 'PubChem.CID.' + df['PUBCHEM']
#     return df


# def load_drug_set_fingerprints(drug_set='Combined_PubChem', ncols=None, usecols=None,
#                                scaling=None, imputing=None, add_prefix=False):
#     fps = ['PFP', 'ECFP']
#     usecols_all = usecols
#     df_merged = None
#     for fp in fps:
#         path = os.path.join(DATADIR, '{}_dragon7_{}.tsv'.format(drug_set, fp))  # path = get_file(DATA_URL + '{}_dragon7_{}.tsv'.format(drug_set, fp))
#         df_cols = pd.read_table(path, engine='c', nrows=0, skiprows=1, header=None)
#         total = df_cols.shape[1] - 1
#         if usecols_all is not None:
#             usecols = [x.replace(fp+'.', '') for x in usecols_all]
#             usecols = [int(x) for x in usecols if x.isdigit()]
#             usecols = [x for x in usecols if x in df_cols.columns]
#             if usecols[0] != 0:
#                 usecols = [0] + usecols
#             df_cols = df_cols.loc[:, usecols]
#         elif ncols and ncols < total:
#             usecols = np.random.choice(total, size=ncols, replace=False)
#             usecols = np.append([0], np.add(sorted(usecols), 1))
#             df_cols = df_cols.iloc[:, usecols]

#         dtype_dict = dict((x, np.float32) for x in df_cols.columns[1:])
#         df = pd.read_table(path, engine='c', skiprows=1, header=None,
#                            usecols=usecols, dtype=dtype_dict)
#         df.columns = ['{}.{}'.format(fp, x) for x in df.columns]

#         col1 = '{}.0'.format(fp)
#         df1 = pd.DataFrame(df.loc[:, col1])
#         df1.rename(columns={col1: 'Drug'}, inplace=True)

#         df2 = df.drop(col1, 1)
#         if add_prefix:
#             df2 = df2.add_prefix('dragon7.')

#         df2 = impute_and_scale(df2, scaling, imputing, dropna=None)

#         df = pd.concat([df1, df2], axis=1)

#         df_merged = df if df_merged is None else df_merged.merge(df)

#     return df_merged


# def load_drug_fingerprints(ncols=None, scaling='std', imputing='mean', dropna=None, add_prefix=True):
#     df_info = load_drug_info()
#     df_info['Drug'] = df_info['PUBCHEM']

#     df_fp = load_drug_set_fingerprints(drug_set='Combined_PubChem', ncols=ncols)
#     df_fp = pd.merge(df_info[['ID', 'Drug']], df_fp, on='Drug').drop('Drug', 1).rename(columns={'ID': 'Drug'})

#     df_fp2 = load_drug_set_fingerprints(drug_set='NCI60', usecols=df_fp.columns.tolist() if ncols else None)

#     df_fp = pd.concat([df_fp, df_fp2]).reset_index(drop=True)
#     df1 = pd.DataFrame(df_fp.loc[:, 'Drug'])
#     df2 = df_fp.drop('Drug', 1)
#     df2 = impute_and_scale(df2, scaling=None, imputing=imputing, dropna=dropna)
#     if add_prefix:
#         df2 = df2.add_prefix('dragon7.')
#     df_fp = pd.concat([df1, df2], axis=1)

#     # logger.info('Loaded combined dragon7 drug fingerprints: %s', df_fp.shape)

#     return df_fp


# def summarize_response_data(df):
#     df_sum = df.groupby('Source').agg({'Growth': 'count', 'Sample': 'nunique',
#                                        'Drug1': 'nunique', 'Drug2': 'nunique'})
#     df_sum['MedianDose'] = df.groupby('Source').agg({'Dose1': 'median'})
#     return df_sum


# def assign_partition_groups(df, partition_by='drug_pair'):
#     if partition_by == 'cell':
#         group = df['Sample']
#     elif partition_by == 'drug_pair':
#         df_info = load_drug_info()
#         id_dict = df_info[['ID', 'PUBCHEM']].drop_duplicates(['ID']).set_index('ID').iloc[:, 0]
#         group = df['Drug1'].copy()
#         group[(df['Drug2'].notnull()) & (df['Drug1'] <= df['Drug2'])] = df['Drug1'] + ',' + df['Drug2']
#         group[(df['Drug2'].notnull()) & (df['Drug1'] > df['Drug2'])] = df['Drug2'] + ',' + df['Drug1']
#         group2 = group.map(id_dict)
#         mapped = group2.notnull()
#         group[mapped] = group2[mapped]
#     elif partition_by == 'index':
#         group = df.reset_index()['index']
#     print('Grouped response data by {}: {} groups'.format(partition_by, group.nunique()))
#     return group