import os
import logging
import datetime
import numpy as np
import pandas as pd
from collections import OrderedDict

import matplotlib
# matplotlib.use('TkAgg')
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import seaborn as sns
from scipy import stats

import sklearn
from sklearn.metrics import r2_score, mean_absolute_error, median_absolute_error, explained_variance_score

import classlogger

DATADIR = '/Users/apartin/work/jdacs/Benchmarks/Data/Pilot1'


def create_outdir(outdir='./', args=None):
    """ Create output dir. """
    t = datetime.datetime.now()
    t = [t.year, '-', t.month, '-', t.day, '_', 'h', t.hour, '-', 'm', t.minute]
    t = ''.join([str(i) for i in t])
    if args is not None:
        name_sffx = '.'.join(args['train_sources'] + [args['model_name']] + [args['cv_method']] + args['cell_features'] + args['drug_features'] + [args['target_name']])
    else:
        name_sffx = 'out'
    run_outdir = os.path.join(outdir, name_sffx + '~' + t)
    os.makedirs(run_outdir)
    return run_outdir


def dump_args(args, outdir='./'):
    """ Dump args (dict) into file.
    Examples:
        utils.dump_args(args, outdir=outdir)
    """
    with open(os.path.join(outdir, 'args.txt'), 'w') as file:
        for k, v in args.items():
            file.write('{} = {}\n'.format(k, v))


def subsample(df, v, axis=0):
    """ Extract a random subset of rows or cols from df. """
    assert v > 0, f'sample must be >0; got {v}'
    if v <= 1.0:
        df = df.sample(frac=v, axis=axis).reset_index(drop=True)
    else:
        df = df.sample(n=v, axis=axis).reset_index(drop=True)
    return df


def reg_auroc(y_true, y_pred):
    """ Compute area under the ROC for regression. TODO: check this func. """
    y_true = np.where(y_true < 0.5, 1, 0)
    y_score = np.where(y_pred < 0.5, 1, 0)
    auroc = sklearn.metrics.roc_auc_score(y_true, y_score)
    return auroc


def calc_preds(estimator, xdata, ydata, mltype):
    """ Calc predictions. """
    if mltype == 'cls':    
        if ydata.ndim > 1 and ydata.shape[1] > 1:
            y_preds = estimator.predict_proba(xdata)
            y_preds = np.argmax(y_preds, axis=1)
            y_true = np.argmax(ydata, axis=1)
        else:
            y_preds = estimator.predict_proba(xdata)
            y_preds = np.argmax(y_preds, axis=1)
            y_true = ydata
            
    elif mltype == 'reg':
        y_preds = estimator.predict(xdata)
        y_true = ydata

    return y_preds, y_true


def calc_scores(y_true, y_preds, mltype, metrics=None):
    """ Create dict of scores.
    Args:
        metrics : TODO allow to pass a string of metrics
    """
    scores = OrderedDict()

    if mltype == 'cls':    
        scores['auroc'] = sklearn.metrics.roc_auc_score(y_true, y_preds)
        scores['f1_score'] = sklearn.metrics.f1_score(y_true, y_preds, average='micro')
        scores['acc_blnc'] = sklearn.metrics.balanced_accuracy_score(y_true, y_preds)

    elif mltype == 'reg':
        scores['r2'] = sklearn.metrics.r2_score(y_true=y_true, y_pred=y_preds)
        scores['mean_absolute_error'] = sklearn.metrics.mean_absolute_error(y_true=y_true, y_pred=y_preds)
        scores['median_absolute_error'] = sklearn.metrics.median_absolute_error(y_true=y_true, y_pred=y_preds)
        scores['mean_squared_error'] = sklearn.metrics.mean_squared_error(y_true=y_true, y_pred=y_preds)
        scores['auroc_reg'] = reg_auroc(y_true=y_true, y_pred=y_preds)

    # score_names = ['r2', 'mean_absolute_error', 'median_absolute_error', 'mean_squared_error']

    # # https://scikit-learn.org/stable/modules/model_evaluation.html
    # for metric_name, metric in metrics.items():
    #     if isinstance(metric, str):
    #         scorer = sklearn.metrics.get_scorer(metric_name) # get a scorer from string
    #         scores[metric_name] = scorer(ydata, preds)
    #     else:
    #         scores[metric_name] = scorer(ydata, preds)

    return scores


def update_cross_validate_scores(cv_scores):
    """ Takes dict of scores from sklean's cross_validate and converts to df
    with certain updates. """
    # TODO: move this func to cvrun.py (rename cvrun.py utils_cv.py)
    cv_folds = len(list(cv_scores.values())[0])

    df = cv_scores_to_df(cv_scores, decimals=3, calc_stats=False)

    # Add `metric` col
    v = list(map(lambda x: '_'.join(x.split('_')[1:]), df.index))
    df.insert(loc=0, column='metric', value=v)

    # Convert `neg` metric to positive and update metric names (drop `neg_`)
    # scikit-learn.org/stable/modules/model_evaluation.html --> explains the `neg` in `neg_mean_absolute_error`
    idx_bool = [True if 'neg_' in s else False for s in df['metric']]
    for i, bl in enumerate(idx_bool):
        if bl:
            df.iloc[i, -cv_folds:] = abs(df.iloc[i, -cv_folds:])
    df['metric'] = df['metric'].map(lambda s: s.split('neg_')[-1] if 'neg_' in s else s)

    # Add `tr_set` col
    v = list(map(lambda x: True if 'train' in x else False, df.index))
    df.insert(loc=1, column='tr_set', value=v)
    return df


def cv_scores_to_df(cv_scores, decimals=3, calc_stats=False):
    """ Takes dict of scores from sklean's cross_validate and converts to df.
    Args:
        scores : that's the output from sklearn.model_selection.cross_validate()
    """
    # Drop certain keys that come from cross_validate()
    for k in ['fit_time', 'train_time', 'score_time']:
        if k in cv_scores.keys():
            del cv_scores[k]  # cv_scores.pop(k, None)

    cv_scores = pd.DataFrame(cv_scores).T
    cv_scores.columns = ['f'+str(c) for c in cv_scores.columns]
    if calc_stats:
        cv_scores.insert(loc=0, column='mean', value=cv_scores.mean(axis=1))
        cv_scores.insert(loc=1, column='std', value=cv_scores.std(axis=1))
    cv_scores = cv_scores.round(decimals=decimals)
    return cv_scores


# def adj_r2_score(ydata, preds, x_size):
#     """ Calc adjusted r^2.
#     https://en.wikipedia.org/wiki/Coefficient_of_determination#Adjusted_R2
#     https://dziganto.github.io/data%20science/linear%20regression/machine%20learning/python/Linear-Regression-101-Metrics/
#     https://stats.stackexchange.com/questions/334004/can-r2-be-greater-than-1
#     """
#     r2 = r2_score(ydata, preds)
#     adj_r2 = 1 - (1 - r2) * (x_size[0] - 1)/(x_size[0] - x_size[1] - 1)
#     return adj_r2


# def calc_scores(model, xdata, ydata):
#     """ Create dict of scores. """
#     # TODO: replace `if` with `try`
#     preds = model.predict(xdata)
#     scores = OrderedDict()
#     scores['r2_score'] = sklearn.metrics.r2_score(ydata, preds)
#     scores['adj_r2_score'] = adj_r2_score(ydata, preds, x_size=xdata.shape)
#     scores['mean_abs_error'] = sklearn.metrics.mean_absolute_error(ydata, preds)
#     scores['median_abs_error'] = sklearn.metrics.median_absolute_error(ydata, preds)
#     # scores['explained_variance_score'] = sklearn.metrics.explained_variance_score(ydata, preds)
    
#     scores['r2'] = sklearn.metrics.r2_score(ydata, preds)
#     #scores['adj_r2_score'] = self.__adj_r2_score(ydata, preds)
#     scores['mean_absolute_error'] = sklearn.metrics.mean_absolute_error(ydata, preds)
#     scores['median_absolute_error'] = sklearn.metrics.median_absolute_error(ydata, preds)
#     scores['mean_squared_error'] = sklearn.metrics.mean_squared_error(ydata, preds)
#     return scores

    
def print_scores(model, xdata, ydata, logger=None):
    preds = model.predict(xdata)
    model_r2_score = r2_score(ydata, preds)
    model_mean_abs_error = mean_absolute_error(ydata, preds)
    model_median_abs_error = median_absolute_error(ydata, preds)
    if logger is not None:
        logger.info(f'r2_score: {model_r2_score:.2f}')
        logger.info(f'mean_abs_error: {model_mean_abs_error:.2f}')
        logger.info(f'median_abs_error: {model_median_abs_error:.2f}')


def dump_preds(model, df_data, xdata, target_name, path, model_name=None):
    """
    Args:
        model : ml model (must have predict() method)
        df : df that contains the cell and drug names, and target value
        xdata : features to make predictions
        target_name : name of the target as it appears in the df (e.g. 'AUC')
    """
    combined_cols = ['CELL', 'DRUG', 'csite', 'ctype', 'simplified_csite', 'simplified_ctype', target_name]
    ccle_org_cols = ['CELL', 'DRUG', 'tissuetype', target_name]

    ##df1 = df_data[['CELL', 'DRUG', 'csite', 'ctype', 'simplified_csite', 'simplified_ctype', target_name]].copy()
    if set(combined_cols).issubset(set(df_data.columns.tolist())):
        df1 = df_data[combined_cols].copy()
    elif set(ccle_org_cols).issubset(set(df_data.columns.tolist())):
        df1 = df_data[ccle_org_cols].copy()
    else:
        df1 = df_data['CELL', 'DRUG'].copy()

    preds = model.predict(xdata)
    abs_error = abs(df_data[target_name] - preds)
    squared_error = (df_data[target_name] - preds)**2
    df2 = pd.DataFrame({target_name+'_pred': model.predict(xdata),
                        target_name+'_error': abs_error,
                        target_name+'_sq_error': squared_error})

    df_preds = pd.concat([df1, df2], axis=1).reset_index(drop=True)
    df_preds.to_csv(path)



# ==============================================================================
# Plot funcs
# ==============================================================================
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


def plot_rf_fi(rf_model, figsize=(8, 5), plot_direction='h', columns=None, max_cols_plot=None,
               color='g', title=None, errorbars=True):
    """ Plot feature importance from a random forest.
    Args:
        plot_direction : direction of the bars (`v` for vertical, `h` for hrozontal)
        columns : list of columns names (df.columns)
        max_cols_plot (int) : number of top most important features to plot
    Returns:
        indices : all feature indices ordered by importance
        fig : handle for plt figure
    """
    fontsize=14
    alpha=0.7

    importance = rf_model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in rf_model.estimators_], axis=0)
    indices = np.argsort(importance)[::-1]  # feature indices ordered by importance
    top_indices = indices[:max_cols_plot]    # get indices of top most important features
    if columns is None:
        columns = top_indices
    else:
        columns = np.array(columns)[top_indices]

    # Start plotting
    fig, ax = plt.subplots(figsize=figsize)
    if title:
        ax.set_title(title)
        
    if plot_direction=='v':
        if errorbars:
            ax.bar(range(len(top_indices)), importance[top_indices], color=color, align='center', alpha=alpha,
                   yerr=std[top_indices], ecolor='black')
        else:
            ax.bar(range(len(top_indices)), importance[top_indices], color=color, align='center', alpha=alpha)
        ax.set_xticks(range(len(top_indices)))
        ax.set_xticklabels(columns, rotation='vertical', fontsize=fontsize)
        ax.set_xlim([-1, len(top_indices)])
        ax.set_xlabel('Feature', fontsize=fontsize)
        ax.set_ylabel('Importance', fontsize=fontsize)
        [tick.label.set_fontsize(fontsize-4) for tick in ax.yaxis.get_major_ticks()]
    else:
        if errorbars:
            ax.barh(range(len(top_indices)), importance[top_indices], color=color, align='center', alpha=alpha,
                    xerr=std[top_indices], ecolor='black')
        else:
            ax.barh(range(len(top_indices)), importance[top_indices], color=color, align='center', alpha=alpha)
        ax.set_yticks(range(len(top_indices)))
        ax.set_yticklabels(columns, rotation='horizontal', fontsize=fontsize)
        ax.set_ylim([-1, len(top_indices)])
        # ax.invert_yaxis()
        ax.set_ylabel('Feature', fontsize=fontsize)
        ax.set_xlabel('Importance', fontsize=fontsize)
        [tick.label.set_fontsize(fontsize-4) for tick in ax.xaxis.get_major_ticks()]

    # ax.grid()
    # plt.tight_layout()

    return indices, fig
# ==============================================================================



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
        
        
def plot_density(df, col, split_by, kind='kde', bins=100, figsize=(15, 5), title=None, outpath=None): 
    """
    Args:
        df : 
        col : the column name in df to plot (e.g. 'GROWTH')
        split_by : by which column to split the plots (e.g., 'SOURCE' --> plot growth for each source)
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
            ax.legend()
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