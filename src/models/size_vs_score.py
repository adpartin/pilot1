"""
Util functions used for the analysis of how score changes with training set size.
"""
import os
import numpy as np
import pandas as pd

import matplotlib
# matplotlib.use('TkAgg')
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_learning_curve(rslt, score_metric='score', title=None, path=None):
    """ 
    Args:
        rslt : output from sklearn.model_selection.learning_curve()
    """
    tr_sizes  = rslt[0]
    tr_scores = rslt[1]
    te_scores = rslt[2]
    
    fig = plt.figure()
    tr_scores_mean = np.mean(tr_scores, axis=1)
    tr_scores_std  = np.std(tr_scores, axis=1)
    te_scores_mean = np.mean(te_scores, axis=1)
    te_scores_std  = np.std(te_scores, axis=1)

    plt.plot(tr_sizes, tr_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(tr_sizes, te_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.fill_between(tr_sizes,
                        tr_scores_mean - tr_scores_std,
                        tr_scores_mean + tr_scores_std,
                        alpha=0.1, color="r")
    plt.fill_between(tr_sizes,
                        te_scores_mean - te_scores_std,
                        te_scores_mean + te_scores_std,
                        alpha=0.1, color="g")
    
    if title is not None:
        plt.title(title)
    else:
        plt.title('Training set size vs score')
    plt.xlabel('Training set size')
    plt.ylabel(score_metric)
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')