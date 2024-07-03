#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File     : utils_features_selection.py
@Time     : 2024/07/03 15:11:58
@Author   : RanPeng 
@Version  : pyhton 3.8.6
@Contact  : 21112030023@m.fudan.edu.cn
@License  : (C)Copyright 2022-2023, DingLab-CHINA-SHNAGHAI
@Function : None
'''
# here put the import lib

import pandas as pd
import numpy as np
import os
from os.path import join as pjoin
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report, roc_auc_score, roc_curve, auc
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
import utils
from save_log import save_print_to_file

warnings.filterwarnings('ignore')
sns.set_style("white")

plt.rcParams['font.sans-serif'] = ['Simhei']
plt.rcParams['axes.unicode_minus'] = False



## Cross-Validation Functions

def StratifiedKFold_func_with_features_sel(x, y, num_iter=100, score_type='auc'):
    """
    Stratified K-Fold Cross Validation with feature selection.

    Args:
        x (np.array): Feature matrix.
        y (np.array): Labels.
        num_iter (int, optional): Number of iterations. Defaults to 100.
        score_type (str, optional): Type of score to use ('auc' or 'f1'). Defaults to 'auc'.

    Returns:
        list: Mean and standard deviation of training and validation scores.
    
    Example:
        mean_acc, mean_val, std_acc, std_val = StratifiedKFold_func_with_features_sel(x, y, num_iter=50, score_type='f1')
    """
    acc_v = []
    acc_t = []
    for i in range(num_iter):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=i)
        for tr_idx, te_idx in skf.split(x, y):
            x_tr = x[tr_idx, :]
            y_tr = y[tr_idx]
            x_te = x[te_idx, :]
            y_te = y[te_idx]
            model = xgb.XGBClassifier(max_depth=3, learning_rate=0.2, reg_alpha=1)
            model.fit(x_tr, y_tr)
            pred = model.predict(x_te)
            train_pred = model.predict(x_tr)
            if score_type == 'auc':
                acc_v.append(roc_auc_score(y_te, pred))
                acc_t.append(roc_auc_score(y_tr, train_pred))
            else:
                acc_v.append(f1_score(y_te, pred))
                acc_t.append(f1_score(y_tr, train_pred))
    return [np.mean(acc_t), np.mean(acc_v), np.std(acc_t), np.std(acc_v)]


def StratifiedKFold_func(x, y, num_iter=100, model=None, score_type='auc'):
    """
    Stratified K-Fold Cross Validation.

    Args:
        x (np.array): Feature matrix.
        y (np.array): Labels.
        num_iter (int, optional): Number of iterations. Defaults to 100.
        model (XGBClassifier, optional): Model to use. Defaults to XGBClassifier.
        score_type (str, optional): Type of score to use ('auc' or 'f1'). Defaults to 'auc'.

    Returns:
        list: Mean and standard deviation of training and validation scores.
    
    Example:
        mean_acc, mean_val, std_acc, std_val = StratifiedKFold_func(x, y, num_iter=50, score_type='f1')
    """
    if model is None:
        model = xgb.XGBClassifier(max_depth=4, learning_rate=0.2, reg_alpha=1)
    
    acc_v = []
    acc_t = []
    
    for i in range(num_iter):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=i)
        for tr_idx, te_idx in skf.split(x, y):
            x_tr = x[tr_idx, :]
            y_tr = y[tr_idx]
            x_te = x[te_idx, :]
            y_te = y[te_idx]

            model.fit(x_tr, y_tr)
            pred = model.predict(x_te)
            train_pred = model.predict(x_tr)

            pred_proba = model.predict_proba(x_te)[:, 1]
            train_pred_proba = model.predict_proba(x_tr)[:, 1]

            if score_type == 'auc':
                acc_v.append(roc_auc_score(y_te, pred_proba))
                acc_t.append(roc_auc_score(y_tr, train_pred_proba))
            else:
                acc_v.append(f1_score(y_te, pred))
                acc_t.append(f1_score(y_tr, train_pred))          

    return [np.mean(acc_t), np.mean(acc_v), np.std(acc_t), np.std(acc_v)]


## Plot Functions

def show_confusion_matrix(validations, predictions):
    """
    Plot confusion matrix.

    Args:
        validations (list): True labels.
        predictions (list): Predicted labels.
    
    Example:
        show_confusion_matrix(y_test, y_pred)
    """
    LABELS = ['Survival', 'Death']
    matrix = metrics.confusion_matrix(validations, predictions)
    plt.figure(figsize=(4.5, 3))
    sns.heatmap(matrix, cmap='coolwarm', linecolor='white', linewidths=1, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt='d')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def plot_roc(labels, predict_prob, model_name, fig, labels_name, k):
    """
    Plot ROC curve.

    Args:
        labels (list): True labels.
        predict_prob (list): Predicted probabilities.
        model_name (str): Name of the model.
        fig (plt.Figure): Matplotlib figure object.
        labels_name (list): List to store labels for the legend.
        k (int): Index for line style.

    Returns:
        list: Updated labels_name.
    
    Example:
        plot_roc(y_test, y_pred_proba, 'XGBoost', fig, labels_name, 0)
    """
    false_positive_rate, true_positive_rate, thresholds = roc_curve(labels, predict_prob)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    line_list = ['--', '-']
    ax = fig.gca()
    plt.title('ROC', fontsize=20)
    ax.plot(false_positive_rate, true_positive_rate, line_list[k % 2], linewidth=1 + (1 - k / 5), label=model_name + ' AUC = %0.4f' % roc_auc)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylabel('TPR', fontsize=20)
    plt.xlabel('FPR', fontsize=20)
    labels_name.append(model_name + ' AUC = %0.4f' % roc_auc)
    return labels_name

def ceate_feature_map(features):
    """
    Create feature map file for XGBoost.

    Args:
        features (list): List of feature names.
    
    Example:
        ceate_feature_map(['Feature1', 'Feature2', 'Feature3'])
    """
    with open('xgb.fmap', 'w') as outfile:
        for i, feat in enumerate(features):
            outfile.write('{0}\t{1}\tq\n'.format(i, feat))

def plot_decision_boundary(model, x_tr, y_tr):
    """
    Plot decision boundary and sample points.

    Args:
        model (XGBClassifier): Trained XGBoost model.
        x_tr (np.array): Training feature matrix.
        y_tr (np.array): Training labels.
    
    Example:
        plot_decision_boundary(trained_model, X_train, y_train)
    """
    coord1_min = x_tr[:, 0].min() - 1
    coord1_max = x_tr[:, 0].max() + 1
    coord2_min = x_tr[:, 1].min() - 1
    coord2_max = x_tr[:, 1].max() + 1

    coord1, coord2 = np.meshgrid(
        np.linspace(coord1_min, coord1_max, int((coord1_max - coord1_min) * 30)).reshape(-1, 1),
        np.linspace(coord2_min, coord2_max, int((coord2_max - coord2_min) * 30)).reshape(-1, 1),
    )
    coord = np.c_[coord1.ravel(), coord2.ravel()]

    category = model.predict(coord).reshape(coord1.shape)

    dir_save = './decision_boundary'
    os.makedirs(dir_save, exist_ok=True)

    plt.close('all')
    plt.figure(figsize=(7, 7))
    custom_cmap = ListedColormap(['#EF9A9A', '#90CAF9'])
    plt.contourf(coord1, coord2, category, cmap=custom_cmap)
    plt.savefig(pjoin(dir_save, 'decision_boundary1.png'), bbox_inches='tight')
    plt.scatter(x_tr[y_tr == 0, 0], x_tr[y_tr == 0, 1], c='yellow', label='Survival', s=30, alpha=1, edgecolor='k')
    plt.scatter(x_tr[y_tr == 1, 0], x_tr[y_tr == 1, 1], c='palegreen', label='Death', s=30, alpha=1, edgecolor='k')
    plt.ylabel('Lymphocytes (%)')
    plt.xlabel('Lactate dehydrogenase')
    plt.legend()
    plt.show()

def plot_3D_fig(X_data):
    """
    Plot 3D figure of data points.

    Args:
        X_data (pd.DataFrame): Data with features and labels.
    
    Example:
        plot_3D_fig(data)
    """
    cols = ['Lactate Dehydrogenase', 'Lymphocyte (%)', 'High-sensitivity C-reactive Protein']
    X_data = X_data.dropna(subset=cols, how='all')
    col = 'Type2'
    data_df_sel2_0 = X_data[X_data[col] == 0]
    data_df_sel2_1 = X_data[X_data[col] == 1]
    
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111, projection='3d')
    i, j, k = 2, 0, 1
    ax.scatter(data_df_sel2_0[cols[i]], data_df_sel2_0[cols[j]], data_df_sel2_0[cols[k]], c=data_df_sel2_0[col], cmap='Blues_r', label='Survival', linewidth=0.5)
    ax.scatter(data_df_sel2_1[cols[i]], data_df_sel2_1[cols[j]], data_df_sel2_1[cols[k]], c=data_df_sel2_1[col], cmap='gist_rainbow_r', label='Death', marker='x', linewidth=0.5)

    cols_en = ['Lactate dehydrogenase', 'Lymphocyte(%)', 'High-sensitivity C-reactive protein', 'Type of Survival(0) or Death(1)']
    ax.set_zlabel(cols_en[k])
    ax.set_ylabel(cols_en[j])
    ax.set_xlabel(cols_en[i])
    fig.legend(['Survival', 'Death'], loc='upper center')
    plt.show()
