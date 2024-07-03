#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File     : utils.py
@Time     : 2024/07/03 15:12:11
@Author   : RanPeng 
@Version  : pyhton 3.8.6
@Contact  : 21112030023@m.fudan.edu.cn
@License  : (C)Copyright 2022-2023, DingLab-CHINA-SHNAGHAI
@Function : None
'''
# here put the import lib

import os
from os.path import join as pjoin
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report

def is_number(s):
    """
    Check if a string is a number.

    Args:
        s (str): Input string.

    Returns:
        bool: True if the string is a number, False otherwise.
    """
    if s is None:
        s = np.nan

    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False

def read(path: str, usecols=None, is_ts='infer'):
    """
    Read processed data from parquet, csv, or excel files.

    Args:
        path (str): File path, must be a parquet, csv, or excel file.
        usecols (list, optional): Columns to select. Defaults to None.
        is_ts (str, optional): Indicates if it is a time series. 'infer', True, False. Defaults to 'infer'.

    Returns:
        pd.DataFrame: Read data as DataFrame.

    Example:
        data = read('data.csv', usecols=['Column1', 'Column2'], is_ts=True)
    """
    # Set index
    if is_ts == 'infer':
        index_col = [0, 1] if os.path.split(path)[1].startswith('time_series') else [0]
    elif is_ts is True:
        index_col = [0, 1]
    elif is_ts is False:
        index_col = [0]
    else:
        raise Exception('Invalid is_ts parameter')

    # Read data
    if path.endswith('.parquet'):
        data = pd.read_parquet(path)
    elif path.endswith('.csv'):
        try:
            data = pd.read_csv(path, index_col=index_col, encoding='gb18030')
        except UnicodeDecodeError:
            data = pd.read_csv(path, index_col=index_col, encoding='utf-8')
        except:
            raise
    elif path.endswith('.xlsx'):
        data = pd.read_excel(path, index_col=index_col)
    else:
        raise Exception('Invalid file type')

    # Extract specified columns
    if usecols is not None:
        data = data[usecols]

    return data

def merge_data_by_sliding_window(data, n_days=1, dropna=True, subset=None, time_form='diff'):
    """
    Merge data using a sliding window.

    Args:
        data (pd.DataFrame): Time series data with PATIENT_ID as the primary index and RE_DATE as the secondary index.
        n_days (int, optional): Window length. Defaults to 1.
        dropna (bool, optional): Drop rows with NaNs after merging. Defaults to True.
        subset (list, optional): Columns to consider for dropna. Defaults to None.
        time_form (str, optional): Time index format to return, 'diff' or 'timestamp'. Defaults to 'diff'.

    Returns:
        pd.DataFrame: Merged data with updated time index.

    Example:
        merged_data = merge_data_by_sliding_window(data, n_days=3, dropna=True, time_form='timestamp')
    """
    data = data.reset_index(level=1)
    t_diff = data['Discharge Time'].dt.normalize() - data['RE_DATE'].dt.normalize()
    data['t_diff'] = t_diff.dt.days.values // n_days * n_days
    data = data.set_index('t_diff', append=True)

    data = (
        data
        .groupby(['PATIENT_ID', 't_diff']).ffill()
        .groupby(['PATIENT_ID', 't_diff']).last()
    )

    if dropna:
        data = data.dropna(subset=subset)

    if time_form == 'timestamp':
        data = (
            data
            .reset_index(level=1, drop=True)
            .set_index('RE_DATE', append=True)
        )
    elif time_form == 'diff':
        data = data.drop(columns=['RE_DATE'])

    return data

def score_form(x: np.array):
    """
    Predict using score table.

    Args:
        x (np.array): Columns order: ['Lactate Dehydrogenase', 'Lymphocyte (%)', 'High-sensitivity C-reactive Protein'].

    Returns:
        tuple: Predicted classes and total scores.

    Example:
        pred, score = score_form(df[['Lactate Dehydrogenase', 'Lymphocyte (%)', 'High-sensitivity C-reactive Protein']].values)
    """
    x = x.copy()

    # Lactate Dehydrogenase
    x[:, 0] = pd.cut(
        x[:, 0],
        [-2, 107, 159, 210, 262, 313, 365, 416, 467, 519, 570, 622, 673, 724, 776, 827, 1e5],
        labels=list(range(-5, 11))
    )

    # Lymphocyte (%)
    x[:, 1] = pd.cut(
        x[:, 1],
        [-2, 1.19, 3.12, 5.05, 6.98, 8.91, 10.84, 12.77, 14.7, 16.62, 18.55, 20.48, 22.41, 24.34, 1e5],
        labels=list(range(8, -6, -1))
    )

    # High-sensitivity C-reactive Protein
    x[:, 2] = pd.cut(
        x[:, 2],
        [-2, 19.85, 41.2, 62.54, 83.88, 1e5],
        labels=list(range(-1, 4))
    )

    # Total score
    total_score = x.sum(axis=1)

    # Threshold of 1 point: > 1 = death, <= 1 = cured
    pred = (total_score > 1).astype(int)
    return pred, total_score

def decision_tree(x: pd.Series):
    """
    Decision tree as described in the main text.

    Args:
        x (pd.Series): A single sample with ['Lactate Dehydrogenase', 'High-sensitivity C-reactive Protein', 'Lymphocyte (%)'].

    Returns:
        int: 0 if cured, 1 if dead.

    Example:
        df.apply(decision_tree, axis=1)
    """
    if x['Lactate Dehydrogenase'] >= 365:
        return 1

    if x['High-sensitivity C-reactive Protein'] < 41.2:
        return 0

    if x['Lymphocyte (%)'] > 14.7:
        return 0
    else:
        return 1

def get_time_in_advance_of_predict(data):
    """
    Get the number of days in advance of correct prediction.

    Args:
        data (pd.DataFrame): Time series data with PATIENT_ID as primary index and t_diff as secondary index.

    Returns:
        pd.Series: Index as PATIENT_ID and values as days in advance of correct prediction.

    Example:
        advance_days = get_time_in_advance_of_predict(data)
    """
    data = data.copy()
    data['right'] = data['pred'] == data['Discharge Method']
    time_advance = []

    for id_ in data.index.remove_unused_levels().levels[0]:
        d = data.loc[id_]

        if len(d) == 1:
            if d.iloc[0]['right']:
                time_advance.append([id_, d.iloc[0].name, d['Discharge Method'].iat[0]])
            continue

        if not d.iloc[0]['right']:
            continue

        if d['right'].all():
            time_advance.append([id_, d.iloc[-1].name, d['Discharge Method'].iat[0]])
            continue

        for i in range(len(d)):
            if d.iloc[i]['right']:
                continue
            else:
                time_advance.append([id_, d.iloc[i-1].name, d['Discharge Method'].iat[0]])
                break

    time_advance = pd.DataFrame(time_advance, columns=['PATIENT_ID', 'time_advance', 'outcome'])
    time_advance = time_advance.set_index('PATIENT_ID')
    return time_advance

class Metrics:
    """
    Metrics class for recording and printing evaluation metrics.

    Attributes:
        y_trues (list): List of true labels.
        y_preds (list): List of predicted labels.
        report (list or None): Report option.
        acc (list or None): Accuracy option.
        f1 (list or None): F1-score option.
        conf_mat (list or None): Confusion matrix option.
    """

    def __init__(self, report=None, acc=None, f1=None, conf_mat=None):
        self.y_trues  = []
        self.y_preds  = []

        if isinstance(report, list):
            self.report = report
        else:
            self.report = [report]

        if isinstance(acc, list):
            self.acc = acc
        else:
            self.acc = [acc]

        if isinstance(f1, list):
            self.f1 = f1
        else:
            self.f1 = [f1]

        if isinstance(conf_mat, list):
            self.conf_mat = conf_mat
        else:
            self.conf_mat = [conf_mat]

    def record(self, y_true, y_pred):
        """
        Record true and predicted labels.

        Args:
            y_true (np.array): True labels.
            y_pred (np.array): Predicted labels.

        Returns:
            Metrics: Self.
        """
        self.y_trues.append(y_true)
        self.y_preds.append(y_pred)
        return self

    def clear(self):
        """
        Clear recorded true and predicted labels.

        Returns:
            Metrics: Self.
        """
        self.y_trues = []
        self.y_preds = []
        return self

    def print_metrics(self):
        """
        Print evaluation metrics.

        Example:
            metrics = Metrics(report='overall', acc='every', f1='overall')
            metrics.record(y_true, y_pred)
            metrics.print_metrics()
        """
        acc_values, f1_values = []
        single_fold = True if len(self.y_trues) == 1 else False

        for i, (y_true, y_pred) in enumerate(zip(self.y_trues, self.y_preds)):
            assert (y_true.ndim == 1) and (y_pred.ndim == 1)
            print(f'\n======================== Fold {i+1} Metrics ========================>')

            if (self.report is not None) and ('every' in self.report):
                print(classification_report(y_true, y_pred))

            a_v = accuracy_score(y_true, y_pred)
            acc_values.append(a_v)
            if (self.acc is not None) and ('every' in self.acc):
                print(f"accuracy: {a_v:.05f}")

            f1_v = f1_score(y_true, y_pred, average='macro')
            f1_values.append(f1_v)
            if (self.f1 is not None) and ('every' in self.f1):
                print(f"F1: {f1_v:.05f}")

            if (self.conf_mat is not None) and ('every' in self.conf_mat):
                print(f"Confusion Matrix:\n{confusion_matrix(y_true, y_pred)}")

        print('\n======================== Overall Metrics ========================>')
        y_true = np.hstack(self.y_trues)
        y_pred = np.hstack(self.y_preds)

        if (self.report is not None) and ('overall' in self.report):
            print(classification_report(y_true, y_pred))

        if (self.acc is not None) and ('overall' in self.acc):
            if single_fold:
                print(f"accuracy:\t{acc_values[0]: .04f}")
            else:
                print(f"accuracy:\t{np.mean(acc_values): .04f} / {'  '.join([str(a_v.round(2)) for a_v in acc_values])}")

        if (self.f1 is not None) and ('overall' in self.f1):
            if single_fold:
                print(f"F1-score:\t{f1_values[0]: .04f}")
            else:
                print(f"F1 Mean:\t{np.mean(f1_values): .04f} / {'  '.join([str(f1_v.round(2)) for f1_v in f1_values])}")

        if (self.conf_mat is not None) and ('overall' in self.conf_mat):
            print(f"Confusion Matrix:\n{confusion_matrix(y_true, y_pred)}")

