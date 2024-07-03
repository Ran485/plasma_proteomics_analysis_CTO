#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File     : model.py
@Time     : 2024/07/03 15:10:49
@Author   : RanPeng 
@Version  : pyhton 3.8.6
@Contact  : 21112030023@m.fudan.edu.cn
@License  : (C)Copyright 2022-2023, DingLab-CHINA-SHNAGHAI
@Function : None
'''
# here put the import lib

import os
import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import logging
import datetime

from sklearn.metrics import (
    confusion_matrix, roc_auc_score, roc_curve, 
    precision_recall_fscore_support
)
from sklearn.model_selection import train_test_split, KFold
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from scipy.stats import norm

# python matplotlib export editable PDF 
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['figure.dpi']= 150

custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)

import warnings
warnings.filterwarnings('ignore')


def delong_roc_variance(ground_truth, predictions):
    order = np.argsort(-predictions)
    predictions = predictions[order]
    ground_truth = ground_truth[order]

    distinct_value_indices = np.where(np.diff(predictions))[0]
    distinct_value_indices = np.concatenate(([0], distinct_value_indices + 1, [len(predictions)]))

    n_1 = np.sum(ground_truth == 1)
    n_0 = np.sum(ground_truth == 0)
    TPR = np.cumsum(ground_truth) / n_1
    FPR = np.cumsum(1 - ground_truth) / n_0

    v_10 = np.cumsum(1 - ground_truth) / n_0
    v_01 = np.cumsum(ground_truth) / n_1

    aucs = []
    for i in range(len(distinct_value_indices) - 1):
        start, end = distinct_value_indices[i], distinct_value_indices[i + 1]
        if len(FPR[start:end]) >= 2 and len(TPR[start:end]) >= 2:
            aucs.append(auc(FPR[start:end], TPR[start:end]))

    auc_value = sum(aucs)
    s_10 = (n_1 - 1) * np.var(v_10)
    s_01 = (n_0 - 1) * np.var(v_01)
    var_auc = (s_10 + s_01) / (n_1 * n_0)

    return auc_value, var_auc


def delong_roc_test(y_true1, y_scores1, y_true2, y_scores2):
    auc1, var1 = delong_roc_variance(y_true1, y_scores1)
    auc2, var2 = delong_roc_variance(y_true2, y_scores2)
    
    z = (auc1 - auc2) / np.sqrt(var1 + var2)
    p_value = norm.sf(abs(z)) * 2
    
    return p_value, auc1, auc2


class XGBoostModel:
    def __init__(self, discovery_file, val_file, output_dir, target, split_data=False, test_size=0.3, params=None, n_splits=10, random_state=0, num_iter=100, top_n=10):
        self.discovery_file = discovery_file
        self.val_file = val_file
        self.output_dir = output_dir
        self.target = target
        self.split_data = split_data
        self.test_size = test_size
        self.random_state = random_state
        self.num_iter = num_iter
        self.top_n = top_n
        os.makedirs(output_dir, exist_ok=True)
        
        self.params = params if params is not None else {
            'objective': 'binary:logistic', 
            'eval_metric': 'logloss'
        }
        self.n_splits = n_splits

        # Setup logging
        logging.basicConfig(filename=os.path.join(output_dir, 'model.log'), level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger()


    def load_data(self):
        self.logger.info("Loading data...")
        discovery_data = pd.read_csv(self.discovery_file)
        val_data = pd.read_csv(self.val_file)

        if self.split_data:
            train_data, test_data = train_test_split(discovery_data, test_size=self.test_size, random_state=self.random_state)
        else:
            train_data = discovery_data
            test_data = None

        self.logger.info("Data loaded successfully.")
        return train_data, test_data, val_data

    def build_model(self):
        train_data, test_data, val_data = self.load_data()
        X_train, y_train = train_data.drop(columns=[self.target]), train_data[self.target]
        X_val, y_val = val_data.drop(columns=[self.target]), val_data[self.target]
        if self.split_data:
            X_test, y_test = test_data.drop(columns=[self.target]), test_data[self.target]
        else:
            X_test, y_test = None, None

        all_results = []
        kfold_results = []

        self.logger.info("Starting model training...")

        for i in range(self.num_iter):
            kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=i)
            iteration_results = {
                'train': [],
                'test': [],
                'val': []
            }

            for train_index, valid_index in kf.split(X_train):
                X_train_fold, X_valid_fold = X_train.iloc[train_index], X_train.iloc[valid_index]
                y_train_fold, y_valid_fold = y_train.iloc[train_index], y_train.iloc[valid_index]

                dtrain = xgb.DMatrix(X_train_fold, label=y_train_fold)
                dvalid = xgb.DMatrix(X_valid_fold, label=y_valid_fold)
                evals = [(dtrain, 'train'), (dvalid, 'eval')]

                model = xgb.train(self.params, dtrain, 
                                num_boost_round=756, 
                                evals=evals, early_stopping_rounds=20)

                dtrain_full = xgb.DMatrix(X_train_fold, label=y_train_fold)
                y_train_pred = model.predict(dtrain_full)
                iteration_results['train'].append((y_train_fold, y_train_pred))

                if self.split_data:
                    dtest = xgb.DMatrix(X_test, label=y_test)
                    y_test_pred = model.predict(dtest)
                    iteration_results['test'].append((y_test, y_test_pred))

                dval = xgb.DMatrix(X_val, label=y_val)
                y_val_pred = model.predict(dval)
                iteration_results['val'].append((y_val, y_val_pred))

            # Aggregate KFold results for the current random seed
            kfold_aggregated = self.aggregate_kfold_results(iteration_results)
            kfold_results.append(kfold_aggregated)

            # Collect all results
            all_results.append(iteration_results)

        self.evaluate_and_plot(all_results, kfold_results)

    def aggregate_kfold_results(self, iteration_results):
        aggregated = {
            'train': self.aggregate_results(iteration_results['train']),
            'test': self.aggregate_results(iteration_results['test']) if self.split_data else None,
            'val': self.aggregate_results(iteration_results['val'])
        }
        return aggregated

    def aggregate_results(self, results):
        y_true = np.concatenate([res[0] for res in results])
        y_pred = np.concatenate([res[1] for res in results])
        auc = roc_auc_score(y_true, y_pred)
        return {
            'y_true': y_true,
            'y_pred': y_pred,
            'auc': auc
        }

    def evaluate_performance(self, y_true, y_pred):
        y_pred_label = np.where(y_pred > 0.5, 1, 0)
        cm = confusion_matrix(y_true, y_pred_label)
        auc = roc_auc_score(y_true, y_pred)
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred_label, average='binary')
        
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        
        return {
            'confusion_matrix': cm,
            'auc': auc,
            'roc_curve': (fpr, tpr),
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'sensitivity': sensitivity,
            'specificity': specificity
        }

    def evaluate_and_plot(self, all_results, kfold_results):
        # Select top N results based on validation AUC
        val_aucs = [res['val']['auc'] for res in kfold_results]
        top_indices = np.argsort(val_aucs)[-self.top_n:]

        for idx in top_indices:
            fig, ax = plt.subplots(figsize=(6, 5))

            # Train data
            y_train, y_train_pred = kfold_results[idx]['train']['y_true'], kfold_results[idx]['train']['y_pred']
            fpr_train, tpr_train, _ = roc_curve(y_train, y_train_pred)
            ax.plot(fpr_train, tpr_train, label=f'Train AUC: {kfold_results[idx]["train"]["auc"]:.2f}')
            train_ci_lower, train_ci_upper = np.percentile([res['train']['auc'] for res in kfold_results], [2.5, 97.5])
            train_mean_auc = np.mean([res['train']['auc'] for res in kfold_results])
            train_median_auc = np.median([res['train']['auc'] for res in kfold_results])
            ax.axhline(train_mean_auc, color='blue', linestyle='--', label=f'Train Mean AUC: {train_mean_auc:.2f} [95% CI: {train_ci_lower:.2f} - {train_ci_upper:.2f}]')

            # Test data
            if self.split_data:
                y_test, y_test_pred = kfold_results[idx]['test']['y_true'], kfold_results[idx]['test']['y_pred']
                fpr_test, tpr_test, _ = roc_curve(y_test, y_test_pred)
                ax.plot(fpr_test, tpr_test, label=f'Test AUC: {kfold_results[idx]["test"]["auc"]:.2f}')
                test_ci_lower, test_ci_upper = np.percentile([res['test']['auc'] for res in kfold_results], [2.5, 97.5])
                test_mean_auc = np.mean([res['test']['auc'] for res in kfold_results])
                test_median_auc = np.median([res['test']['auc'] for res in kfold_results])
                ax.axhline(test_mean_auc, color='green', linestyle='--', label=f'Test Mean AUC: {test_mean_auc:.2f} [95% CI: {test_ci_lower:.2f} - {test_ci_upper:.2f}]')

            # Val data
            y_val, y_val_pred = kfold_results[idx]['val']['y_true'], kfold_results[idx]['val']['y_pred']
            fpr_val, tpr_val, _ = roc_curve(y_val, y_val_pred)
            ax.plot(fpr_val, tpr_val, label=f'Val AUC: {kfold_results[idx]["val"]["auc"]:.2f}')
            val_ci_lower, val_ci_upper = np.percentile([res['val']['auc'] for res in kfold_results], [2.5, 97.5])
            val_mean_auc = np.mean([res['val']['auc'] for res in kfold_results])
            val_median_auc = np.median([res['val']['auc'] for res in kfold_results])
            ax.axhline(val_mean_auc, color='red', linestyle='--', label=f'Val Mean AUC: {val_mean_auc:.2f} [95% CI: {val_ci_lower:.2f} - {val_ci_upper:.2f}]')

            ax.plot([0, 1], [0, 1], 'k--')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'ROC Curves (Top {idx + 1})')
            ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'roc_curve_with_ci_Top{idx + 1}.pdf'))
            plt.close(fig)

        # Save all_results and kfold_results to CSV files
        all_metrics = ['auc', 'precision', 'recall', 'f1_score', 'sensitivity', 'specificity']
        all_results_dict = {f'{set_}_{metric}': [] for set_ in ['train', 'test', 'val'] for metric in all_metrics}
        for i, res in enumerate(all_results):
            for k in range(self.n_splits):
                for set_ in ['train', 'test', 'val']:
                    if set_ == 'test' and not self.split_data:
                        continue
                    y_true, y_pred = res[set_][k]
                    perf = self.evaluate_performance(y_true, y_pred)
                    for metric in all_metrics:
                        all_results_dict[f'{set_}_{metric}'].append(perf[metric])

        all_results_df = pd.DataFrame(all_results_dict)
        all_results_df.to_csv(os.path.join(self.output_dir, 'all_results.csv'), index=False)

        kfold_results_dict = {f'{set_}_{metric}': [] for set_ in ['train', 'test', 'val'] for metric in all_metrics}
        for res in kfold_results:
            for set_ in ['train', 'test', 'val']:
                if set_ == 'test' and not self.split_data:
                    continue
                perf = self.evaluate_performance(res[set_]['y_true'], res[set_]['y_pred'])
                for metric in all_metrics:
                    kfold_results_dict[f'{set_}_{metric}'].append(perf[metric])

        kfold_results_df = pd.DataFrame(kfold_results_dict)
        kfold_results_df.to_csv(os.path.join(self.output_dir, 'kfold_results.csv'), index=False)

        # Plot confusion matrices and barplot of performance metrics
        self.plot_metrics(all_results, kfold_results)

    def plot_metrics(self, all_results, kfold_results):
        sets = ['train', 'test', 'val'] if self.split_data else ['train', 'val']
        metrics = ['auc', 'precision', 'recall', 'f1_score', 'sensitivity', 'specificity']

        metric_values = {set_: [] for set_ in sets}
        for results in kfold_results:
            for set_ in sets:
                if set_ == 'test' and not self.split_data:
                    continue
                perf = self.evaluate_performance(results[set_]['y_true'], results[set_]['y_pred'])
                metric_values[set_].append(perf)

        metric_means = {metric: [np.mean([res[metric] for res in metric_values[set_]]) for set_ in sets] for metric in metrics}
        metric_stds = {metric: [np.std([res[metric] for res in metric_values[set_]]) for set_ in sets] for metric in metrics}

        df_means = pd.DataFrame(metric_means, index=sets).T
        df_stds = pd.DataFrame(metric_stds, index=sets).T

        fig, ax = plt.subplots(figsize=(10, 6))
        df_means.plot(kind='bar', yerr=df_stds, capsize=4, ax=ax)
        plt.title('Performance Metrics Comparison with Error Bars')
        plt.xlabel('Metrics')
        plt.ylabel('Scores')
        plt.xticks(rotation=45)
        plt.legend(title='Datasets')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'performance_metrics_comparison_with_error_bars.pdf'))
        plt.close(fig)

        fig, axes = plt.subplots(1, len(sets), figsize=(20, 5))
        for i, set_ in enumerate(sets):
            cm = np.mean([self.evaluate_performance(res[set_]['y_true'], res[set_]['y_pred'])['confusion_matrix'] for res in kfold_results], axis=0)
            cm = cm // self.n_splits
            cm_int = cm.astype(int)
            sns.heatmap(cm_int, annot=True, fmt="d", ax=axes[i], cmap="Blues")
            axes[i].set_title(f'{set_.capitalize()} Confusion Matrix')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        plt.savefig(os.path.join(self.output_dir, 'confusion_matrices.pdf'))
        plt.close(fig)


if __name__ == '__main__':
    # The best hyperparams of the Combined Features 
    Combined_best_hyperparams = {
        "colsample_bytree": 0.5,
        "eta": 0.30000000000000004,
        "gamma": 0.0,
        "max_depth": 6,
        "min_child_weight": 0.0,
        "n_estimators": 374.0,
        "reg_alpha": 0.0,
        "reg_lambda": 0.43,
        "subsample": 0.9500000000000001,
    }

    model = XGBoostModel(
    n_splits=10, 
    random_state=15,
    discovery_file='/Volumes/Samsung_T5/xinjiang_CTO/model/Combined/clinical_proteome_combined_discovery_input.csv',
    val_file='/Volumes/Samsung_T5/xinjiang_CTO/model/Combined/clinical_proteome_combined_validation_input.csv',
    output_dir='/Volumes/Samsung_T5/xinjiang_CTO/model/Results/Combined/combined_features_20240612',
    target='Feature',
    split_data=True,
    test_size=0.30,
    params=Combined_best_hyperparams
    )

    results = model.build_model()