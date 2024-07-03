#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File     : Fature_selection.py
@Time     : 2024/07/03 15:11:10
@Author   : RanPeng 
@Version  : pyhton 3.8.6
@Contact  : 21112030023@m.fudan.edu.cn
@License  : (C)Copyright 2022-2023, DingLab-CHINA-SHNAGHAI
@Function : None
'''
# here put the import lib

import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import KFold

from utils_features_selection import *
from save_log import save_print_to_file

# python matplotlib PDF 不断字
import matplotlib as mpl
from datetime import datetime

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["figure.figsize"] = 20, 20
date_string = datetime.now().strftime("%Y_%m_%d")


class FeatureSelector:

    def __init__(
        self, models, feature_names, n_iterations=5, n_repeats=10, outpath=None
    ):
        self.models = models
        self.feature_names = feature_names
        self.n_iterations = n_iterations
        self.n_repeats = n_repeats
        self.feature_importances = pd.DataFrame(index=feature_names)
        self.outpath = outpath

    def fit(self, X, y):
        repeat_importances = {model.__class__.__name__: [] for model in self.models}

        for i_repeat in range(self.n_repeats):
            kf = KFold(n_splits=self.n_iterations, shuffle=True, random_state=i_repeat)

            for model in self.models:
                fold_importances = []
                for train_index, _ in kf.split(X):
                    X_train, y_train = X.iloc[train_index], y.iloc[train_index]
                    model.fit(X_train, y_train)
                    if hasattr(model, "feature_importances_"):
                        importance = model.feature_importances_
                    elif hasattr(model, "coef_"):
                        importance = abs(model.coef_[0])
                    else:
                        continue
                    importance = (importance - importance.min()) / (
                        importance.max() - importance.min()
                    )
                    fold_importances.append(importance)

                repeat_importances[model.__class__.__name__].append(
                    np.mean(fold_importances, axis=0)
                )

        for model_name, importances in repeat_importances.items():
            importances = np.array(importances)
            self.feature_importances[model_name] = importances.mean(axis=0)
            self.feature_importances[model_name + "_std"] = importances.std(axis=0)

    def rank_features(self, X, y):
        self.fit(X, y)
        model_weights = self.get_model_weights(X, y)
        print(f"The model weights are: {model_weights}")

        weighted_importance = []
        weighted_importance_std = []
        for model in self.models:
            model_name = model.__class__.__name__
            weighted_score = (
                self.feature_importances[model_name] * model_weights[model_name]
            )
            weighted_score_std = (
                self.feature_importances[model_name + "_std"]
                * model_weights[model_name]
            )
            weighted_importance.append(weighted_score)
            weighted_importance_std.append(weighted_score_std)

        self.feature_importances["Weighted Importance Score"] = np.mean(
            weighted_importance, axis=0
        )
        self.feature_importances["Weighted Importance Score_std"] = np.mean(
            weighted_importance_std, axis=0
        )

        # return self.feature_importances['Weighted Importance Score'].sort_values()
        return self.feature_importances[
            ["Weighted Importance Score", "Weighted Importance Score_std"]
        ].sort_values(by="Weighted Importance Score", ascending=False)

    def get_model_weights(self, X, y):
        model_weights = {}
        for model in self.models:
            model_name = model.__class__.__name__
            score = np.mean(cross_val_score(model, X, y, cv=5))
            model_weights[model_name] = score
        total = sum(model_weights.values())
        for model in model_weights:
            model_weights[model] /= total
        return model_weights

    def export_feature_importances(self, file_name):
        self.feature_importances.to_csv(self.outpath + "/" + file_name)

    def plot_feature_importances(self, top_n=10):
        feature_importances = self.feature_importances[
            [
                col
                for col in self.feature_importances.columns
                if not col.endswith("_std")
            ]
        ]
        feature_importances = feature_importances.sort_values(
            "Weighted Importance Score", ascending=False
        )
        top_features = feature_importances.iloc[:top_n]
        std_errors = self.feature_importances.loc[
            top_features.index, "Weighted Importance Score_std"
        ]

        # plt.figure(figsize=(10, 6))
        # plt.barh(y=top_features.index,
        #         width=top_features['Weighted Importance Score'].values,
        #         color='b', align='center',
        #         xerr=std_errors.values)
        # plt.xlabel('Weighted Importance Score')
        # plt.ylabel('Features')
        # plt.title('Feature Importance Scores')
        # plt.gca().invert_yaxis()
        # plt.savefig(self.outpath + '/' + 'feature_importance.pdf', dpi=300, bbox_inches='tight')

        # Visualise these with a barplot
        plt.subplots(figsize=(12, 10))
        g = sns.barplot(
            x=top_features["Weighted Importance Score"].values,
            y=top_features.index,
            xerr=std_errors.values,
            orient="h",
        )
        g.set_xlabel("Weighted Importance Score", fontsize=18)
        g.set_ylabel("Features", fontsize=18)
        g.set_title(
            "The Weight Feature Importance Scores of Baseline Models", fontsize=18
        )
        g.tick_params(labelsize=14)
        sns.despine()
        plt.savefig(self.outpath + "/" + "Top 10 feature_importances.pdf")
        # Get the top 10 important features
        top_features.to_csv(outpath + "Top10_features.csv")

    def iterative_select_features(
        self, import_feature_cols, X_data_all_features, Y_data
    ):
        # 画特征金字塔
        num_i = 1
        val_score_old = 0
        val_score_new = 0
        while val_score_new >= val_score_old:
            val_score_old = val_score_new
            # 按重要程度顺序取特种
            x_col = import_feature_cols[:num_i]
            print(x_col)
            X_data = X_data_all_features[x_col]  # .values
            ## 交叉验证
            print("5-Fold CV:")
            acc_train, acc_val, acc_train_std, acc_val_std = (
                StratifiedKFold_func_with_features_sel(X_data.values, Y_data.values)
            )
            print(
                "Train AUC-score is %.4f ; Validation AUC-score is %.4f"
                % (acc_train, acc_val)
            )
            print(
                "Train AUC-score-std is %.4f ; Validation AUC-score-std is %.4f"
                % (acc_train_std, acc_val_std)
            )
            val_score_new = acc_val
            num_i += 1

        print("Selected features:", x_col[:-1])
        Selected_features = x_col[:-1]
        return list(x_col[:-1])

def preprocess(data):
    X = data.iloc[:, 1:]
    y = data.iloc[:, 0]
    feature_names = X.columns

    return X, y, feature_names


models = [
    RandomForestClassifier(n_estimators=100, random_state=0),
    XGBClassifier(),
    LGBMClassifier(),
    CatBoostClassifier(verbose=False),
    SVC(kernel="linear"),
    LogisticRegression(max_iter=1000),
]


if __name__ == "__main__":

    # 读取数据
    inpath = "/Volumes/Samsung_T5/xinjiang_CTO/model/XGBoost/input_raw_data_filter.csv"
    data = pd.read_csv(inpath)
    outpath = r"/Volumes/Samsung_T5/xinjiang_CTO/model_results/all_feature_selection_filter/"
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    # Save the output parameters
    save_print_to_file(outpath)

    X, y, gene_names = preprocess(data)
    fs = FeatureSelector(
        models, gene_names, n_iterations=5, n_repeats=10, outpath=outpath
    )
    feature_ranks = fs.rank_features(X, y)

    # 导出特征重要性
    fs.export_feature_importances("feature_importances.csv")

    # 可视化
    fs.plot_feature_importances()

    # 递归特征选择
    # feature_ranks.sort_values('Weighted Importance Score', ascending=False, inplace=True)
    select_features = fs.iterative_select_features(feature_ranks.index, X, y)
