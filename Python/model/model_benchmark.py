#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File     : model_benchmark.py
@Time     : 2024/07/03 14:32:05
@Author   : RanPeng 
@Version  : pyhton 3.8.6
@Contact  : 21112030023@m.fudan.edu.cn
@License  : (C)Copyright 2022-2023, DingLab-CHINA-SHNAGHAI
@Function : None
'''
# here put the import lib

import pandas as pd
import numpy as np
import pickle
import os
from utils_features_selection import *
from xgboost import plot_tree
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from datetime import datetime
import pydotplus
from save_log import save_print_to_file
from dtreeviz.trees import *

import matplotlib as mpl

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["figure.figsize"] = 20, 20
date_string = datetime.now().strftime("%Y_%m_%d")

def features_selection(outpath=None):
    """
    Select features based on their importance using XGBoost.

    Args:
        outpath (str, optional): Output path to save the results. Defaults to None.

    Returns:
        list: List of selected feature names.
    
    Example:
        selected_features = features_selection(outpath='./results/')
    """
    X_data_all_features = df.drop("Feature", axis=1)
    Y_data = df["Feature"]
    x_col = df.columns[1:]
    
    import_feature = pd.DataFrame()
    import_feature["col"] = x_col
    import_feature["xgb"] = 0

    for i in range(100):
        x_train, x_test, y_train, y_test = train_test_split(
            X_data_all_features, Y_data, test_size=TEST_SIZE, random_state=i
        )
        model = xgb.XGBClassifier(
            max_depth=4,
            learning_rate=0.2,
            reg_lambda=1,
            n_estimators=150,
            subsample=0.9,
            colsample_bytree=0.9,
        )
        model.fit(x_train, y_train)
        import_feature["xgb"] += model.feature_importances_ / 100

    import_feature = import_feature.sort_values(axis=0, ascending=False, by="xgb")
    print("Top 10 features:")
    print(import_feature.head(10))
    
    indices = np.argsort(import_feature["xgb"].values)[::-1]
    Num_f = 12
    indices = indices[:Num_f]

    plt.subplots(figsize=(12, 10))
    g = sns.barplot(
        y=import_feature.iloc[:Num_f]["col"].values[indices],
        x=import_feature.iloc[:Num_f]["xgb"].values[indices],
        orient="h",
    )
    g.set_xlabel("Relative importance", fontsize=18)
    g.set_ylabel("Features", fontsize=18)
    g.tick_params(labelsize=14)
    sns.despine()
    plt.savefig(outpath + "Top 10 feature_importances.pdf")

    import_feature_cols = import_feature["col"].values[:20]
    import_feature.to_csv(outpath + "Top10_features.csv")

    num_i = 1
    val_score_old = 0
    val_score_new = 0
    while val_score_new >= val_score_old:
        val_score_old = val_score_new
        x_col = import_feature_cols[:num_i]
        print(x_col)
        X_data = X_data_all_features[x_col]
        print("5-Fold CV:")
        (
            acc_train,
            acc_val,
            acc_train_std,
            acc_val_std,
        ) = StratifiedKFold_func_with_features_sel(X_data.values, Y_data.values)
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

def single_tree(cols=None):
    """
    Train and visualize a single decision tree model.

    Args:
        cols (list): List of selected feature columns.
    
    Example:
        single_tree(cols=['Feature1', 'Feature2', 'Feature3'])
    """
    cols.insert(0, "Feature")
    selected_features_data = df[cols]
    selected_features_data.to_csv(
        outpath + date_string + "_selected_features_data.csv", index=False
    )
    df_single = selected_features_data
    y_col = df_single["Feature"]
    x_col = df_single.drop("Feature", axis=1)

    x_np = x_col.values
    y_np = y_col.values
    X_train, x_test, y_train, y_test = train_test_split(
        x_np, y_np, test_size=TEST_SIZE, random_state=42
    )
    model = xgb.XGBClassifier(max_depth=4, n_estimators=1)
    model.fit(X_train, y_train)

    pred_train = model.predict(X_train)
    LABELS = ["Good_circulation", "Bad_circulation"]
    matrix = metrics.confusion_matrix(y_train, pred_train)
    plt.figure(figsize=(4.5, 3))
    sns.heatmap(
        matrix,
        cmap="coolwarm",
        linecolor="white",
        linewidths=1,
        xticklabels=LABELS,
        yticklabels=LABELS,
        annot=True,
        fmt="d",
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig(outpath + date_string + "_train_CM.pdf")
    print(classification_report(y_train, pred_train))

    pred_test = model.predict(x_test)
    print("True test label:", y_test)
    print("Predict test label:", pred_test.astype("int32"))
    matrix = metrics.confusion_matrix(y_test, pred_test)
    plt.figure(figsize=(4.5, 3))
    sns.heatmap(
        matrix,
        cmap="coolwarm",
        linecolor="white",
        linewidths=1,
        xticklabels=LABELS,
        yticklabels=LABELS,
        annot=True,
        fmt="d",
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig(outpath + date_string + "_test_CM.pdf")
    print(classification_report(y_test, pred_test))

    features = x_col.columns
    ceate_feature_map(features)
    model_single = xgb.XGBClassifier(max_depth=6, n_estimators=1)
    model_single.fit(X_train, y_train)
    graph = xgb.to_graphviz(
        model_single, fmap="xgb.fmap", num_trees=0, **{"size": str(10)}
    )
    graph.render(filename="single-tree.dot")
    graph.save(outpath + date_string + "_single_tree.dot")

    plt.figure(dpi=300, figsize=(8, 6))
    plot_tree(model_single, fmap="xgb.fmap")
    plt.savefig(outpath + date_string + "_single-tree.pdf")

    classifier = model
    iris = load_iris()
    classifier.fit(iris.data, iris.target)

    viz = dtreeviz(
        classifier,
        x_col,
        y_col,
        target_name="variety",
        feature_names=df.Feature,
        class_names=["good-CCC", "bad-CCC"],
    )

    viz.view()

    dot_data = tree.export_graphviz(
        model,
        out_file=None,
        feature_names=df.Feature,
        class_names=["good-CCC", "bad-CCC"],
        filled=True,
        rounded=True,
        special_characters=True,
    )
    graph2 = pydotplus.graph_from_dot_data(dot_data)
    graph2.write_pdf(outpath + date_string + "_tree_split.pdf")

def compare_with_other_method(sub_cols=None, auc_type="Train"):
    """
    Compare different models using selected features.

    Args:
        sub_cols (list): List of selected feature columns.
        auc_type (str): Type of AUC to plot ('Train' or 'Test'). Defaults to 'Train'.
    
    Example:
        compare_with_other_method(sub_cols=['Feature1', 'Feature2'], auc_type='Test')
    """
    selected_features_data = df[sub_cols]
    print(selected_features_data)
    selected_features_data.to_csv(
        outpath + date_string + "_selected_features_data.csv", index=False
    )
    compare_model_data = selected_features_data
    x_np = compare_model_data.drop("Feature", axis=1)
    y_np = compare_model_data["Feature"]
    x_col = compare_model_data.columns[1:]

    X_train, X_val, y_train, y_val = train_test_split(
        x_np, y_np, test_size=TEST_SIZE, random_state=15
    )

    xgb_n_clf = xgb.XGBClassifier(
        max_depth=3,
        learning_rate=0.3,
        reg_lambda=1,
        n_estimators=150,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=0,
    )
    tree_clf = tree.DecisionTreeClassifier(
        random_state=0, max_depth=4
    )
    RF_clf1 = RandomForestClassifier(random_state=0, n_estimators=150, max_depth=4)
    LR_clf = linear_model.LogisticRegression(random_state=0, C=1, solver="lbfgs")
    LR_reg_clf = linear_model.LogisticRegression(random_state=0, C=0.1, solver="lbfgs")

    fig = plt.figure(dpi=400, figsize=(16, 12))
    Num_iter = 100

    i = 0
    labels_names = []
    Moodel_name = [
        "Multi-tree XGBoost with all features",
        "Decision tree with all features",
        "Random Forest with all features",
        "Logistic regression with all features with regularization parameter = 1 (by default)",
        "Logistic regression with all features with regularization parameter = 10",
    ]

    AUC_test = ["Train", "Test"]
    for model in [xgb_n_clf, tree_clf, RF_clf1, LR_clf, LR_reg_clf]:
        print("Model:" + Moodel_name[i])
        acc_train, acc_val, acc_train_std, acc_val_std = StratifiedKFold_func(
            x_np.values, y_np.values, Num_iter, model, score_type="f1"
        )
        acc_train, acc_val, acc_train_std, acc_val_std = StratifiedKFold_func(
            x_np.values, y_np.values, Num_iter, model, score_type="auc"
        )
        print(
            "AUC of Train:%.6f with std:%.4f \nAUC of Validation:%.6f with std:%.4f "
            % (acc_train, acc_train_std, acc_val, acc_val_std)
        )

        model.fit(X_train, y_train)
        pred_train_probe = model.predict_proba(X_train)[:, 1]
        pred_val_probe = model.predict_proba(X_val)[:, 1]
        if auc_type == "Test":
            plot_roc(
                y_val, pred_val_probe, Moodel_name[i], fig, labels_names, i
            )
        elif auc_type == "Train":
            plot_roc(
                y_train, pred_train_probe, Moodel_name[i], fig, labels_names, i
            )
        print("AUC socre:", roc_auc_score(y_val, pred_val_probe), "\n")

        i += 1

    X_train, X_val, y_train, y_val = train_test_split(
        x_np, y_np, test_size=TEST_SIZE, random_state=6
    )

    xgb_clf = xgb.XGBClassifier(
        max_depth=3, n_estimators=1, random_state=0,
    )

    tree_clf = tree.DecisionTreeClassifier(random_state=0, max_depth=3)
    RF_clf2 = RandomForestClassifier(random_state=0, n_estimators=1, max_depth=3)

    feature_num = len(x_np.columns)
    Moodel_name = [
        "Single-tree XGBoost with {} features".format(feature_num),
        "Decision tree with {} features".format(feature_num),
        "Random Forest with a single {} constraint with three features".format(
            feature_num
        ),
    ]
    for model in [xgb_clf, tree_clf, RF_clf2]:
        print("Model" + Moodel_name[i - 5])
        acc_train, acc_val, acc_train_std, acc_val_std = StratifiedKFold_func(
            x_np.values, y_np.values, Num_iter, model, score_type="f1"
        )
        print(
            "F1-score of Train:%.6f with std:%.4f \nF1-score of Validation:%.4f with std:%.6f "
            % (acc_train, acc_train_std, acc_val, acc_val_std)
        )
        acc_train, acc_val, acc_train_std, acc_val_std = StratifiedKFold_func(
            x_np.values, y_np.values, Num_iter, model, score_type="auc"
        )
        print(
            "AUC of Train:%.6f with std:%.4f \nAUC of Validation:%.6f with std:%.4f "
            % (acc_train, acc_train_std, acc_val, acc_val_std)
        )

        model.fit(X_train, y_train)
        pred_train_probe = model.predict_proba(X_train)[:, 1]
        pred_val_probe = model.predict_proba(X_val)[:, 1]
        if auc_type == "Test":
            plot_roc(
                y_val, pred_val_probe, Moodel_name[i - 5], fig, labels_names, i
            )
        elif auc_type == "Train":
            plot_roc(
                y_train, pred_train_probe, Moodel_name[i - 5], fig, labels_names, i
            )
        print("AUC socre:", roc_auc_score(y_val, pred_val_probe), "\n")

        i += 1

        plt.plot([0, 1], [0, 1], "r--")
        plt.legend(loc="lower right", fontsize=14)
        if auc_type == "Test":
            plt.savefig(outpath + date_string + "_AUC_test.pdf")
        elif auc_type == "Train":
            plt.savefig(outpath + date_string + "_AUC_train.pdf")

if __name__ == "__main__":

    TEST_SIZE = 0.3
    inpath = "/Volumes/Samsung_T5/xinjiang_CTO/model/model_blend/model_blend_input.csv"
    df = pd.read_csv(inpath, skiprows=0)
    outpath = r"/Volumes/Samsung_T5/xinjiang_CTO/results/XGBoost/model_blend/result/"
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    save_print_to_file(outpath)

    selected_cols = [
        "Basophil_count",
        "Erythrocyte_distribution_width",
        "PT_activity",
        "Medical_History_Year",
        "Chlorine",
        "Serum_HDL_cholesterol",
        "SOD1",
        "UFD1",
        "DDX5",
        "TOM1",
        "MIA3",
        "LMAN2",
    ]
    print(f"The selected features are: {selected_cols}")
    single_tree(cols=selected_cols)
    print("Compare with other methods")
    AUC_test = ["Train", "Test"]
    for auc_type in AUC_test:
        compare_with_other_method(sub_cols=selected_cols, auc_type=auc_type)
