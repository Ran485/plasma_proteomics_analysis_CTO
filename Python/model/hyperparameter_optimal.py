#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File     : hyperparameter_optimal.py
@Time     : 2024/07/03 15:11:01
@Author   : RanPeng 
@Version  : pyhton 3.8.6
@Contact  : 21112030023@m.fudan.edu.cn
@License  : (C)Copyright 2022-2023, DingLab-CHINA-SHNAGHAI
@Function : None
'''
# here put the import lib

import xgboost as xgb
import shap
import os
import pandas as pd
import numpy as np
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt

from itertools import cycle
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

import matplotlib as mpl

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["figure.dpi"] = 400  # 一般将dpi设置在150到300之间

custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)


def plot_optimization_history(
    trials, window=10, outpath=None, file_name="Hyperparametric Optimization Curve.pdf"
):
    """
    This function plots optimization history.
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    # Extract the values from the trial history
    losses = [x["result"]["loss"] for x in trials.trials]

    # Create a pandas Series for easy window-based calculations
    losses_series = pd.Series(losses)

    # Calculate the moving average
    losses_moving_avg = losses_series.rolling(window=window).mean()

    # Plot the raw losses and the moving average
    ax.plot(losses, alpha=0.3, linewidth=2, color="red", label="Raw loss")
    ax.plot(
        losses_moving_avg,
        linewidth=2,
        color="darkblue",
        label=f"Moving average (window size={window})",
    )

    ax.set_title("Optimization History", fontsize=18)
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.legend()
    # ax.grid(True)
    if outpath:
        outpath = outpath + file_name
        plt.savefig(outpath, dpi=500, bbox_inches="tight")
    else:
        plt.show()


def plot_evals_result(
    evals_result,
    metric,
    outpath=None,
    file_name="XGBoost Training Vs Validation Accuracy.pdf",
):
    """
    Function to plot the training and validation errors from the model training process

    Parameters:
    evals_result: The evals result from the xgboost model training process
    metric: The metric to plot. For instance, 'auc'

    Returns:
    A seaborn plot
    """
    train_errors = evals_result["train"][metric]
    val_errors = evals_result["test"][metric]

    plt.figure(figsize=(6, 4))
    sns.lineplot(data=train_errors, label="Train")
    sns.lineplot(data=val_errors, label="Validation")
    plt.xlabel("Iterations", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title("XGBoost Training Vs Validation Accuracy", fontsize=15)
    plt.legend()
    # plt.grid(True)
    if outpath:
        outpath = outpath + file_name
        plt.savefig(outpath, dpi=500, bbox_inches="tight")
    else:
        plt.show()


def plot_model_loss_curve(
    evals_result,
    metric,
    outpath=None,
    file_name="XGBoost Training, Validation Loss over Epochs.pdf",
):
    """
    Function to plot the training loss, validation loss, and their sum from the model training process

    Parameters:
    evals_result: The evals result from the xgboost model training process
    metric: The metric to plot. For instance, 'auc'

    Returns:
    A seaborn plot
    """
    train_errors = evals_result["train"][metric]
    val_errors = evals_result["test"][metric]
    combined_errors = [sum(x) for x in zip(train_errors, val_errors)]

    plt.figure(figsize=(6, 4))
    sns.lineplot(data=train_errors, label="Train")
    sns.lineplot(data=val_errors, label="Validation")
    plt.xlabel("Iterations", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("XGBoost Training, Validation Loss over Epochs", fontsize=15)
    plt.legend()
    # plt.grid(True)
    if outpath:
        outpath = outpath + file_name
        plt.savefig(outpath, dpi=500, bbox_inches="tight")
    else:
        plt.show()


def plot_learning_rate_curve(
    evals_result, outpath=None, file_name="XGBoost Learning Rate Curve.pdf"
):
    """
    Function to plot the training loss and test accuracy from the model training process

    Parameters:
    evals_result: The evals result from the xgboost model training process

    Returns:
    A seaborn plot
    """
    train_loss = evals_result["train"]["logloss"]
    # test_accuracy = [1-x for x in evals_result['test']['error']]
    test_accuracy = evals_result["test"]["auc"]

    fig, ax1 = plt.subplots(figsize=(6, 4))

    color = "tab:red"
    ax1.set_xlabel("Iteration", fontsize=14)
    ax1.set_ylabel("Train Log Loss", color=color, fontsize=14)
    ax1 = sns.lineplot(data=train_loss, color=color, ax=ax1, label="Train Log Loss")
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = "tab:blue"
    ax2.set_ylabel("Test Accuracy", color=color, fontsize=14)
    ax2 = sns.lineplot(data=test_accuracy, color=color, ax=ax2, label="Test Accuracy")
    ax2.tick_params(axis="y", labelcolor=color)

    fig.tight_layout()
    plt.title("Training Loss and Test Accuracy over Iterations", fontsize=16)

    # Adding legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="best")

    if outpath:
        outpath = outpath + file_name
        plt.savefig(outpath, dpi=500, bbox_inches="tight")
    else:
        plt.show()


def plot_roc_curve_multiclass(X, y):
    # Binarize the output
    y = label_binarize(y, classes=[0, 1, 2])
    n_classes = y.shape[1]

    # Learn to predict each class against the other
    # classifier = OneVsRestClassifier(
    #     svm.SVC(kernel="linear", probability=True, random_state=random_state)
    # )
    # y_score = classifier.fit(X_train, y_train).decision_function(X_test)

    # Learn to predict each class against the other
    classifier = OneVsRestClassifier(
        XGBClassifier(
            use_label_encoder=False, eval_metric="logloss", random_state=random_state
        )
    )
    y_score = classifier.fit(X_train, y_train).predict_proba(X_test)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    plt.figure(figsize=(4, 4))
    lw = 1.5

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()

    colors = cycle(["aqua", "darkorange", "cornflowerblue"])
    for i, color in zip(range(n_classes), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=lw,
            label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]),
        )

    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
        color="deeppink",
        linestyle="--",
        linewidth=1.5,
    )

    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
        color="navy",
        linestyle="--",
        linewidth=1.5,
    )

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves for XGBClassifier model")
    plt.legend(loc="lower right")
    plt.savefig(outpath + "/roc_curve.pdf", dpi=500, bbox_inches="tight")


def plot_roc_curve_binary(X, y):
    # shuffle and split training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0
    )

    # Learn to predict the positive class
    classifier = XGBClassifier(
        use_label_encoder=False, eval_metric="logloss", random_state=42
    )
    y_score = classifier.fit(X_train, y_train).predict_proba(X_test)[:, 1]

    # Compute ROC curve and ROC area
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure(figsize=(4, 4))
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=1.5,
        label="ROC curve (area = {0:0.2f})".format(roc_auc),
    )
    plt.plot([0, 1], [0, 1], "k--", lw=1.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve for XGBClassifier model")
    plt.legend(loc="lower right")
    plt.savefig(outpath + "/roc_curve.pdf", dpi=500, bbox_inches="tight")


def preprocess_data(df, target_col, test_size=0.25, random_state=42):
    """
    Function to preprocess the data

    Parameters:
    df: The dataframe containing the data
    target_col: The name of the target column
    test_size: The size of the test set
    random_state: The random state to use for reproducibility

    Returns:
    X_train: The training features
    X_test: The test features
    y_train: The training target
    y_test: The test target
    """
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    # split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Create classification matrices
    # To set enable_categorical to True, enable automatic encoding of Pandas category columns
    dtrain_class = xgb.DMatrix(X_train, y_train, enable_categorical=True)
    dtest_class = xgb.DMatrix(X_test, y_test, enable_categorical=True)

    evals = [(dtrain_class, "train"), (dtest_class, "test")]
    return X, y, X_train, X_test, y_train, y_test, evals, dtrain_class, dtest_class


# Define the objective function for Bayesian Optimization
def objective(hyperparameters):
    params = {
        "num_boost_round": int(hyperparameters["n_estimators"]),
        "max_depth": int(hyperparameters["max_depth"]),
        "gamma": hyperparameters["gamma"],
        "reg_alpha": hyperparameters["reg_alpha"],
        "reg_lambda": hyperparameters["reg_lambda"],
        "colsample_bytree": int(hyperparameters["colsample_bytree"]),
        "min_child_weight": int(hyperparameters["min_child_weight"]),
        "eta": hyperparameters["eta"],
        "subsample": int(hyperparameters["subsample"]),
        "eval_metric": "auc",
        "objective": "binary:logistic",
        "booster": "gbtree",
        "tree_method": "hist",
        "seed": 42,  # This sets the random seed for reproducibility
    }
    # num_round = int(params['n_estimators'])
    # del params['n_estimators']
    model = xgb.train(params=params, dtrain=dtrain_class, evals=evals, verbose_eval=50)
    pred = model.predict(dtest_class, iteration_range=(1, model.best_ntree_limit))
    score = roc_auc_score(y_test, pred)
    print("SCORE:", score)
    return {"loss": 1 - score, "status": STATUS_OK}


def config(outpath):
    # Define hyperparameters for optimization
    hyperparameter_grid = {
        "num_boost_round": hp.quniform("n_estimators", 100, 1000, 1),
        "max_depth": hp.quniform("max_depth", 3, 18, 1),
        "gamma": hp.quniform("gamma", 0.1, 1, 0.25),
        "reg_alpha": hp.quniform("reg_alpha", 0, 1, 0.01),
        "reg_lambda": hp.quniform("reg_lambda", 0, 1, 0.01),
        "colsample_bytree": hp.quniform("colsample_bytree", 0.5, 1, 0.05),
        "min_child_weight": hp.quniform("min_child_weight", 0, 10, 1),
        "n_estimators": 180,
        "eta": hp.quniform("eta", 0.025, 0.5, 0.05),
        "subsample": hp.quniform("subsample", 0.5, 1, 0.05),
        "seed": 0,
        "eval_metric": "auc",
        "objective": "binary:logistic",
        "booster": "gbtree",
        "tree_method": "exact",
    }

    # # Define hyperparameters for optimization (finer grid)
    # hyperparameter_grid = {
    #     'n_estimators': hp.quniform('n_estimators', 100, 1000, 1),
    #     'max_depth': hp.quniform("max_depth", 3, 18, 1),
    #     'gamma': hp.quniform('gamma', 0.1, 1, 0.05),  # finer grid
    #     'reg_alpha': hp.quniform('reg_alpha', 0, 1, .01),
    #     'reg_lambda': hp.quniform('reg_lambda', 0, 1, .01),
    #     'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.05),
    #     'min_child_weight': hp.quniform('min_child_weight', 0, 10, 1),
    #     'eta': hp.quniform('eta', 0.025, 0.5, 0.01),  # finer grid
    #     'subsample': hp.quniform('subsample', 0.5, 1, 0.05),
    #     'seed': 0,
    #     'eval_metric': 'auc',
    #     'objective': 'binary:logistic',
    #     'booster': 'gbtree',
    #     'tree_method': 'exact'
    # }

    # # load data
    # df = pd.read_csv(inpath)
    # X_train, X_test, y_train, y_test, evals, dtrain_class, dtest_class = preprocess_data(df,
    #                             target_col=target_col,
    #                             test_size=test_size,
    #                             random_state=random_state)

    # Define hyperparameter optimization loop
    trials = Trials()
    best_hyperparams = fmin(
        fn=objective,
        space=hyperparameter_grid,
        algo=tpe.suggest,
        max_evals=200,
        trials=trials,
        rstate=np.random.RandomState(42),
    )

    print("The best hyperparameters are : ", "\n")
    print(best_hyperparams)
    plot_optimization_history(trials, window=10, outpath=outpath)

    params = {
        "num_boost_round": int(best_hyperparams["n_estimators"]),
        "max_depth": int(best_hyperparams["max_depth"]),
        "min_child_weight": int(best_hyperparams["min_child_weight"]),
        "eta": best_hyperparams["eta"],
        "subsample": best_hyperparams["subsample"],
        "colsample_bytree": best_hyperparams["colsample_bytree"],
        "gamma": best_hyperparams["gamma"],
        "reg_alpha": best_hyperparams["reg_alpha"],
        "reg_lambda": best_hyperparams["reg_lambda"],
        "objective": "binary:logistic",
        "tree_method": "hist",
        "eval_metric": ["auc", "error", "logloss"],
    }

    evals_result = {}

    # Train the model with best hyperparameters
    model = xgb.train(
        params=params,
        dtrain=dtrain_class,
        evals=evals,
        num_boost_round=int(best_hyperparams["n_estimators"]),
        verbose_eval=100,
        evals_result=evals_result,
        early_stopping_rounds=200,  # Activate early stopping
    )

    plot_evals_result(evals_result, metric="auc", outpath=outpath)
    plot_model_loss_curve(evals_result, metric="logloss", outpath=outpath)
    plot_learning_rate_curve(evals_result, outpath=outpath)

    # Evaluate the accuracy of the model
    y_pred = model.predict(dtest_class)
    predictions = [round(value) for value in y_pred]
    # evaluate predictions
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))

    # 2、 Explain the model's predictions using SHAP values
    model = xgb.train(
        params,
        dtrain_class,
        500,
        evals=[(dtest_class, "test")],
        verbose_eval=100,
        early_stopping_rounds=100,
    )
    # this takes a minute or two since we are explaining over 30 thousand samples in a model with over a thousand trees
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # shap.force_plot(
    #     explainer.expected_value, shap_values[0, :], X_display.iloc[0, :], show=False
    # )
    # plt.savefig("force_plot.pdf", bbox_inches="tight")
    # plt.close()

    # shap.summary_plot(shap_values, X_display, plot_type="bar")
    # plt.savefig("summary_barplot.pdf", bbox_inches="tight")
    # plt.close()

    # Plot and save the summary_plot figure
    shap.summary_plot(shap_values, X, show=False)
    plt.savefig(outpath + "/summary_plot.pdf", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":

    inpath = "/Volumes/Samsung_T5/xinjiang_CTO/model_blend/clinical_proteome_combined_discovery_input.csv"
    # load data
    df = pd.read_csv(inpath)
    (
        X,
        y,
        X_train,
        X_test,
        y_train,
        y_test,
        evals,
        dtrain_class,
        dtest_class,
    ) = preprocess_data(df, target_col="Feature", test_size=0.25, random_state=42)
    outpath = "/Volumes/Samsung_T5/xinjiang_CTO/model_results/all_feature_selection/HPO/"
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    # config(outpath)
    # y = label_binarize(y, classes=[0, 1])
    # n_classes = y.shape[1]

    plot_roc_curve_binary(X, y)
