import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd


def plot_confusion_matrix(y_true, y_pred, classes):
    """Plots confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cm, index=[i for i in classes], columns=[i for i in classes])
    plt.figure(figsize=(6, 6))
    ax = sns.heatmap(df_cm, annot=True, square=True, fmt="d", linewidths=.2, cmap="RdYlGn", cbar_kws={"shrink": 0.8})
    return ax


def plot_classification_report(y_true, y_pred, target_names):
    """Plots classification report."""
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    sns.heatmap(pd.DataFrame(report).iloc[:-1, :].T, cmap="RdYlGn", annot=True)
