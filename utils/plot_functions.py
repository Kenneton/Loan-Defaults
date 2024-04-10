import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.ticker as ticker
from matplotlib.axes import Axes
import seaborn as sns


import textwrap
from typing import List, Any, Union

from sklearn.metrics import (
    roc_curve,
    auc,
    confusion_matrix,
    precision_recall_curve,
    ConfusionMatrixDisplay,
)

# 1. Colors

my_palette = "muted"
one_scale_color = "Blues"
one_scale_color_r = "Blues_r"

two_colors = [sns.color_palette(my_palette)[0], sns.color_palette(my_palette)[1]]
custom_colors = sns.color_palette(two_colors, 2)


# 2. Text formater and helper functions


def thousands_formatter(x: float, pos: None = None, decimals: int = 0) -> str:
    return f"{x / 1000:.{decimals}f}k"


def millions_formatter(x: float, pos: None = None, decimals: int = 0) -> str:
    return f"{x * 1e-6:.{decimals}f}M"


def to_display_format(s: str) -> str:
    """Converts a string with underscores to a display format, capitalizing each word and separating them with spaces."""
    return " ".join(word.capitalize() for word in s.split("_"))


def wrap_labels(ax: Axes, width: int, break_long_words: bool = False) -> None:
    """
    Wrap x-axis labels in a plot.

    Parameters:
    - ax: The matplotlib axis object to format.
    - width: The maximum line width for wrapping the text.
    - break_long_words: If True, long words will be broken to fit the width.
    """
    labels = []
    for label in ax.get_xticklabels():
        text = label.get_text()
        text = text.lower().replace("_", " ")
        wrapped_label = textwrap.fill(
            text, width=width, break_long_words=break_long_words
        )
        labels.append(wrapped_label)

    ax.set_xticklabels(labels, rotation=0, ha="center")


def format_axis(millions: bool = True, decimals: int = 0) -> None:
    """
    Formats the y-axis of the current plot to display values in millions or thousands, with a specified number of decimal places.

    - millions: If True, format the y-axis values in millions; otherwise, format them in thousands.
    - decimals: The number of decimal places to use in the formatted values.
    """
    if millions == True:
        plt.gca().yaxis.set_major_formatter(
            FuncFormatter(lambda x, pos: millions_formatter(x, pos, decimals=decimals))
        )
    else:
        plt.gca().yaxis.set_major_formatter(
            FuncFormatter(lambda x, pos: thousands_formatter(x, pos, decimals=decimals))
        )


# 3. EDA chart functions


def plot_barchart(
    df: pd.DataFrame,
    order: bool = False,
    percent: bool = True,
    scale: str = "as_is",
    decimals: int = 0,
    ax=None
) -> None:
    """
    Plots a barchart with a low ink-to-data ratio from a dataframe.
    Annotates the values and percentages (optional).
    Formats large values to millions or thousands based on 'scale'.
    """

    if order:
        df = df.sort_values(by=df.columns[1], ascending=False)

    index_ord = df[df.columns[0]]

    if ax is None:  
        g = sns.barplot(
            data=df,
            x=df.columns[0],
            y=df.columns[1],
            palette=one_scale_color_r,
            width=0.6,
            order=index_ord,
        )
    else:
        g = sns.barplot(
            data=df,
            x=df.columns[0],
            y=df.columns[1],
            palette=one_scale_color_r,
            width=0.6,
            order=index_ord,
            ax=ax
        )

    # Update formatter function to handle decimals when scale is 'as_is'
    formatter_func = lambda x: f"{x:.{decimals}f}"
    if scale == "millions":
        formatter_func = lambda x: millions_formatter(x, decimals=decimals)
    elif scale == "thousands":
        formatter_func = lambda x: thousands_formatter(x, decimals=decimals)

    for index, value in enumerate(df[df.columns[1]]):
        annotation = formatter_func(value)
        if percent:
            total = df[df.columns[1]].sum()
            percentage = f"({100 * value / total:.1f}%)"
            annotation = f"{annotation}\n{percentage}"

        if ax is not None:
            ax.text( 
                index,
                value,
                annotation,
                horizontalalignment="center",
                verticalalignment="bottom",
                color="black",
            )
        else:
            plt.text( 
                index,
                value,
                annotation,
                horizontalalignment="center",
                verticalalignment="bottom",
                color="black",
            ) 

    sns.despine(left=True, bottom=True)
    plt.yticks([])  
    g.set_xlabel("")
    g.set_ylabel("")



def plot_hued_histogram(df: pd.DataFrame, col_name: str, hue_col: str = "TARGET",
                           bins: int = 50, alpha: float = 0.5) -> None:
    """
    Plots a histogram for a specified column in a DataFrame, with the data split by hue.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data to plot.
    - col_name (str): The name of the column in `df` to plot.
    - hue_col (str, optional): The name of the column to use for hue (color coding by category). Defaults to "TARGET".
    - bins (int, optional): The number of bins to use for the histogram. Defaults to 50.
    - alpha (float, optional): The opacity level of the histogram bars. Defaults to 0.5.

    The plot is configured with default labels for the x and y axes as empty strings and a legend 
    indicating "Loan Status" with "Repaid" and "Defaulted" categories. The histogram bars are colored 
    according to the hue_order, with a specific palette for two colors.
    """
    
    plt.figure(figsize=(9, 6))
    sns.histplot(data=df, x=col_name, hue=hue_col, bins=bins, alpha=alpha, 
                 hue_order=[True, False], palette=["#ee854a", "#4878d0"])

    plt.ylabel("")
    plt.xlabel("")
    plt.legend(title="Loan Status", loc="upper right", labels=["Repaid", "Defaulted"])



def stacked_barchart(
    df: pd.DataFrame,
    category_col: str,
    target_col: str,
    ax: Axes,
    palette: List[str] = None,
    legend_title: str = None,
    x_label: str = None,
    y_label: str = "",
    table: bool = False,
) -> None:
    """
    Plots a stacked bar chart with annotations showing count and percentage.

    Args:
        df (pd.DataFrame): DataFrame containing the data to plot.
        category_col (str): Name of the column to be used as category (x-axis).
        target_col (str): Name of the column to be used as target (stacked bars).
        ax (Axes): Matplotlib axes object to plot on.
        palette (List[str], optional): List of colors for the bars. Defaults to None.
        legend_title (str, optional): Title for the legend. Defaults to None.
        x_label (str, optional): Label for the x-axis. Defaults to None.
        y_label (str, optional): Label for the y-axis. Defaults to "".
    """

    if palette is None:
        palette = plt.cm.tab10.colors

    # Create crosstab for stacked bar chart
    ct = pd.crosstab(df[category_col], df[target_col])

    # Plot stacked bar chart
    ct.plot(kind="bar", stacked=True, ax=ax, color=palette, alpha=0.5, width=0.6)
    ax.spines[["right", "top"]].set_visible(False)
    ax.set_title(category_col if legend_title is None else legend_title)
    ax.set_xlabel(category_col if x_label is None else x_label)
    ax.set_ylabel(y_label)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

    bar_totals = ct.sum(axis=1)
    grand_total = bar_totals.sum()

    # Annotate the individual segments of the stacked bars
    for p in ax.patches:
        width, height = p.get_width(), p.get_height()
        x, y = p.get_xy()
        total = bar_totals[int(x + width / 2)]
        percentage = f"{height / total:.1%}" if total else ""

        ax.text(
            x + width / 2,
            y + height / 2,
            f"{percentage}",
            ha="center",
            va="center",
            fontsize=10,
        )

    # Annotate total bar values
    for i, total in enumerate(bar_totals):
        percentage_of_grand_total = f"({total / grand_total:.1%})"
        ax.text(
            i,
            total + grand_total * 0.001,  # Slightly above the bar
            f"{int(total)} {percentage_of_grand_total}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    if table:
        ct_percentage = ct.div(ct.sum(axis=1), axis=0) * 100
        ct_percentage.columns = [f"{col}, %" for col in ct_percentage.columns]
        ct_extended = pd.concat([ct, ct_percentage], axis=1)

        print(ct_extended)




def waterfall_chart(increments: List[float], categories: List[str], y_annotate: float = 700000000) -> None:
    """
    Generate a waterfall chart for financial impact analysis.

    Parameters:
    increments (List[float]): A list of incremental values representing financial changes.
    categories (List[str]): Corresponding categories for each financial change.
    y_annotate (float, optional): The y-axis value at which to annotate the bars. Defaults to 700,000,000.

    Returns:
    None
    """
    original_profit = increments[0] + increments[1]
    profit = sum(increments)

    value_add = profit - original_profit
    value_add_perc = round((value_add / original_profit * 100), 2)

    position = 0
    fig, ax = plt.subplots(figsize=(12, 7))

    bar_colors = ["skyblue", "salmon", "skyblue", "salmon"] 
    
    for index, increment in enumerate(increments):
        ax.bar(categories[index], increment, bottom=position,
               color=bar_colors[index], width=0.7)
        position += increment
    
    ax.bar("Profit", profit, color="#FFD700", width=0.7)

    ax.bar("Value Add", -value_add, bottom=profit, color="lightgreen")
    
    formatter = FuncFormatter(lambda y, _: f"{y/1E6:,.0f}M")
    ax.yaxis.set_major_formatter(formatter)
    
    ax.set_ylabel("Amount")
    ax.set_title("Waterfall Financial Impact Analysis")

    y_annotate = 700000000

    for index, increment in enumerate(increments):
        ax.annotate(f"{increment/1E6:,.1f}M",
                    xy=(index, y_annotate),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center")

    ax.annotate(f"{profit/1E6:,.1f}M", xy=(len(categories), y_annotate),
                xytext=(0, 3),
                textcoords="offset points", ha="center")
    
    ax.annotate(f"{value_add/1E6:,.1f}M\n(+{value_add_perc:.2f}%)", xy=(len(categories) + 1, y_annotate),
            xytext=(0, -12),
            textcoords="offset points", ha="center")
    
    ax.grid(linestyle="dotted")
    plt.tight_layout()
    plt.show()



# 4. Machine Learning related plotting funtions


def plot_roc_curve(model: Any, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series], name: str) -> None:
    """
    Plot the Receiver Operating Characteristic (ROC) curve for a given model and dataset.

    Parameters:
    - model: The machine learning model that supports predict_proba method.
    - X: The feature set for which the ROC curve is to be plotted, accepted as either a NumPy array or a Pandas DataFrame.
    - y: The true binary labels, accepted as either a NumPy array or a Pandas Series.
    - name: The name of the model or curve to be displayed in the plot title.

    This function does not return any value but directly shows the ROC curve plot.
    """
    # Predict probabilities for the validation set
    y_pred_proba = model.predict_proba(X)[:, 1]

    fpr, tpr, thresholds = roc_curve(y, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    # Plot the ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(
        fpr, tpr, color="#EE854A", lw=2, label=f"ROC curve (area = {roc_auc:.2f})"
    )
    plt.plot(
        [0, 1], [0, 1], color="#4878d0", lw=2, linestyle="--"
    )  # Diagonal 45 degree line
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{name} Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")


def plot_confusion_matrix(y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series], name: str) -> None:
    """
    Plots a confusion matrix for the given true and predicted labels.

    Parameters:
    - y_true: The true labels of the data.
    - y_pred: The predicted labels by the model.
    - name: A string to specify the model or dataset name to be included in the plot's title.

    This function does not return any value but directly shows the confusion matrix plot.
    """
    cm = confusion_matrix(y_true, y_pred)

    cmd = ConfusionMatrixDisplay(cm)

    cmd.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix ({name})")


def plot_precision_recall_threshold(model: Any, X: np.ndarray, y: np.ndarray) -> None:
    """
    Plots precision and recall as functions of the threshold for a classifier.

    Parameters:
    - model: A fitted classifier with a predict_proba method that returns probability estimates.
    - X: Feature data for making predictions. Expected to be a 2D array.
    - y: True binary labels for X.

    This function does not return a value but displays a matplotlib plot of precision and recall scores across different thresholds.
    """
    y_scores = model.predict_proba(X)[:, 1]

    precision, recall, thresholds = precision_recall_curve(y, y_scores)

    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, precision[:-1], label="Precision", color="#EE854A")
    plt.plot(thresholds, recall[:-1], label="Recall", color="#4878d0")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Precision and Recall for different thresholds")
    plt.legend()



