# BASED ON:
# https://github.com/Nuclearstar/K-Anonymity/blob/master/k-Anonymity.ipynb

import pandas as pd
import matplotlib.pylab as plt
import matplotlib.patches as patches
import sys
import os
import itertools
import numpy as np


def getSpans(df: pd.DataFrame, partition: list, scale: dict = None) -> dict:
    """
This function calculates the span for each column and returns a dictionary containing them.

Function: getSpans
Args:  
  Mandatory
    df (pandas.DataFrame):   Dataframe used for spans.
    partition (list):        Contains the dataframe indexes that is used for the span calculations.
  Optional
    scale (dict):              Divide span with given value.
Returns:
    dict:                    Dictionary where keys are the columns and the values are the span.
"""
    spans = {}
    for column in df.columns:
        if df[column].dtype == 'category':
            # Span is the number of unique values in category
            span: int = df[column][partition].nunique()
        else:
            # Get the mean of highest and lowest value
            span: float = round(df[column][partition].max() - df[column][partition].min(),2)
        if scale is not None:
            span = span / scale[column]
        # Save the columns span in the given partition
        spans[column] = span
    return spans


def split(df: pd.DataFrame, partition: list, column: str) -> tuple:
    """
This function divides and returns the data in two parts, left and right, containing the indexes.

Function: split
Args:  
  Mandatory
    df (pandas.DataFrame):    Dataframe used for splitting.
    partition (list):         Contains the dataframe indexes that is used for the split.
    column (str):             The column name that is split.
Returns:
    tuple:                    Tuple containing the left and right splits
"""
    dfp = df[column][partition]
    if df[column].dtype == 'category':
        values = dfp.unique()
        # First half of the unique values are divided to the left
        left = set(values[:len(values)//2])
        # Second half of the unique values are divided to the right
        right = set(values[len(values)//2:])
        # Return two arrays containing the indexis of the left and right values
        return dfp.index[dfp.isin(left)], dfp.index[dfp.isin(right)]
    median = dfp.median()
    # Everything that is below the median goes to the left
    left = dfp.index[dfp < median]
    # Everything that's equal or greater to the median goes right
    right = dfp.index[dfp >= median]
    return (left, right)


def buildIndexes(df: pd.DataFrame) -> dict:
    """
This function relabels the categorical columns for visualization.

Function: buildIndexes
Args:  
  Mandatory
    df (pandas.DataFrame):      Dataframe where the categorical columns are taken.
Returns:
    dict:                       Dictionary containing old labels as key and new labels as values.
"""
    indexes = {}
    # Create a list containing all categorical columns
    categorical = [column for column in df.columns if df[column].dtype == 'category']
    for column in categorical:
        # Sort columns unique values
        values = sorted(df[column].unique())
        # Key is columns unique values, or old values, and values is the new label.
        indexes[column] = {x: y for x, y in zip(values, range(len(values)))}
    return indexes



#       -----------------------------------------
#     ---------------------------------------------
#    ---------------------    ----------------------
#   -------------------           -------------------
#  -----------------                 -----------------
#  ---------------    Singlefeature   ----------------
#  -----------------                 -----------------
#   -------------------           -------------------
#    ---------------------    ----------------------
#     ---------------------------------------------
#      -------------------------------------------



def aggColumns(series: pd.Series) -> list:
    """
This function aggragates the series to a single value and return them in a list. Used with df.agg() method. Used with single feature anonymisation.

Function: aggColumns
Args:  
  Mandatory
    df (pandas.Series):      Series that is aggragated.
Returns:
    list:                    List containing the aggragated values.
"""
    # Combines the category values to a single value.
    if series.dtype == 'category':
        return [','.join(set(series))]
    # Aggregate the the values to their mean.
    return [series.mean()]


def partitionDataset(df: pd.DataFrame, feature_columns: list, sensitive_column: str, scale: dict, is_valid: callable) -> list:
    """
This function divides the dataframe to smaller partitions and returns a list containing them. Used with single feature anonymisation.

Function: partitionDataset
Args:  
  Mandatory
    df (pandas.DataFrame):      Dataframe that is partitioned
    feature_columns (list):     Columns that are aggragated.
    sensitive_column (str):    Columns that are left as is.
    scale (dict):               Divide span with given value.
    is_valid (callable):        Function used for checking the anonymisation
Returns:
    list:                       List containing the divided dataframe.
"""
    finished_partition = []
    # Set all of the dataframes indexes as list
    partitions = [df.index]
    while partitions:
        partition = partitions.pop(0)
        # Get the spans for the current partition
        spans = getSpans(df[feature_columns], partition, scale)
        # Sort the values in dictionary and iterate through the columns and spans.
        for column, span in sorted(spans.items(), key = lambda x: -x[1]):
            # Split the data in two.
            left, right = split(df, partition, column)
            # Check if the partition is at least k-anonymous, l-diverse and/or t-closeness is high enough.
            if not is_valid(df, left, sensitive_column) or not is_valid(df, right, sensitive_column):
                # Continue if the partition/data is not anonymous enough
                continue
            # Save the partition
            partitions.extend((left, right))
            break
        else:
            finished_partition.append(partition)
    return finished_partition


def buildAnonymizedDataset(df: pd.DataFrame, partitions: list, feature_columns: list, sensitive_column: str, max_partitions: int = None) -> pd.DataFrame:
    """
This function creates anonymized dataframe by aggregating all of the data except the sensitive feature and returns it.

Function: buildAnonymizedDataset
Args:  
  Mandatory
    df (pandas.DataFrame):      Dataframe that is anonymized
    feature_columns (list):     Columns that are aggragated.
    sensitive_column (list:    Columns that are left as is.
    max_partitions (int):       Limit how many partition is handled
Returns:
    pd.DataFrame:               New anonymized dataframe
"""
    rows = []
    for i, partition in enumerate(partitions):
        if i % 100 == 1:
            print(f'Finished {i} partitions..')
        if max_partitions is not None and i > max_partitions:
            break
        # Aggregate column value in partition.
        grouped_columns = df.loc[partition].agg(aggColumns)
        # Group the dataframe on sensitive column based on its count value.
        sensitive_counts = df.loc[partition].groupby(sensitive_column).agg({sensitive_column: 'count'})
        values = grouped_columns.iloc[0].to_dict()
        # Create anonymised data for dataframe and update count value for each row.
        for sensitive_value, count in sensitive_counts[sensitive_column].items():
            if count == 0:
                continue
            values.update({sensitive_column: sensitive_value, 'count': count})
            rows.append(values.copy())
    return pd.DataFrame(rows)


def isKAnonymous(df: pd.DataFrame, partition: list, column: str, k = 3) -> bool:
    """
This function checks if the partition has at least k number of entries. Return True if k-anonymous, else returns False.

Function: isKAnonymous
Args:  
  Mandatory
    df (pandas.DataFrame):    Dataframe used for checking the k-anonymity. NOT USED IN THIS.
    partition (list):     Contains the dataframe indexes that is used for the split.
    column (str):             The column name that is used for checking. NOT USED IN THIS.
Returns:
    bool:                     Return True if k-anonymous else returns False
"""
    return len(partition) >= k


def diversify(df: pd.DataFrame, partition: list, column: str) -> int:
    """
This function calculates the number of unique values and return it.

Function: diversify
Args:  
  Mandatory
    df (pandas.DataFrame):    Dataframe that is used for the l-diversity check.
    partition (list):         List of indexes that are used to select the rows from dataframe for check.
    column (str):              Columns from which the number of unique values are counted from.
Returns:
    int:                      Number of unique values.
"""
    return df[column][partition].nunique()


def isLDiverse(df: pd.DataFrame, partition: list, sensitive_column: str, l = 2) -> bool:
    """
This function checks if the partitioned dataframe is l-diverse.

Function: isLDiverse
Args:  
  Mandatory
    df (pandas.DataFrame):    Dataframe that is used for the l-diversity check.
    partition (list):         List of indexes that are used to select the rows from dataframe for check.
    sensitive_column (str):   Columns from which the number of unique values are counted from.
  Optional
    l (int):                  The minimum number of unique values required for l-diversity.
Returns:
    bool:                     Returns True if returned value is greater than l, else returns False
"""
    return diversify(df, partition, sensitive_column) >= l


def tCloseness(df: pd.DataFrame, partition: list, column: str, global_freqs: dict) -> float:
    """
This function calculates the largest distribution of the values within the partition compared to the whole data and returns the percentage of it.

Function: tCloseness
Args:  
  Mandatory
    df (pandas.DataFrame):    Dataframe that is used for the l-diversity check.
    partition (list):         List of indexes that are used to select the rows from dataframe for check.
    column (str):             Columns from which the number of unique values are counted from.
    global_freqs (dict):      Dictionary containing the unique values and their count. This is equal to value_counts method.
Returns:
    float:                    Returns the highest t-close value in the partition
"""
    # The number of unique values
    total_counts = len(partition)
    d_max = None
    group_counts = df.loc[partition].groupby(column)[column].agg('count')
    # Calculates the portion of the the unique values in partition in relation to the whole data.
    for value, count in group_counts.to_dict().items():
        # Calculates the percentage of occurrence of a count in the partition.
        p = count / total_counts
        # Calculates the absolute difference between the occurrence in the partition and the overall frequency in the dataframe
        d = abs(p - global_freqs[column][value])
        # Save the higher value.
        if d_max is None or d > d_max:
            d_max = d
    return d_max


def isTClose(df: pd.DataFrame, partition: list, sensitive_column: str, global_freqs: dict, p = 0.2) -> bool:
    """
This function checks if the sensitive data is t-close.

Function: isTClose
Args:  
  Mandatory
    df (pandas.DataFrame):    Dataframe that is used for the l-diversity check.
    partition (list):         List of indexes that are used to select the rows from dataframe for check.
    sensitive_column (str):   Columns that is used for the check.
    global_freqs (dict):      Dictionary containing the unique values and their count. This is equal to value_counts method.
    p (float):                The t-close value that the column must remain under.
Returns:
    bool:                     Returns True if the highest t-close value is at or under the given t-close value. Else returns False.
"""
    if df[sensitive_column].dtype != 'category':
        raise ValueError('this method only works for categorical values')
    return tCloseness(df, partition, sensitive_column, global_freqs) <= p



#       -----------------------------------------
#     ---------------------------------------------
#    ---------------------    ----------------------
#   -------------------           -------------------
#  -----------------                 -----------------
#  ---------------    Multifeatures   ----------------
#  -----------------                 -----------------
#   -------------------           -------------------
#    ---------------------    ----------------------
#     ---------------------------------------------
#      -------------------------------------------



def diversifyMulti(df: pd.DataFrame, partition: list, column: str) -> int:
    """
This function calculates the number of unique values and return it.

Function: diversifyMulti
Args:  
  Mandatory
    df (pandas.DataFrame):    Dataframe that is used for the l-diversity check.
    partition (list):         List of indexes that are used to select the rows from dataframe for check.
    column (str):              Columns from which the number of unique values are counted from.
Returns:
    int:                      Number of unique values.
"""
    return df.iloc[partition][column].nunique()


def isLDiverseMulti(df: pd.DataFrame, partition: list, sensitive_column: str, l = 2) -> bool:
    """
This function checks if the partitioned dataframe is l-diverse.

Function: isLDiverseMulti
Args:  
  Mandatory
    df (pandas.DataFrame):    Dataframe that is used for the l-diversity check.
    partition (list):         List of indexes that are used to select the rows from dataframe for check.
    sensitive_column (str):   Columns from which the number of unique values are counted from.
  Optional
    l (int):                  The minimum number of unique values required for l-diversity.
Returns:
    bool:                     Returns True if returned value is greater than l, else returns False
"""
    return diversifyMulti(df, partition, sensitive_column) >= l
    
def aggColumnsMulti(series: pd.Series):
    """
This function aggragates the series to a single value and return them in a list. Used with df.agg() method.

Function: aggColumnsMulti
Args:  
  Mandatory
    df (pandas.Series):      Series that is aggragated.
Returns:
    list:                    List containing the aggragated values.
"""
    result = []
    if series.dtype == 'category':
        # Combines the category values to a single value.
        result.append(','.join(set(series)))
    else:
        # Aggregate the the values to their mean.
        result.append(series.mean())
    return result


def buildAnonymizedDatasetMulti(df: pd.DataFrame, partitions: list, feature_columns: list, sensitive_columns: list, max_partitions: int = None) -> pd.DataFrame:
    """
This function creates anonymized dataframe and returns it.

Function: buildAnonymizedDatasetMulti
Args:  
  Mandatory
    df (pandas.DataFrame):      Dataframe that is anonymized
    feature_columns (list):     Columns that are aggragated.
    sensitive_columns (list):    Columns that are left as is.
    max_partitions (int):       Limit how many partition is handled
Returns:
    pd.DataFrame:               New anonymized dataframe
"""
    new_df = pd.DataFrame()
    for i, partition in enumerate(partitions):
        if i % 100 == 1:
            print(f'Finished {i} partitions..')
        # Check if max_partitions is given and it's equal to or less than the partition index.
        if max_partitions is not None and i > max_partitions:
            break
        # Aggregate column values in the partition.
        grouped_columns = df.loc[partition][feature_columns].agg(aggColumnsMulti)
        # Group the DataFrame on sensitive columns based on their count values.
        sensitive_counts = df.loc[partition].groupby(sensitive_columns).agg({
            col: 'count' for col in sensitive_columns
        })
        # Prepare the data.
        grouped_values = grouped_columns.to_numpy()
        sensitive_values = sensitive_counts.to_numpy()
        columns = list(grouped_columns.columns) + list(sensitive_counts.columns)
        # Create anonymised data for dataframe and update count value for each row.
        for values in grouped_values:
            # Repeat each row of grouped_values for each row in sensitive_values
            repeated_values = np.repeat([values], len(sensitive_values), axis=0)
            # Combine the data
            combined_data = np.column_stack((repeated_values, list(sensitive_counts.index)))
            # Create a new DataFrame with the repeated_values and sensitive_values
            temp_df = pd.DataFrame(data = combined_data, columns = columns)
            # Combine temp with new
            new_df = pd.concat([new_df, temp_df])

    return new_df


def partitionDatasetMulti(df: pd.DataFrame, feature_columns: list, sensitive_columns: list, scale, is_valid: callable) -> list:
    """
This function divides the dataframe to smaller partitions and returns a list containing them.

Function: partitionDatasetMulti
Args:  
  Mandatory
    df (pandas.DataFrame):      Dataframe that is partitioned
    feature_columns (list):     Columns that are aggragated.
    sensitive_columns (list):    Columns that are left as is.
    scale (dict):               Divide span with given value.
    is_valid (callable):        Function used for checking the anonymisation
Returns:
    list:                       List containing the divided dataframe.
"""
    finished_partition = []
    # Set the all of the datas indexes as list
    partitions = [df.index]
    while partitions:
        partition = partitions.pop(0)
        # Get the spans for the current partition
        spans = getSpans(df[feature_columns], partition, scale)
        # Sort the values in dictionary and iterate through the columns and spans.
        for column, span in sorted(spans.items(), key = lambda x: -x[1]):
            # Split the data in two.
            left, right = split(df, partition, column)
            # Check if the partition is at least k-anonymous, l-diverse and/or t-closeness is high enough.
            # This is a mess..
            try:
                if not (is_valid(df, left, sensitive_columns).all() and is_valid(df, right, sensitive_columns).all()):
                    # Continue if the partition/data is not anonymous enough
                    continue
            except (AttributeError, ValueError):
                try:
                    if not (is_valid(df, left, sensitive_columns) and is_valid(df, right, sensitive_columns)):
                        # Continue if the partition/data is not anonymous enough
                        continue
                except:
                    if not (is_valid(df, left, sensitive_columns).sum() == len(sensitive_columns) and is_valid(df, right, sensitive_columns)):
                    # Continue if the partition/data is not anonymous enough
                        continue
            # Save the partition
            partitions.extend((left, right))
            break
        else:
            finished_partition.append(partition)
    return finished_partition


def tClosenessMulti(df: pd.DataFrame, partition: list, sensitive_columns: list, global_freqs: dict) -> dict:
    """
This function calculates the largest distribution of the values within the partition compared to the whole data and returns the percentage of it.

Function: tClosenessMulti
Args:  
  Mandatory
    df (pandas.DataFrame):      Dataframe that is used for the l-diversity check.
    partition (list):           List of indexes that are used to select the rows from dataframe for check.
    sensitive_columns (list):   Columns from which the number of unique values are counted from.
    global_freqs (dict):        Dictionary containing the unique values and their count. This is equal to value_counts method.
Returns:
    dict:                       Returns a dictionary containing highest t-close value for each column.
"""
    total_counts = len(partition)
    # Get uniqie values and their count for each sensitive feature.
    group_counts = {column: df.iloc[partition][column].value_counts() for column in sensitive_columns}
    highest_counts = {}
    # Calculate how the values are distributed in the partition.
    for column in group_counts.keys():
        count_distribution = group_counts[column] / total_counts
        highest_counts[column] = count_distribution.max()

    return highest_counts


def isTCloseMulti(df: pd.DataFrame, partition: list, sensitive_columns: list, global_freqs: dict, p = 0.2) -> bool:
    """
This function checks if the sensitive data is t-close.

Function: isTCloseMulti
Args:  
  Mandatory
    df (pandas.DataFrame):     Dataframe that is used for the l-diversity check.
    partition (list):          List of indexes that are used to select the rows from dataframe for check.
    sensitive_columns (list):  Columns that is used for the check.
    global_freqs (dict):       Dictionary containing the unique values and their count. This is equal to value_counts method.
    p (float):                 The highest acceptable t-close value that all of the column must remain under.
Returns:
    bool:                     Returns True if the highest t-close value is at or under the given t-close value. Else returns False.
"""
    # Get the c-closeness value for each sensitive feture.
    t_close_values = np.array(list(tClosenessMulti(df, partition, sensitive_columns, global_freqs).values()))
    return (t_close_values < p).all()



#       -----------------------------------------
#     ---------------------------------------------
#    ---------------------    ----------------------
#   -------------------           -------------------
#  -----------------                 -----------------
#  ---------------    Visualization   ----------------
#  -----------------                 -----------------
#   -------------------           -------------------
#    ---------------------    ----------------------
#     ---------------------------------------------
#      -------------------------------------------



def getCoords(df: pd.DataFrame, column: str, partition: list, indexes: dict, offset = 0.1) -> float:
    """
This function calculates the rectangle coordinates for plotting and returns float containing the rectangle edge coordinates in an axis.

Function: getCoords
Args:  
  Mandatory
    df (pandas.DataFrame):   Dataframe that is used for the calculations.
    column (str):            Column that is one of the axis.
    partition (list):        The rectangle that are being calculated
    indexes (dict):          List of indexes that is used for the coordinate calculations.
  Optional
    offset (float):          A value that is used to avoid overlapping rectangles.
Returns:
    float:                   Returns tuple containing the rectangle coordinates in an axis.
"""
    # Sort the values in partition.
    sorted_values = df[column][partition].sort_values()
    # Get the coordinates in regards the plot.
    if df[column].dtype == 'category':
        # Get two categories at the opposite sides.
        left = indexes[column][sorted_values[sorted_values.index[0]]]
        right = indexes[column][sorted_values[sorted_values.index[-1]]] + 1.0
    else:
        # Get the lowest value in partition and the lowest value outside of partition that is higher than the highest in partition.
        next_value = sorted_values[sorted_values.index[-1]]
        larger_values = df[df[column] > next_value][column]
        if len(larger_values) > 0:
            next_value = larger_values.min()
        left = sorted_values[sorted_values.index[0]]
        right = next_value
    # Adjust the coordinates to not get overlaps.
    left -= offset
    right += offset
    return round(left, 2), round(right, 2)


def getPartitionRects(df: pd.DataFrame, partitions: list, column_x: str, column_y: str, indexes: dict, offsets = [0.1, 0.1]) -> list:
    """
This function calculates the rectangles coordinates for plotting and returns a list containing all of the rectangles.

Function: getPartitionRects
Args:  
  Mandatory
    df (pandas.DataFrame):   Dataframe that is used for the rectangle calculations.
    partition (list):        The rectangles that are being calculated
    column_x (str):          Column that is on the x-axis.
    column_y (str):          Column that is on the y-axis.
    indexes (dict):          List of indexes that is used for the coordinate calculations.
  Optional
    offset (float):          A value that is used to avoid overlapping rectangles.
Returns:
    list:                   Returns a list of rectangles for plotting.
"""
    rects = []
    # Get the rectangles corner coordinates for each partition.
    for partition in partitions:
        x_left, x_right = getCoords(df, column_x, partition, indexes, offsets[0])
        y_left, y_right = getCoords(df, column_y, partition, indexes, offsets[1])
        rects.append(((x_left, y_left), (x_right, y_right)))
    return rects


def getBounds(df: pd.DataFrame, column: str, indexes: dict, offset = 1.0) -> float:
    """
This function calculates the size of the axis in the plot.

Function: getBounds
Args:  
  Mandatory
    df (pandas.DataFrame):   Dataframe that is used for the axis.
    column (str):            Column that is on either x-axis or the y-axis.
    indexes (dict):          List of indexes that is used for determining the size of the plot.
  Optional
    offset (float):          A value that is used to avoid overlapping rectangles.
Returns:
    float:                   Returns the lowest value and the highest value in the axis of the plot.
"""
    # Returns the lowest and the highest value of the axis.
    if df[column].dtype == 'category':
        return 0-offset, len(indexes[column])+offset
    return df[column].min()-offset, df[column].max()+offset


def plotRects(df: pd.DataFrame, ax: plt.Axes, rects: list, column_x: str, column_y: str, indexes: dict, edgecolor = 'black', facecolor = 'none') -> None:
    """
This function plots rectangles that represent the anonymised data, where each rectangle is an anonymised sub group.

Function: plotRects
Args:  
  Mandatory
    df (pandas.DataFrame):   Dataframe that is used for the plot.
    ax (plt.Axes):           The subplot in which the tectangles are plotted into.
    rects (list):            List of the renctangles.
    column_x (str):          Column that is on the x-axis.
    column_y (str):          Column that is on the y-axis.
    indexes (dict):          List of indexes that is used for the coordinate calculations.
  Optional
    edgecolor (str):         The color of the renctangles edges.
    facecolor (str):         The color of the rectangles.
Returns:
    None:
"""
    for (x_left, y_left), (x_right, y_right) in rects:
        ax.add_patch(patches.Rectangle((x_left, y_left), x_right - x_left, y_right - y_left, linewidth = 1, edgecolor = edgecolor, facecolor = facecolor, alpha = 0.5))
    ax.set_xlim(*getBounds(df, column_x, indexes))
    ax.set_ylim(*getBounds(df, column_y, indexes))
    ax.set_xlabel(column_x)
    ax.set_ylabel(column_y)