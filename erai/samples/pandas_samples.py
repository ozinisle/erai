import numpy as np
import pandas as pd

# Note
# 1. You cannot DELETE index of a dataframe. It will aways exist. dataframe.drop(index=true) is not possible
#   If you are trying to create a csv or json file with out index , then do df.to_csv(filename, index=False)


def createSamplePandasDataframe():
    data = np.array([[1, 5.8, 2.8, 4.0, 3.9], [1, 6.0, 2.2, 4.3, 2.7],
                     [1, 6.0, 2.2, 4.3, 2.7], [1, 6.0, 2.2, 4.3, 2.7],
                     [1, 6.0, 2.2, 4.3, 2.7], [1, 6.0, 2.2, 4.3, 2.7]])
    dataset = pd.DataFrame({'Column1': data[:, 0], 'Column2': data[:, 1],
                            'Column3': data[:, 2], 'Column4': data[:, 3],
                            'Column5': data[:, 4]
                            })
    dataset.head()


def changeDataframeColumnName():
    df = pd.DataFrame({'$a': [1, 2], '$b': [10, 20]})
    df.columns = ['a', 'b']
    df


# drop the last row of a dataframe
# ---------------------------------------------
# preparedTrainingDataDF = preparedTrainingDataDF[:-1]
# ---------------------------------------------

# reverse a dataframe
# ---------------------------------------------
# reverseDf = normalDf[::-1]
# ---------------------------------------------

# Drop column from data frame reference
# ---------------------------------------------
# using the inputRawDataDF created above do DATA PREPARATION for the machine learning model which is yet to be created
# drop the seventh column in the array which contains 0 as values for all the

# print('dropping the last column in the created input data as all values in it are 0s')
# inputProcessedDataDF=inputRawDataDF.drop([6],axis=1)
# ---------------------------------------------


# Rename column in pandas dataframe
# ---------------------------------------------
# df.rename(columns={"A": "a", "B": "c"})
# ---------------------------------------------

# get first 5 columns of first ith row of a dataframe
# ---------------------------------------------
# dataframe[dataframe.columns[:5]][i:i+1]
# ---------------------------------------------

# adding columns to numpy array
# ---------------------------------------------
# dummy_cols_data = np.random.random((predicted_values.shape[0],9))
# predicted_values_adj = np.append(predicted_values, dummy_cols_data, 1)
# ---------------------------------------------

# extracting columns from numpy array
# ---------------------------------------------
# predicted_values_orig_scale= predicted_values_orig_scale_withDummies[:, [1,2,3,4,5]]
# ---------------------------------------------

# get unique values from a column
# ---------------------------------------------
# bufferDF.unique()
# ---------------------------------------------

# numpy array to dataframe
# ---------------------------------------------
# dataset = pd.DataFrame({'Column1': numpyArr[:, 0], 'Column2': numpyArr[:, 1]})
# ---------------------------------------------

# convert float to integer : numpy array
# ---------------------------------------------
# a = numpy.array([1, 2, 3, 4], dtype=numpy.float64)
# array([ 1.,  2.,  3.,  4.])
# a.astype(numpy.int64)
# ---------------------------------------------

# create one column based on values from another column
# # ---------------------------------------------
# def label_race (row):
#     if row['attrib'] = 'some value':
#         return 'some value'
#     return 'some other value'

# df.apply (lambda row: label_race(row), axis=1)

# df['race_label'] = df.apply (lambda row: label_race(row), axis=1)
# ---------------------------------------------

# get row index in lambda.apply method
# --------------------------------------------------------
# def test(row):
#    print(type(row.name))
