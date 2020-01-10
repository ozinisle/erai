import math

m = trainingData.shape[0]
m_train = math.ceil(m*6/10)

#get data upto the m_train'th training example
training_data_portion = trainingData[:m_train]

#allocate the remaining data after the m_train'th training example to validation protion
validation_portion =  trainingData[m_train:]

#get the total number of examples in the validation portion
m_validation = validation_portion.shape[0]

# assign the first half of the data in validation portion to cross_validation portion
m_cv = math.ceil(m_validation/2)
cv_data_portion = validation_portion[:m_cv]

# assign the other half of the data in the validation portion as test_data
test_data_portion = validation_portion[m_cv:]
m_test = test_data_portion.shape[0]

# check if the original training set size is same as the cummulative of training, cv and test data sets
m, m_train+m_cv+m_test

# configure input_training_data and output_training_data
x_train_unscaled = training_data_portion[training_data_portion.columns[:-5]].values
y_train_unscaled = training_data_portion[training_data_portion.columns[-5:]].values

x_train_unscaled.shape,y_train_unscaled.shape

# configure cross validation input and output
cv_data_portion=cv_data_portion.reset_index(drop=True)
x_cv_unscaled = cv_data_portion[cv_data_portion.columns[:-5]].values
y_cv_unscaled = cv_data_portion[cv_data_portion.columns[-5:]].values

x_cv_unscaled.shape,y_cv_unscaled.shape

#configure test data input and output
test_data_portion=test_data_portion.reset_index(drop=True)
x_test_unscaled = test_data_portion[test_data_portion.columns[:-5]].values
y_test_unscaled = test_data_portion[test_data_portion.columns[-5:]].values

x_test_unscaled.shape,y_test_unscaled.shape

# defining Hyper parameters
batch_size = 512
epochs = 8

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

min_max_scaler = MinMaxScaler()

x_train = min_max_scaler.fit_transform(x_train_unscaled)
x_cv = min_max_scaler.fit_transform(x_cv_unscaled)
x_test = min_max_scaler.fit_transform(x_test)

y_train = min_max_scaler.fit_transform(y_train_unscaled)
y_cv = min_max_scaler.fit_transform(y_cv_unscaled)
y_test = min_max_scaler.fit_transform(y_test)





def doBasicOperation():
    import os,sys,traceback    
    from datetime import datetime, timedelta

    import pandas as pd  
    import numpy as np

    
    from dataPreprocessing.basicRawDataProcess import getAutoConfigData
    from dataPreprocessing.basicRawDataProcess import setAutoConfigData

    
    from utilities.fileFolderManipulations import getJupyterRootDirectory
    from utilities.fileFolderManipulations import getParentFolder
    from utilities.fileFolderManipulations import createFolder

    # Variable to hold the original source folder path which is calculated from the input relative path of the source folder (relativeDataFolderPath)
    # using various python commands like os.path.abspath and os.path.join
    jupyterNodePath = None

    configFilePath = None

    # @Return Type
    success = False # help mark successful execution of function 
    outputFilePath = None

    # holds data from input data file - Truth source, should be usd only for reference and no updates should happen to this variable
    inputRawProcessedDataDF = None
    # Variable to hold a dataframe created with the data from input data files 
    # Will be used for data preparation
    preparedTrainingDataDF = None

    # declaring other variables necessary for further manipulations
    bufferDF = None;  #for interim data frame manipulations

    #caluclate the deployment directory path of the current juypter node in the operating system
    jupyterNodePath = getJupyterRootDirectory()

    configFilePath=jupyterNodePath+autoConfigFileRelativePath

    autoConfigData = getAutoConfigData(configFilePath)

    preProcessedDataFilePath=autoConfigData[dataName][dataFrequency][KEY_preProcessedDataFilePath]

    # read the raw processed data from csv file
    inputRawProcessedDataDF = pd.read_csv(preProcessedDataFilePath)  

    basicDf = createFundamentalFeatures(inputRawProcessedDataDF)
    
    return basicDf,variation_degree,preProcessedDataFilePath,autoConfigData,configFilePath
    

def doFeatureAssessment(newFeatureDf,basicDf,variation_degree,preProcessedDataFilePath,autoConfigData,configFilePath):
    import numpy as np
    from utilities.fileFolderManipulations import getJupyterRootDirectory
    from utilities.fileFolderManipulations import getParentFolder
    from utilities.fileFolderManipulations import createFolder
    
    from dataPreprocessing.basicRawDataProcess import setAutoConfigData
    
    
    featureOfInterest = newFeatureDf.name
    newTrainingSetDf = createNewTrainingSetWithFeatureVariations(basicDf,newFeatureDf,featureOfInterest,variation_degree) 

    correlation = newTrainingSetDf.corr()

    reasonableCorelation = correlation.loc[ (np.abs(correlation['open'])>requiredMinimumCorrelation) | 
     (np.abs(correlation['high'])>requiredMinimumCorrelation) |
     (np.abs(correlation['low'])>requiredMinimumCorrelation) | 
     (np.abs(correlation['close'])>requiredMinimumCorrelation)]

    # create necessary file folder structure for storing and filtering features
    outputFolderPath = getParentFolder(preProcessedDataFilePath)

    featuresFolder = outputFolderPath+"\\features"
    createFolder(featuresFolder)

    featureCreationHistoryFolder = featuresFolder+"\\featureHistory"
    createFolder(featureCreationHistoryFolder)

    correlationsFolder = featuresFolder+"\\correlations"
    createFolder(correlationsFolder)

    trainableFeaturesListtFilePath = featuresFolder+"\\"+"trainableFeaturesList.csv"

    currentFeatureListFilePath = featureCreationHistoryFolder+"\\"+featureIndexStamp+featureOfInterest+"_variations_list.csv"
    currentFeatureCorrelationListFilePath = correlationsFolder+"\\"+featureIndexStamp+featureOfInterest+"_variations_correlation_list.csv"
    reasonableCorelationListFilePath = correlationsFolder+"\\"+featureIndexStamp+featureOfInterest+"_variations_reasonable_correlation_list.csv"

    # store output information related to current 
    newTrainingSetDf.to_csv(currentFeatureListFilePath, sep=',', index=False)
    correlation.to_csv(currentFeatureCorrelationListFilePath, sep=',', index=True)
    reasonableCorelation.to_csv(reasonableCorelationListFilePath, sep=',', index=True)
        
    # store trainable features in global file - to be used by other training feature creation procedures    
    trainableFeaturesDf = newTrainingSetDf[[filteredIndex for filteredIndex in reasonableCorelation.index] ]
    trainableFeaturesDf.to_csv(trainableFeaturesListtFilePath, sep=',', index=False)

    # assertions
    print("newTrainingSetDf shape>>>"+str(newTrainingSetDf.shape[0])+","+str(newTrainingSetDf.shape[1]))
    print("trainableFeaturesDf shape>>>"+str(trainableFeaturesDf.shape[0])+","+str(trainableFeaturesDf.shape[1]))
    
    autoConfigData[dataName][dataFrequency].update({'trainableFeaturesListtFile':trainableFeaturesListtFilePath})
    setAutoConfigData(configFilePath,autoConfigData)


    return correlation, reasonableCorelation ,newTrainingSetDf ,trainableFeaturesDf
    
def createFundamentalFeatures(rawDf):    
    import pandas as pd

    #initialize the straight forward input features
    df = pd.DataFrame({            
        'open':rawDf['open'],
        'high':rawDf['high'],
        'low':rawDf['low'],
        'close':rawDf['close']    
    })

    print("added INPUT FEATURES >>> 4 count >>> open-high-low-close")

    
    
    return df

def getFeatureVariations(row, featureName, variation_degree_count):
    import numpy as np
    
    featureVal=row[featureName]    
    
    for iterator in range(1,variation_degree_count+1):
        row[featureName+'_exp_1'] = np.exp(featureVal)
        row[featureName+'_exp_inv_1'] = np.exp(-1*featureVal)
    
        if iterator>1:
            val= np.power(featureVal,iterator)
            valInv = 0
            if not val==0:
                valInv = 1/val
                row[featureName+'_times_inv_'+str(iterator)] = 1/(val*iterator)
            else:
                row[featureName+'_times_inv_'+str(iterator)] = 0

            row[featureName+'_pow_'+str(iterator)] = val
            row[featureName+'_pow_inv_'+str(iterator)] = valInv
            row[featureName+'_exp_'+str(iterator)] = np.exp(iterator*featureVal)
            row[featureName+'_exp_inv_'+str(iterator)] = np.exp(-iterator*featureVal)

            row[featureName+'_times_'+str(iterator)] = val*iterator        

            if val>0:
                row[featureName+'_log_times_'+str(iterator)] = np.log(val*iterator)
            elif val<0:
                 row[featureName+'_log_times_'+str(iterator)] = -np.log(-1*val*iterator)
        
    return row 

def createNewTrainingSetWithFeatureVariations(basicDf,newFeatureDf,featureOfInterest,variation_degree) :
    import pandas as pd
    import numpy as np
    from tqdm import tqdm

    # Create and register a new `tqdm` instance with `pandas`
    # (can use tqdm_gui, optional kwargs, etc.)
    tqdm.pandas()



    newTrainingSetDf = pd.concat([basicDf,newFeatureDf],axis=1)

    newTrainingSetDf = newTrainingSetDf.progress_apply(lambda row,
                    featureName, variation_degree_count:getFeatureVariations(row,featureOfInterest,variation_degree),axis=1,
                    args=[featureOfInterest,variation_degree])

    return newTrainingSetDf

