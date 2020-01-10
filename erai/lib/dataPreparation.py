traceback_template = '''Traceback (most recent call last):
  File "%(filename)s", line %(lineno)s, in %(name)s
%(type)s: %(message)s\n''' # Skipping the "actual line" item

# Note : Configure values as need :: keyword for search - Configuration
# @Param : minLongCandleLength_config - this will vary for different stocks as per market flow history - for crude it can be 5 Rupees, for Adani it can be 1 Rupee
# @Param : relative minimum for open and close values to be considered for doji features relativeOpenCloseValuePercent_config (1% for shares like Adani(1-2 Rs) and 0.01% for crude(.5 to 1 Rs))
def createInputData(processedRawInputDataFilepath, outputFileName='preparedFeaturesWithTrainingData.csv', minLongCandleLength_config=1, relativeOpenCloseValuePercent_config=1):    
    
    import os,sys,traceback    
    from datetime import datetime, timedelta

    import pandas as pd    

    from myUtilities import getParentFolder
    from myUtilities import oneHotEncode_ExternalData

    from dataPreparation import isGreenCandle, isRedCandle, is_red_green_candle, getBigCandleBoundaries, markBigGreenCandle 
    from dataPreparation import markBigRedCandle, getDarkCloudCoverCandles, getBearishBullishNormalCandles, getLongLeggedCandleBoundary, simpleDataPatternFeatures
    from dataPreparation import getRunningDataFeatures,updateRunningAverages

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
    
    try:        
        # read the raw processed data from csv file
        inputRawProcessedDataDF = pd.read_csv(processedRawInputDataFilepath)        

        #initialize the straight forward input features
        preparedTrainingDataDF = pd.DataFrame({            
            'open':inputRawProcessedDataDF['open'],
            'high':inputRawProcessedDataDF['high'],
            'low':inputRawProcessedDataDF['low'],
            'close':inputRawProcessedDataDF['close'],
            'quantity':inputRawProcessedDataDF['quantity']
        })

        print("added INPUT FEATURES >>> 5 count >>> open-high-low-close-quantity")

        # prepare derived input features namely high_low_diff,open_close_dif,high_low_mid,open_close_mid
        # high_low_diff high - low
        # open_close_diff = open - close
        # high_low_mid = (high + low)/2
        # open_close_mid = (open + close)/2
        preparedTrainingDataDF=pd.concat([preparedTrainingDataDF, 
            (preparedTrainingDataDF['high']-preparedTrainingDataDF['low']).rename('high_low_diff'),
            (preparedTrainingDataDF['open']-preparedTrainingDataDF['close']).rename('open_close_diff'),            
            ((preparedTrainingDataDF['high']+preparedTrainingDataDF['low'])/2).rename('high_low_mid'),
            ((preparedTrainingDataDF['open']+preparedTrainingDataDF['close'])/2).rename('open_close_mid'),         
        ], axis=1)

        #preparedTrainingDataDF = preparedTrainingDataDF.rename(columns={0:'high_low_diff',1:'open_close_diff',2:'high_low_mid',3:'open_close_mid'})

        print("added INPUT FEATURES >>> 4 count >>> high_low_diff - open_colse_diff - high_low_mid - open_close_mid")

        # CREATE 7 training features based on 7 weekdays by one hot encoding
        # convert first column with date values in the raw input file into corresponding week days
        print ("attempting to create week_day input feature")
        bufferDF = pd.to_datetime(inputRawProcessedDataDF['date-time'])
        bufferDF = bufferDF.dt.weekday+1
                
        # one hot encode data_weekday_map into the trainig data list, to create additional features
        preparedTrainingDataDF=oneHotEncode_ExternalData(preparedTrainingDataDF, bufferDF,'week_day_')
        print ("added INPUT FEATURES >>> 7 count >>> week_day_1 - ... - week_day_7")

        # CREATE training feature - normal day - does not have a leave prior or latter
        normalDaysDF=inputRawProcessedDataDF.apply (lambda row: getNormalDays(row), axis=1)
        preparedTrainingDataDF=pd.concat([preparedTrainingDataDF, normalDaysDF.rename('normal_days')], axis=1)
        
        #CREATE prior_holidays feature
        print ("attempting to create prior_holidays feature...")
        preparedTrainingDataDF=pd.concat([preparedTrainingDataDF, inputRawProcessedDataDF['prior_holidays']],axis=1)
        print("created >>> prior_holidays")

         #CREATE prior_holidays_count feature
        print ("attempting to create following_holidays feature...")
        preparedTrainingDataDF=pd.concat([preparedTrainingDataDF, inputRawProcessedDataDF['following_holidays']],axis=1)
        print("created >>> following_holidays")
        
        # CREATE training features based on prior_holidays count
        print ("attempting to create one hot encoded month based input feature...")
        preparedTrainingDataDF=oneHotEncode_ExternalData(preparedTrainingDataDF, inputRawProcessedDataDF['prior_holidays'],'prior_holidays_')
        print("added INPUT FEATURES >>> prior_holidays_*")

        # CREATE training features based on following_holidays count
        print ("attempting to create one hot encoded month based input feature...")
        preparedTrainingDataDF=oneHotEncode_ExternalData(preparedTrainingDataDF, inputRawProcessedDataDF['following_holidays'],'following_holidays_')
        print("added INPUT FEATURES >>> following_holidays*")

        # year trend feature
        print ("attempting to create year trend input")
        preparedTrainingDataDF=pd.concat([preparedTrainingDataDF, inputRawProcessedDataDF['date-timeYear'].rename('year_trend')],axis=1)
        print("added INPUT FEATURES >>> year_trend")

        # month trend feature
        print ("attempting to create month trend input")
        preparedTrainingDataDF=pd.concat([preparedTrainingDataDF, inputRawProcessedDataDF['date-timeMonth'].rename('month_trend')],axis=1)
        print("added INPUT FEATURES >>> month_trend")

        # REDUNDANT FEATURES - Removed after testing and finding that these are useless plus bad for the model
        # HAS NEGATIVE IMPACT ON THE MODEL
        # CREATE training features based on date-timeYear column in raw data to its corresponding one hot encoded values
        print ("attempting to create one hot encoded year based input feature")
        preparedTrainingDataDF=oneHotEncode_ExternalData(preparedTrainingDataDF, inputRawProcessedDataDF['date-timeYear'],'year_')
        print("added INPUT FEATURES >>> year_**")

        # # CREATE training features based on date-timeMonth column in raw data to its corresponding one hot encoded values
        print ("attempting to create one hot encoded month based input feature")
        preparedTrainingDataDF=oneHotEncode_ExternalData(preparedTrainingDataDF, inputRawProcessedDataDF['date-timeMonth'],'month_')
        print("added INPUT FEATURES >>> month_**")

        # creating input features namely month_end, month_start, quarter_end, quarter_start, year_end, year_start
        print ("attempting to create features namely month_end, month_start, quarter_end, quarter_start, year_end, year_start")
        bufferDF = pd.DataFrame({            
            'month_end':inputRawProcessedDataDF['date-timeIs_month_end'],
            'month_start':inputRawProcessedDataDF['date-timeIs_month_start'],
            'quarter_end':inputRawProcessedDataDF['date-timeIs_quarter_end'],
            'quarter_start':inputRawProcessedDataDF['date-timeIs_quarter_start'],
            'year_end':inputRawProcessedDataDF['date-timeIs_year_end'],
            'year_start':inputRawProcessedDataDF['date-timeIs_year_start']
        })

        bufferDF['month_end'].loc[bufferDF['month_end']==True] = 1
        bufferDF['month_end'].loc[bufferDF['month_end']==False] = 0
        bufferDF['month_start'].loc[bufferDF['month_start']==True] = 1
        bufferDF['month_start'].loc[bufferDF['month_start']==False] = 0
        bufferDF['quarter_end'].loc[bufferDF['quarter_end']==True] = 1
        bufferDF['quarter_end'].loc[bufferDF['quarter_end']==False] = 0
        bufferDF['quarter_start'].loc[bufferDF['quarter_start']==True] = 1
        bufferDF['quarter_start'].loc[bufferDF['quarter_start']==False] = 0
        bufferDF['year_end'].loc[bufferDF['year_end']==True] = 1
        bufferDF['year_end'].loc[bufferDF['year_end']==False] = 0
        bufferDF['year_start'].loc[bufferDF['year_start']==True] = 1
        bufferDF['year_start'].loc[bufferDF['year_start']==False] = 0

        preparedTrainingDataDF=pd.concat([preparedTrainingDataDF,bufferDF], axis=1) 
        print ("created features >>> month_end, month_start, quarter_end, quarter_start, year_end, year_start")  

        # CREATE training feature - Bearish candle, Bullish candle, Neither bearish nor bullish (normal)candle,  Red candle, green candle, dark cloud cover candles and dark cloud cover confirmed candles
        print("creating bearish, bullish, normal, red, green, darkCloudCover  and darkCloudCoverConfirmed candle features ....")
        bbnCandlesDF = getBearishBullishNormalCandles(inputRawProcessedDataDF)
        preparedTrainingDataDF=pd.concat([preparedTrainingDataDF, bbnCandlesDF], axis=1)
        print('added INPUT FEATURES >>> 7 count ---------------------------------')

        print("creating big_red_candles, very_big_red_candles, big_green_candles, very_big_green_candles ...")
        bigRedCandleBoundary_value,bigGreenCandleBoundary_value = getBigCandleBoundaries(inputRawProcessedDataDF,.4) 
        veryBigRedCandleBoundary_value,veryBigGreenCandleBoundary_value = getBigCandleBoundaries(inputRawProcessedDataDF,.2) 

        print("bigRedCandleBoundary_value,bigGreenCandleBoundary_value  >>> " + str(bigRedCandleBoundary_value)+" >> " +str(bigGreenCandleBoundary_value))
        print("veryBigRedCandleBoundary_value,veryBigGreenCandleBoundary_value  >>> " + str(veryBigRedCandleBoundary_value)+" >> " +str(veryBigGreenCandleBoundary_value))
        
        bigRedCandleDf=inputRawProcessedDataDF.apply(lambda row: markBigRedCandle(row,bigRedCandleBoundary_value,minLongCandleLength_config), axis=1)
        bigGreenCandleDf=inputRawProcessedDataDF.apply(lambda row: markBigGreenCandle(row,bigGreenCandleBoundary_value,minLongCandleLength_config), axis=1)
        veryBigRedCandleDf=inputRawProcessedDataDF.apply(lambda row: markBigRedCandle(row,veryBigRedCandleBoundary_value,minLongCandleLength_config), axis=1)
        veryBigGreenCandleDf=inputRawProcessedDataDF.apply(lambda row: markBigGreenCandle(row,veryBigGreenCandleBoundary_value,minLongCandleLength_config), axis=1)

        preparedTrainingDataDF=pd.concat([preparedTrainingDataDF, 
                    bigRedCandleDf.rename('big_red_candle'),
                    bigGreenCandleDf.rename('big_green_candle'),            
                    veryBigRedCandleDf.rename('very_big_red_candle'),
                    veryBigGreenCandleDf.rename('very_big_green_candle'),         
                ], axis=1)
        print('added INPUT FEATURES >>> 4 count ---------------------------------')

        print("creating simple candle pattern features >>>  Doji ,Long Legged Candle, Long-Legged Doji ,Dragonfly Doji, \n  \
          Gravestone Doji ,Hammer  Inverted Hammer ,Hanging Man  Long Upper Shadow ,  Shooting Star, Long Lower Shadow, \n  \
            Marubozu   Shaven Head ,  Shaven Bottom ,  Spinning Top  ... ")
        _shortLegBoundary=getLongLeggedCandleBoundary(preparedTrainingDataDF)
        preparedTrainingDataDF=preparedTrainingDataDF.apply(lambda row: simpleDataPatternFeatures(row,minLongCandleLength_config,relativeOpenCloseValuePercent_config), axis=1)

        print('added INPUT FEATURES >>> 15 count >>> SIMPLE CANDLE PATTERN FEATURES---------------------------------')         

        print("Creating COMPLEX INPUT FEATURE >>> MOVING AVERAGES ....")
        preparedTrainingDataDF = getRunningDataFeatures(preparedTrainingDataDF,"hourly")
        print("Added COMPLEX INPUT FEATURE >>> MOVING AVERAGES ")

     
        # write the prepared data to create a training dataset file for future ML models
        outputFilePath=getParentFolder(processedRawInputDataFilepath) +'/' + outputFileName
        preparedTrainingDataDF.to_csv(outputFilePath,sep=',', index=False)
        
        print('created raw easy to use csv data to be used for preparing training data in the location  >>>'+outputFileName)
        success=True
    except:
        print("Error executing method >>> ")
        # exc_type, exc_obj, exc_tb = sys.exc_info()
        # fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        # print("Unexpected error:", sys.exc_info())
        # print(exc_type, fname, exc_tb.tb_lineno)
        
        # http://docs.python.org/2/library/sys.html#sys.exc_info
        exc_type, exc_value, exc_traceback = sys.exc_info() # most recent (if any) by default
        
        '''
        Reason this _can_ be bad: If an (unhandled) exception happens AFTER this,
        or if we do not delete the labels on (not much) older versions of Py, the
        reference we created can linger.

        traceback.format_exc/print_exc do this very thing, BUT note this creates a
        temp scope within the function.
        '''

        traceback_details = {
                            'filename': exc_traceback.tb_frame.f_code.co_filename,
                            'lineno'  : exc_traceback.tb_lineno,
                            'name'    : exc_traceback.tb_frame.f_code.co_name,
                            'type'    : exc_type.__name__,
                            'message' : traceback.extract_tb(exc_traceback)
                            }
        
        del(exc_type, exc_value, exc_traceback) # So we don't leave our local labels/objects dangling
        # This still isn't "completely safe", though!
        # "Best (recommended) practice: replace all exc_type, exc_value, exc_traceback
        # with sys.exc_info()[0], sys.exc_info()[1], sys.exc_info()[2]
        
        print
        print(traceback.format_exc())
        print
        print(traceback_template % traceback_details)
        print

        #traceback.print_exception()
        raise

    finally:
        return success, outputFilePath, outputFileName,preparedTrainingDataDF

def getNormalDays (row):
   if row['prior_holidays'] == 0 and row['following_holidays'] == 0:
      return 1      
   return 0

