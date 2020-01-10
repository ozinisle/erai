traceback_template = '''Traceback (most recent call last):
  File "%(filename)s", line %(lineno)s, in %(name)s
%(type)s: %(message)s\n''' # Skipping the "actual line" item

def readRawData(relativeDataFolderPath,outputFileName = "processedRawData.csv"):
    
    import os,sys,traceback
    
    import pandas as pd    
    import glob

    from myUtilities import getParentFolder
    from myUtilities import createFolder

    from fastai.tabular import  add_datepart
    
    # Variable to hold the original source folder path which is calculated from the input relative path of the source folder (relativeDataFolderPath)
    # using various python commands like os.path.abspath and os.path.join
    jupyterNodePath = None
    
    # Variable to hold a dataframe created with the data from input data files in the relativeDataFolderPath provided
    inputRawDataDF = None    

    # Variable to hold the original source folder path which is calculated from the input relative path of the source folder (relativeDataFolderPath)
    # using various python commands like os.path.abspath and os.path.join
    dataFolderPath = None

    # Variable to hold query like value of python to query all json file names in the source folder (dataFolderPath).
    # Will be used in the glob function to execute the query
    json_pattern = None         
    
    # Variable to contain the list of all input json file names in the source folder (dataFolderPath)
    file_list = None        
    
    # return values of this method
    # -------------------------------------------------------------------------------
    # Current methods return value initialized to false. Will be maked as true 
    # after every single line in the method has been executed with out errors
    returnValue = False
    # complete filepath of the csv file with the processed raw data
    output_file_name = None
    
    # -------------------------------------------------------------------------------        
    try:
        #caluclate the deployment directory path of the current juypter node in the operating system
        jupyterNodePath = os.path.abspath(os.path.join('.'))

        # TO BE MODIFIED - NOT SURE WHY I USED THIS - WILL HAVE TO CHECK
        pd.set_option('display.max_columns', None)

        # creating pandas dataframe references for further modification
        inputRawDataDF = pd.DataFrame()        

        #calculating the complete data folder path of the relative path provided as parameter
        dataFolderPath = jupyterNodePath +'/'+relativeDataFolderPath

        # creating OS queryable object for python to work with to find json files in the dataFolderPath calcuated in the previous step
        json_pattern = os.path.join(dataFolderPath,'*.json')

        # store all the json file paths in the dataFolderPath for further processing
        file_list = glob.glob(json_pattern)

        # execution assertion/ui progress update info
        print('looping through all the files to create input data')
        # loop through all the files in the folder and create inputRawDataDF pandas datafram
        for file in file_list:        
            data = pd.read_json(file, lines=True)
            data=data.values[0][0]['candles']
            inputRawDataDF = inputRawDataDF.append(data, ignore_index = True)

        inputRawDataDF.columns = ['date-time', 'open', 'high', 'low', 'close', 'quantity','dont-know']

        buffer = inputRawDataDF['date-time']
        add_datepart(inputRawDataDF, 'date-time')

        inputRawDataDF=pd.concat([buffer,inputRawDataDF], axis=1)

        #create prior_holidays feature
        priorHolidaysStamps=getPriorHoliDaysStamps(inputRawDataDF['date-timeDayofyear'])
        priorHolidaysStamps_df=pd.DataFrame({'prior_holidays': priorHolidaysStamps[:]})

        inputRawDataDF = pd.concat([inputRawDataDF,priorHolidaysStamps_df],axis=1)

        #create following_holidays feature
        followingHolidaysStamps=getFollowingHolidaysDaysStamp(inputRawDataDF['date-timeDayofyear'])
        followingHolidaysStamps_df=pd.DataFrame({'following_holidays': followingHolidaysStamps[:]})

        inputRawDataDF = pd.concat([inputRawDataDF,followingHolidaysStamps_df],axis=1)
        
        '''
        w  write mode
        r  read mode
        a  append mode

        w+  create file if it doesn't exist and open it in (over)write mode
            [it overwrites the file if it already exists]
        r+  open an existing file in read+write mode
        a+  create file if it doesn't exist and open it in append mode
        '''
        
        output_csvdata_path =  getParentFolder(dataFolderPath,2) + '\\processed'
        print('Attempting to create folder if it does not exist >>>'+ output_csvdata_path)
        createFolder(output_csvdata_path)
        
        output_file_name = output_csvdata_path+'/'+outputFileName
        print('Attempting to create/update file >>>'+ output_file_name)
        #f = open(output_file_name, 'w+')  # open file in append mode
        #f.write('')
        #f.close()
        #np.savetxt(output_file_name, inputRawDataDF, delimiter=",")
        inputRawDataDF.to_csv(output_file_name,sep=',', index=False)
        
        print('created raw easy to use csv data to be used for preparing training data in the location  >>>'+output_file_name)
        returnValue = True
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
        return [returnValue, output_file_name, outputFileName,inputRawDataDF]

def getPriorHoliDaysStamps(df):
    import numpy as np
    priorHolidays = np.zeros(df.shape[0])

    count=0
    
    lastValue = 0
    difference = 0
    
    for day in df:    
        currentValue = day

        if count==0:
            lastValue = currentValue             
        

        if currentValue != lastValue:
            difference = currentValue-lastValue 

        if difference == -365:
            difference = 0
        elif difference < 0:
            difference = 365+difference
            
        priorHolidays[count] = (difference-1,0)[difference==0]

        lastValue=currentValue
        count=count+1   

    return priorHolidays.astype(np.int64)

def getFollowingHolidaysDaysStamp(df):
    import numpy as np
    import pandas as pd

    df_reverse = df[::-1]

    followingHoliDays = np.zeros(df_reverse.shape[0])

    count=0
    lastValue = 0
    difference = 0

    for dayCount in df_reverse:    
        currentValue = dayCount

        if count==0:
            lastValue = currentValue 

        if currentValue != lastValue:        
            difference = lastValue-currentValue


        if difference == -365:
            difference = 0
        elif difference < 0:
            difference = 365+difference

        followingHoliDays[count] = (difference-1,0)[difference==0]


        lastValue=currentValue
        count=count+1   

    return followingHoliDays.astype(np.int64)