traceback_template = '''Traceback (most recent call last):
  File "%(filename)s", line %(lineno)s, in %(name)s
%(type)s: %(message)s\n'''  # Skipping the "actual line" item

def updateRunningAverages(row,current_df,featureList,key):  
    import numpy as np

    rowIndex = row.name
    extract=current_df.iloc[rowIndex:rowIndex+featureList[key]]
    row[key+"_open_avg"]=np.mean(extract["open"])
    row[key+"_high_avg"]=np.mean(extract["high"])
    row[key+"_low_avg"]=np.mean(extract["low"])
    row[key+"_close_avg"]=np.mean(extract["close"])
    return row

def getRunningDataFeatures(df,dataCategory="hourly"):
    import os
    import sys
    import traceback
    import numpy as np
    import pandas as pd
    import constants

    try:
        df_reverse=df[::-1]
        df_reverse=df_reverse.reset_index(drop=True)
        featureList = constants.DATA_BATCH[dataCategory]

        # for key in featureList.keys():
        #     print(DATA_BATCH[dataCategory][key])

        for key in featureList.keys():
            df_reverse = df_reverse.apply (updateRunningAverages, axis=1, args=[df_reverse,featureList,key])
    
        df=df_reverse[::-1]
        
        return df.reset_index(drop=True)

    except:
        print("Error executing method >>> ")
        # exc_type, exc_obj, exc_tb = sys.exc_info()
        # fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        # print("Unexpected error:", sys.exc_info())
        # print(exc_type, fname, exc_tb.tb_lineno)

        # http://docs.python.org/2/library/sys.html#sys.exc_info
        # most recent (if any) by default
        exc_type, exc_value, exc_traceback = sys.exc_info()

        '''
        Reason this _can_ be bad: If an (unhandled) exception happens AFTER this,
        or if we do not delete the labels on (not much) older versions of Py, the
        reference we created can linger.

        traceback.format_exc/print_exc do this very thing, BUT note this creates a
        temp scope within the function.
        '''

        traceback_details = {
            'filename': exc_traceback.tb_frame.f_code.co_filename,
            'lineno': exc_traceback.tb_lineno,
            'name': exc_traceback.tb_frame.f_code.co_name,
            'type': exc_type.__name__,
            'message': traceback.extract_tb(exc_traceback)
        }

        # So we don't leave our local labels/objects dangling
        del(exc_type, exc_value, exc_traceback)
        # This still isn't "completely safe", though!
        # "Best (recommended) practice: replace all exc_type, exc_value, exc_traceback
        # with sys.exc_info()[0], sys.exc_info()[1], sys.exc_info()[2]

        print
        print(traceback.format_exc())
        print
        print(traceback_template % traceback_details)
        print

        # traceback.print_exception()
        raise
