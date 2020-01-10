traceback_template = '''Traceback (most recent call last):
  File "%(filename)s", line %(lineno)s, in %(name)s
%(type)s: %(message)s\n'''  # Skipping the "actual line" item


def isGreenCandle(data):
    flag = False
    if data['open'] is not None and data['close'] is not None and data['close']-data['open'] > 0:
        flag = True
    return flag


def isRedCandle(data):
    flag = False
    if data['open'] is not None and data['close'] is not None and data['close']-data['open'] < 0:
        flag = True
    return flag

def is_red_green_candle(open,close):
    green,red=(close-open) > 0 , (close-open) < 0 
    return green,red
    
def getBigCandleBoundaries(df,bigCandlesPercentage = .4) :
    candlesByBodyLengthDf=(df['close']-df['open']).rename('close_open_diff')

    candlesByBodyLengthDf=candlesByBodyLengthDf.sort_values(axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last')
    sortedRedCandles=candlesByBodyLengthDf.loc[candlesByBodyLengthDf[0:]<0].reset_index(drop=True)
    sortedGreenCandles=candlesByBodyLengthDf.loc[candlesByBodyLengthDf[0:]>0].reset_index(drop=True)

    bigRedCandleBoundary=int(sortedRedCandles[0:].shape[0] * (bigCandlesPercentage)) # boundary has to calculated from first 20 %
    bigGreenCandleBoundary=int(sortedGreenCandles[0:].shape[0] * (1-bigCandlesPercentage)) # boundary has to calculated from last 20%

    bigRedCandleBoundary_value =  sortedRedCandles[bigRedCandleBoundary]
    bigGreenCandleBoundary_value = sortedGreenCandles[bigGreenCandleBoundary]

    return bigRedCandleBoundary_value,bigGreenCandleBoundary_value

def markBigGreenCandle(row,boundary,defaultConfiguration):    
    if(row['close']-row['open'] > 0):
        if (row['high']-row['low'] > boundary) or (row['high']-row['low'] > 1):
            return 1
    return 0

def markBigRedCandle(row,boundary,defaultConfiguration):   
    if(row['close']-row['open'] < 0): 
        if (row['low']-row['high'] < boundary) or (row['high']-row['low'] > 1):
            return 1
    return 0

def getDarkCloudCoverCandles(currentEntry=None, previousEntry=None, priorToPreviousEntry=None):
    # Dark Cloud Cover is a bearish reversal candlestick pattern where a down candle (typically black or red)
    # opens above the close of the prior up candle (typically white or green),
    # and then closes below the midpoint of the up candle.
    # https://www.investopedia.com/terms/d/darkcloud.asp
    import os
    import sys
    import traceback
    import numpy as np
    import pandas as pd

    isDarkCloudCover = 0
    isDarkCloudCoverConfirmed = 0
    try:
        if currentEntry is None:
            return isDarkCloudCover, isDarkCloudCoverConfirmed

        if previousEntry is None:
            return isDarkCloudCover, isDarkCloudCoverConfirmed
        
        if isRedCandle(currentEntry) and isGreenCandle(previousEntry) :           
            if currentEntry['open']>previousEntry['close']:               
                if currentEntry['close'] <= (previousEntry['close']+previousEntry['open'])/2:                    
                    isDarkCloudCover= 1

        if priorToPreviousEntry is not None:
            if isRedCandle(previousEntry) and isGreenCandle(priorToPreviousEntry) :
                if previousEntry['open']>priorToPreviousEntry['close']:
                    if previousEntry['close']<= (priorToPreviousEntry['close']+priorToPreviousEntry['open'])/2:
                        if currentEntry['close'] < previousEntry['close']:
                            isDarkCloudCoverConfirmed = 1
        
        return isDarkCloudCover,isDarkCloudCoverConfirmed

    except:
        print("Error executing method >>> \n")
        
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

        # traceback.print_exception()
        raise

def getBearishBullishNormalCandles(df):
    import os,sys,traceback 
    import numpy as np
    import pandas as pd 

    try:  
    
        data_count=df.shape[0]
        
        bullishCandles = np.zeros((data_count))
        bearishCandles = np.zeros((data_count))
        normalCandles = np.zeros((data_count))
        greenCandles = np.zeros((data_count))
        redCandles = np.zeros((data_count))
        quantityDifference = np.zeros((data_count))

        darkCloudCover_col = np.zeros((data_count))
        darkCloudCover_confirmed_col = np.zeros((data_count))
        
        # df.loc[0, 'C'] = df.loc[0, 'D']
        for dataItr in range(0, data_count):
            neitherBearishNorBullish = True
            
            currentEntry = df.loc[dataItr]
            previousEntry = None
            priorToPreviousEntry = None

            prev_open= 0
            curr_open= 0
                
            prev_close= 0
            curr_close= 0
            
            _isRedCandle = False

            if dataItr==0 :
                quantityDifference[dataItr] = 0
                normalCandles[dataItr] = 1            
            else:
                

                previousEntry = df.loc[dataItr-1]
                    
                quantityDifference[dataItr] = currentEntry['quantity'] - previousEntry['quantity']

                prev_open= previousEntry['open']
                curr_open= currentEntry['open']
                
                prev_close= previousEntry['close']
                curr_close= currentEntry['close']
                
                if isGreenCandle(currentEntry) :
                    greenCandles[dataItr] = 1
                elif isRedCandle(currentEntry):
                    _isRedCandle = True
                    redCandles[dataItr] = 1
            
                if curr_open>=prev_close and curr_close<prev_open :
                    bullishCandles[dataItr] = 1
                    neitherBearishNorBullish = False                
                    
                if curr_close > prev_open and curr_open >= prev_close :
                    bearishCandles[dataItr] = 1
                    neitherBearishNorBullish = False
                                   
                if neitherBearishNorBullish:
                    normalCandles[dataItr] = 1
                    
            if dataItr>1:
                priorToPreviousEntry = df.loc[dataItr-2]

            if _isRedCandle:
                # can happen only if this is a red candle
                darkCloudCover_entry,darkCloudCover_confirmed_entry = getDarkCloudCoverCandles(currentEntry, previousEntry, priorToPreviousEntry)
                darkCloudCover_col[dataItr] = darkCloudCover_entry
                darkCloudCover_confirmed_col[dataItr] = darkCloudCover_confirmed_entry

        quantityDifference[0] = np.mean(quantityDifference)  # update the mean value to the very first row to assume a more reasonable value for training rather than having a 0 value

        bbnCandlesDf = pd.DataFrame({ 'is_green_candle': greenCandles.astype(np.int64),'is_red_candle': redCandles.astype(np.int64),
                            'is_bearish': bearishCandles.astype(np.int64), 'is_bullish': bullishCandles.astype(np.int64),'is_neither_bearish_nor_bullish': normalCandles.astype(np.int64),
                            'is_darkCloudCover': darkCloudCover_col.astype(np.int64), 'is_darkCloudCover_confirmed':darkCloudCover_confirmed_col.astype(np.int64),
                            'quantityDifference':quantityDifference.astype(np.int64)
        })
        
        return bbnCandlesDf

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

        # traceback.print_exception()
        raise

def getLongLeggedCandleBoundary(df,longLeggedCandlePercentage = .5) :
    candlesByshadowLengthDf=(df['high']-df['low'])    
    return candlesByshadowLengthDf.mean(axis=0)

def simpleDataPatternFeatures(row,shortLegBoundary_config,relativeOpenCloseValuePercent_config):
    import numpy as np

    open = row['open']
    close = row['close']
    high = row['high']
    low = row['low']
    
    
    is_green,is_red = is_red_green_candle(open,close)
    
    relativeOpenCloseValue = (open-close)/close    
    if relativeOpenCloseValue<0:
        relativeOpenCloseValue=relativeOpenCloseValue*-1
    
    relativeOpenCloseValuePercent = relativeOpenCloseValue*100
    
    legLength = high-low
    midLeg = (high+low)/2
    
    deviation = .05 * legLength

    # 1.Doji    
    # Doji 
    # Formed when opening and closing prices are virtually the same    
    
    if relativeOpenCloseValuePercent<=relativeOpenCloseValuePercent_config:
        isDoji = 1
    else:
        isDoji = 0
    
    # 2.isLongLeggedCandle    
    # Long Legged Candle
    # determine if this is a relative long legged candle
    if (high-low)>= shortLegBoundary_config:
        isLongLeggedCandle = 1
    else:
        isLongLeggedCandle = 0
    
    # 3.is_longLeggedDoji    
    # Long-Legged Doji 
    # Consists of a Doji with very long upper and lower shadows. Indicates strong forces balanced in opposition
    is_longLeggedDoji = 0
    
    if isDoji and isLongLeggedCandle:        
        start_boundary = midLeg - deviation
        end_boundary = midLeg + deviation
        
        if open > start_boundary and open < end_boundary:
            is_longLeggedDoji=1
        
        if close > start_boundary and close < end_boundary:
            is_longLeggedDoji=1
     
    # 4.is_dragonFlyDoji    
    # Dragonfly Doji 
    # Formed when the opening and the closing prices are at the highest of the day. If it has a 
    # longer lower shadow it signals a more bullish trend. 
    # When appearing at market bottoms it is considered to be a reversal signal
    is_dragonFlyDoji = 0
    
    if isDoji and isLongLeggedCandle:        
        start_boundary = high
        end_boundary = high - deviation
        
        
        if open <= start_boundary and open >= end_boundary:
            is_dragonFlyDoji=1
        
        if close <= start_boundary and close >= end_boundary:
            is_dragonFlyDoji=1

    # 5.is_graveStoneDoji
    # Gravestone Doji 
    # Formed when the opening and closing prices are at the lowest of the day. 
    # If it has a longer upper shadow it signals a bearish trend. 
    # When it appears at market top it is considered a reversal signal
    is_graveStoneDoji = 0
    if isDoji and isLongLeggedCandle:        
        start_boundary = low + deviation
        end_boundary = low        
        if open <= start_boundary and open <= end_boundary:
            is_graveStoneDoji=1
        
        if close <= start_boundary and close <= end_boundary:
            is_graveStoneDoji=1
    
    # 6.is_hammer
    # Hammer 
    # A black or a white candlestick that consists of a small body near the high with a little or 
    # no upper shadow and a long lower tail. Considered a bullish pattern during a downtrend
    is_hammer = 0
    if not isDoji:
        start_boundary = high
        end_boundary = high - 2*deviation
        hammer_boundary=start_boundary-(.3*legLength)   
        if is_green:           
            if close <= start_boundary and close >= hammer_boundary:
                        
                if open >= hammer_boundary:
                    is_hammer = 1
        
        if is_red:
            if open <= start_boundary and open >= hammer_boundary:                
                if close >= hammer_boundary:
                    is_hammer = 1
            
    # 7.is_inverted_hammer
    # Inverted Hammer 
    # A black or a white candlestick in an upside-down hammer position.
    is_inverted_hammer = 0
    if not isDoji:
        start_boundary = low + 2*deviation
        end_boundary = low
        
        if is_green:            
            # print("green >> open>>"+str(open)+" start_boundary-end_boundary>>"+str(start_boundary)+"-"+str(end_boundary)+" ::"+str(open <= start_boundary and open >= end_boundary))
            if open <= start_boundary and open >= end_boundary:                
                # print("is_inverted_hammer >>> "+str(close <= end_boundary + .3*legLength))
                if close <= end_boundary + .3*legLength:
                    is_inverted_hammer = 1
        
        if is_red:
            # print('red >>> '+ str(close <= start_boundary and close >= end_boundary))
            if close <= start_boundary and close >= end_boundary:                
                # print("is_inverted_hammer >>> "+str(open <= end_boundary + .3*legLength))
                if open <= end_boundary + .3*legLength:
                    is_inverted_hammer = 1
    
    # 8.is_hanging_man
    # Hanging Man 
    # A black or a white candlestick that consists of a small body near the high with a little or 
    # no upper shadow and a long lower tail. The lower tail should be two or three times the height of the body. 
    # Considered a bearish pattern during an uptrend.
    is_hanging_man = 0
    if not isDoji:
        start_boundary = high
        end_boundary = high - 2*deviation
        hanging_man_boundary=start_boundary-(.5*legLength) 
        if is_green:           
            if close <= start_boundary and close >= hanging_man_boundary:                          
                if open >= hanging_man_boundary:
                    is_hanging_man = 1
                    
        if is_red:
            if open <= start_boundary and open >= hanging_man_boundary:                
                if close >= hanging_man_boundary:
                    is_hanging_man = 1
                    
    # 9. Long Upper Shadow    
    # A black or a white candlestick with an upper shadow that has a length of 2/3 or more of 
    # the total range of the candlestick. Normally considered a bearish signal when it appears around price resistance levels.
    is_long_upper_shadow = 0
    # NOTE 9 and 10 computed together
    
    # 10. Shooting Star
    # A black or a white candlestick that has a small body, a long upper shadow and a little or no lower tail. 
    # Considered a bearish pattern in an uptrend.
    is_shooting_star = 0
    if not isDoji:
        start_boundary = low + (1/3)*legLength
        end_boundary = low
        if (close <= start_boundary and close >= end_boundary) and (open <=start_boundary and open >= end_boundary):
            is_long_upper_shadow = 1
            if(is_green):
                is_shooting_star = 1
                
    # 11.Long Lower Shadow 
    # A black or a white candlestick is formed with a lower tail that has a length of 2/3 or more of the total 
    # range of the candlestick. Normally considered a bullish signal when it appears around price support levels.
    is_long_lower_shadow=0
    if not isDoji:
        start_boundary = high
        end_boundary = high - (1/3)*legLength
        if (close <= start_boundary and close >= end_boundary) and (open <=start_boundary and open >= end_boundary):
            is_long_lower_shadow = 1
            
    # 12.Marubozu 
    # A long or a normal candlestick (black or white) with no shadow or tail. 
    # The high and the lows represent the opening and the closing prices. Considered a continuation pattern.
    is_Marubozu = 0
    if np.absolute(high-low) == np.absolute(open-close):
        is_Marubozu = 1
        
    # 13. Shaven Head 
    # A black or a white candlestick with no upper shadow. [Compared with hammer.]
    is_shavenHead = 0
    if not is_Marubozu:
        if open == high or close == high:
            is_shavenHead = 1
            
    # 14. Shaven Bottom 
    # A black or a white candlestick with no lower tail. [Compare with Inverted Hammer.]
    is_shavenBottom = 0
    if not is_Marubozu:
        if open == low or close == low:
            is_shavenBottom = 1
            
    # 15. Spinning Top 
    # A black or a white candlestick with a small body. The size of shadows can vary. 
    # Interpreted as a neutral pattern but gains importance when it is part of other formations.
    is_spinningTop = 0
    if not (is_Marubozu or isDoji or is_shavenHead or is_shavenBottom):
        if is_green:
            if high-open >= (.2*legLength) and close-low >= (.2*legLength) :
                is_spinningTop = 1
        if is_red:
            if high-close >= (.2*legLength) and open-low >= (.2*legLength) :
                is_spinningTop = 1
        
    row['is_spinningTop'] = is_spinningTop
    row['is_shavenBottom'] = is_shavenBottom
    row['is_shavenHead'] = is_shavenHead
    row['is_Marubozu'] = is_Marubozu
    row['is_long_lower_shadow']=is_long_lower_shadow
    row['is_long_upper_shadow']=is_long_upper_shadow
    row['is_shooting_star']=is_shooting_star
    row['is_hanging_man'] = is_hanging_man
    row['is_LongLeggedCandle']=isLongLeggedCandle
    row['is_Doji']=isDoji    
    row['is_longLeggedDoji']=is_longLeggedDoji
    row['is_dragonFlyDoji']=is_dragonFlyDoji
    row['is_graveStoneDoji']=is_graveStoneDoji
    row['is_hammer']=is_hammer
    row['is_inverted_hammer']=is_inverted_hammer
    row['is_hanging_man']=is_hanging_man

    return row
