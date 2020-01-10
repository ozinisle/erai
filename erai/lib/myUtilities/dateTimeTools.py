import pandas as pd
import datetime

def convertStringToDate(dateString,dateFormat='%Y-%m-%dT%H:%M:%S+0530'):

    return datetime.datetime.strptime(dateString, dateFormat)

def getWeekDay(dateTimeObj):
    # weekday Monday is 0 and Sunday is 6
    # print("weekday():", datetime.date(2010, 6, 16).weekday())

    # isoweekday() Monday is 1 and Sunday is 7
    return dateTimeObj.isoweekday()

def convertDateStringColumnToWeekDayColumn(dateStringColumn):
    dateColumn = pd.to_datetime(dateStringColumn)
    return dateColumn.dt.weekday