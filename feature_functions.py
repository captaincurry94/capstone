import pandas as pd
import numpy as np

#add features related to time
def df_add_timefeatures(dataframe):
    '''Adds features related to time that are generated from datetime index of dataframe'''
    dataframe['day_of_week'] = [x.weekday() +1 for x in dataframe.index]
    dataframe['day_of_month'] = [x.date().day for x in dataframe.index]
    dataframe['day_of_year'] = [x.dayofyear for x in dataframe.index]
    dataframe['weekofyear'] = [x.weekofyear for x in dataframe.index]
    dataframe['month'] = [x.month for x in dataframe.index]
    return dataframe

#get one value for lag of size n for step-forward prediction
def get_lagvalue(history,lagvalue):
    if len(history) < lagvalue:
        return np.nan
    else:
        return history[-(lagvalue):-(lagvalue-1)][0]
    
#get mean value for weekday in past n weeks for one step in step-forward prediction
def get_weekdaymean(history,weeks):
    return np.mean([get_lagvalue(history,7*n) for n in range(1,weeks+1)])

#combine the functions th
def apply_time_featurefun(df,column,daylags=[7,14],weekdaymeans=[2,4],history_df=[]):
    if len(history_df) >0:
        dataframe = pd.concat((history_df[[column]],df[[column]]))
    else:
        dataframe = df[[column]]
    
    dataframe.columns = ['y']
    
    for i in daylags:
        dataframe['lag_'+str(i)] = [get_lagvalue(dataframe['y'][:idx],i) for idx in range(len(dataframe))]
    for i in weekdaymeans:
        dataframe['weekdaymean_'+str(i)+'_weeks'] = [get_weekdaymean(dataframe['y'][:idx],i) for idx in range(len(dataframe))]
            
    dataframe = df_add_timefeatures(dataframe)
    
    return dataframe[-len(df):]

#make dummies from given columns and drop first value
def dummies_from_cols(df,cols):
    for col in cols:
        df = pd.get_dummies(df,columns=[col],drop_first=True,prefix=col)
    return df