'''This script contains various functions used for the execution and evaluation of time series analysis projects'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns

#function to calculate MAPE for all observations where y_true is not 0
def mape(y_true, y_predict):
    '''Returns mean percentage error for all predictions where y_true is not 0. Where y_true is 0, the percentage error is 0 as well '''
    return np.mean([np.absolute(y_true[idx] - y_predict[idx])/y_true[idx] * 100 if y_true[idx] != 0 else 0 for idx,_ in enumerate(y_true) ])

def median_pe(y_true, y_predict):
    '''Returns mean percentage error for all predictions where y_true is not 0. Where y_true is 0, the percentage error is 0 as well '''
    return np.median([np.absolute(y_true[idx] - y_predict[idx])/y_true[idx] * 100 if y_true[idx] != 0 else 0 for idx,_ in enumerate(y_true) ])

def residuals(y_true,y_predict):
    '''Returns list with residuals for all observations where y_true not 0. Where y_true is 0, the residuals are 0 as well '''
    return [y_true[idx] - y_predict[idx] if y_true[idx] != 0 else 0 for idx,_ in enumerate(y_true) ]

def pct_residuals(y_true,y_predict):
    '''Returns list with percentage errors for all observations where y_true not 0. Where y_true is 0, the percentage error is 0 as well'''
    return [(y_true[idx] - y_predict[idx])/y_true[idx] * 100 if y_true[idx] != 0 else 0 for idx,_ in enumerate(y_true) ]

#naive model that predicts future by taking value from 7 days ago
def naive_base_shift(y_true,shift=7):
    y_predict = y_true.shift(shift)
    return y_predict

#plot the cdf function for the residuals
def plot_residuals(data):
    '''Plots cdf of residual input data'''
    # sort the data:
    data_sorted = np.sort(data)

    # calculate the proportional values of samples
    p = 1. * np.arange(len(data)) / (len(data) - 1)

    # plot the sorted data:
    fig = plt.figure(figsize=(20,15))


    ax1 = fig.add_subplot(311)
    ax1.plot(data_sorted, p)
    ax1.set_title('Residuals Cumulative Distribution Function')
    ax1.set_xlabel('Residuals');
    ax1.set_ylabel('Cumulative Distribution');
    ax1.axvline(x=np.percentile(data,5),color='r') 
    ax1.axvline(x=np.percentile(data,95),color='r')

    ax2 = fig.add_subplot(312)
    ax2.plot([idx for idx,_ in enumerate(data)],data,'bo');
    ax2.plot([idx for idx,_ in enumerate(data)],np.zeros(len(data)),'r-');
    ax2.set_title('Residuals over time')
    ax2.set_xlabel('Time in days');
    ax2.set_ylabel('Residual');  
    
    #Here, we could also add Q-Q plot and auto correlation plot for the residual
    
def plot_prediction(y_true,y_predict,title=None):
    '''Plots true and predicted values on same y-axis'''
    fig = plt.figure(figsize=(20,15))
    ax1 = fig.add_subplot(311)
    ax1.plot(range(len(y_true)), y_true,'bo')
    ax1.plot(range(len(y_predict)),y_predict,'r-')
    
    if title == None:
        ax1.set_title('Complete prediction')
    else:
        ax1.set_title(f'{title}: Complete prediction')
    
    
    ax2 = fig.add_subplot(312)
    ax2.plot(range(len(y_true[:60])), y_true[:60],'bo')
    ax2.plot(range(len(y_predict[:60])),y_predict[:60],'r-o')
    ax2.set_title('Prediction first 60 days')
    ax2.set_ylim(0,max(y_true))
    ax3 = fig.add_subplot(313)
    ax3.plot(range(len(y_true[-60:])), y_true[-60:],'bo')
    ax3.plot(range(len(y_predict[-60:])),y_predict[-60:],'ro-')
    ax3.set_title('Prediction last 60 days')
    
def management_summary(y_true,y_predict):
    data = pd.DataFrame.from_dict({'y_true':y_true, 'y_predict':y_predict})
    
    #only regard data where y_true is not 0
    ex_0 = data[data['y_true'] != 0]
    
    #calculate how ofter we under- and over-estimate the revenue
    pct_lower = round(sum(ex_0.y_predict - ex_0.y_true < 0)/len(ex_0.y_true) * 100,1)
    pct_higher = round(100 - pct_lower,1)
    
    #calculate cumulative sums of under- and over estimation
    cumsum_lower = np.cumsum([np.abs(ex_0.y_predict[idx] - ex_0.y_true[idx]) if ex_0.y_predict[idx] < ex_0.y_true[idx] else 0 for idx,y in enumerate(ex_0.y_true) ])
    cumsum_higher = np.cumsum([np.abs(ex_0.y_predict[idx] - ex_0.y_true[idx]) if ex_0.y_predict[idx] > ex_0.y_true[idx] else 0 for idx,y in enumerate(ex_0.y_true)])

    
    fig = plt.figure(figsize=(20,15))
    ax1 = fig.add_subplot(211)
    ax1.plot(range(len(ex_0.y_true)), cumsum_lower,'b-o')
    ax1.plot(range(len(ex_0.y_true)),cumsum_higher,'r-o')
    
    ax1.set_title('Cumulative Sums of Under- and Over-Estimation')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Cumulated sum of errors')
    ax1.legend(['Under Estimation', 'Over Estimation', 'True Values'])
    
    ax2 = ax1.twinx()
    color = 'black'
    ax2.set_ylabel('Measured Values', color = color)
    ax2.plot(range(len(ex_0.y_true)),ex_0.y_true,'--', color=color, marker=10)
    
    return f'The model underestimates {pct_lower}% of the time'