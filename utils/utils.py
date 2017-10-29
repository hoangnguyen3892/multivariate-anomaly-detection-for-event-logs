import pandas as pd
import numpy as np
import math
from math import sqrt
from datetime import timedelta

import torch
import torch.nn as nn



def calculateCumDuration(df):
    df['CumDuration'] = (df['CompleteTimestamp'] - df['CompleteTimestamp'].iloc[0])
    return df

def calculateAnomalousCumDuration(df):
    df['AnomalousCumDuration'] = (df['AnomalousCompleteTimestamp'] - df['AnomalousCompleteTimestamp'].iloc[0])
    return df

def calculateDuration(df):
    df['Duration'] = (df['CompleteTimestamp'] - df['CompleteTimestamp'].shift(1)).fillna(0)
    return df

def convert2seconds(x):
    x = x.total_seconds()
    return x

def OHE(df, categorical_variables):
    for i in categorical_variables:
        enc_df = pd.get_dummies(df, columns=categorical_variables, drop_first=False)
    return enc_df

def findLongestLength(groupByCase):
    '''This function returns the length of longest case'''
    #groupByCase = data.groupby(['CaseID'])
    maxlen = 1
    for case, group in groupByCase:
        temp_len = group.shape[0]
        if temp_len > maxlen:
            maxlen = temp_len
    return maxlen

def padwithzeros(vector, maxlen):
    '''This function returns the (maxlen, num_features) vector padded with zeros'''
    npad = ((maxlen-vector.shape[0], 0), (0, 0))
    padded_vector = np.pad(vector, pad_width=npad, mode='constant', constant_values=0)
    return padded_vector

def getInput(groupByCase, cols, maxlen):
    full_list = []
    for case, data in groupByCase:
        temp = data.as_matrix(columns=cols)
        temp = padwithzeros(temp, maxlen)
        full_list.append(temp)
    inp = np.array(full_list)
    return inp

def getModifiedInput(groupByCase, cols, maxlen):
    full_list = []
    for case, data in groupByCase:
        temp = data.as_matrix(columns=cols)
        #temp = padwithzeros(temp, maxlen)
        full_list.append(temp)
    inp = np.array(full_list)
    return inp

def getProbability(recon_test):
    '''This function takes 3d tensor as input and return a 3d tensor which has probabilities for 
    classes of categorical variable'''
    softmax = nn.Softmax()
    #recon_test = recon_test.view(c_test.shape)
    
    for i in range(recon_test.size(0)):
        cont_values = recon_test[i, :, 0].contiguous().view(recon_test.size(1),1) #(35,1)
        softmax_values = softmax(recon_test[i, :, 1:])
        if i == 0:
            recon = torch.cat([cont_values, softmax_values], 1)
            recon = recon.contiguous().view(1,recon_test.size(1), recon_test.size(2)) #(1, 35, 8)
        else:
            current_recon = torch.cat([cont_values, softmax_values], 1)
            current_recon = current_recon.contiguous().view(1,recon_test.size(1), recon_test.size(2)) #(1, 35, 8)
            recon = torch.cat([recon, current_recon], 0)
    return recon


def getError(predicted_tensor, true_tensor, pad_matrix):
    '''
    This function converts a tensor to a pandas dataframe
    Return: Dataframe with columns (NormalizedTime, PredictedActivity)

    - predicted_tensor: recon
    - df: recon_df_w_normalized_time
    '''
    predicted_tensor = getProbability(predicted_tensor) #get probability for categorical variables
    predicted_array = predicted_tensor.data.cpu().numpy() #convert to numpy array

    true_array = true_tensor.data.cpu().numpy()
    
    #Remove 0-padding
    temp_predicted_array = predicted_array*pad_matrix
    temp_predicted_array = temp_predicted_array.reshape(predicted_array.shape[0]*predicted_array.shape[1], predicted_array.shape[2])
    temp_predicted_array = temp_predicted_array[np.any(temp_predicted_array != 0, axis=1)]
    
    temp_true_array = true_array*pad_matrix
    temp_true_array = temp_true_array.reshape(true_array.shape[0]*true_array.shape[1], true_array.shape[2])
    temp_true_array = temp_true_array[np.any(temp_true_array != 0, axis=1)]

    predicted_time = temp_predicted_array[:, 0]
    predicted_activity = temp_predicted_array[:, 1:]

    true_time = temp_true_array[:, 0]
    true_activity = temp_true_array[:, 1:]
    return predicted_time, predicted_activity, true_time, true_activity










