import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def splitDataByYear(
    df, 
    station,
    yearTestStart,
    varListInputs, 
    varListOutputs,
    preprocessing = 'standardization'):

    """
    It splits the dataset into training and testing according to the station.
    Very useful in regional scenarios

    Inputs:
        df (dataframe) - Input DataFrame
        station  (str) - station
        yearTestStart (int) - year to split training-test. Year where the test start
        varListInputs (list) - List with input variable configuration
        varListOutputs (list) - List with target variables
        preprocessing (str) - 'Standardization' or 'Normalization' or 'None'

    outputs:
        xTrain (np.array) - numpy array with training input data
        yTrain (np.array) - numpy array with training target data
        xTest (np.array) - numpy array with testing input data
        yTest (np.array) - numpy array with testing target data
    """
    # errors
    assert 'station' in df.columns, "'station'does not exist in the dataframe"
    assert 'date' in df.columns, "'date' does not exist in the dataframe"
    assert 'year' in df.columns, "'year' does not exist in the dataframe"

    #join all var configurations
    varList = varListInputs + varListOutputs

    df = df[df['station']==station]

    # split to train and test
    dfStationTrain = df[df['year']<yearTestStart]
    dfStationTrain = dfStationTrain.filter(items=varList)
    xTrain = dfStationTrain.filter(items=varListInputs).to_numpy()
    yTrain = dfStationTrain.filter(items=varListOutputs).to_numpy()

    dfStationTest = df[df['year']>=yearTestStart]
    dfStationTest = dfStationTest.filter(items=varList)
    xTest = dfStationTest.filter(items=varListInputs).to_numpy()
    yTest = dfStationTest.filter(items=varListOutputs).to_numpy()

    # standardization or normalization
    if preprocessing == 'standardization':
        scaler = StandardScaler()
        scaler.fit(xTrain)
        xTrain = scaler.transform(xTrain)
        xTest = scaler.transform(xTest)

    elif preprocessing == 'normalization':
        scaler = MinMaxScaler()
        scaler.fit(xTrain)
        xTrain = scaler.transform(xTrain)
        xTest = scaler.transform(xTest)
    
    return xTrain, xTest, yTrain, yTest