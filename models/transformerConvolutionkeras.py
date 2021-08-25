import keras
from keras.models import Sequential
from keras import optimizers, losses, layers
from keras import backend as K
import tensorflow as tf
import skopt
from skopt import gp_minimize
from skopt.utils import use_named_args
from skopt.space import Real, Categorical, Integer 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

def transformer_encoder(inputs, head_size, num_heads, ff_dim, n_kernel, n_strides, dropout=0):
    # Normalization and Attention
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.Conv1D(filters=ff_dim, kernel_size=(n_kernel,), padding="same", activation="relu")(res)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=(n_kernel,), padding="same")(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)

    return x + res

class transformer:
    '''
    TRANSFORMER for time series regression tasks
    '''
    def __init__(self, x_train, x_test, y_train, y_test):
        # input data
        self.x_train = np.array(x_train)
        self.x_test = np.array(x_test)
        self.y_train = np.array(y_train)
        self.y_test = np.array(y_test)
        # n_inputs
        if self.x_train.shape == (x_train.size,):
            self.n_inputs = 1
        elif len(self.x_train.shape)==2:
            self.n_inputs = self.x_train.shape[1]
            self.lagDays = 1
        else:
            self.n_inputs = self.x_train.shape[2]
            self.lagDays = self.x_train.shape[1]
            
        # n_outputs
        if self.y_train.shape == (y_train.size,):
            self.n_outputs = 1
        else:
            self.n_outputs = self.y_train.shape[1]

    def build_model(self, head_size, num_heads, ff_dim, num_transformer_blocks, 
        n_hidden_layers, n_hidden_neurons, n_kernel, n_strides, dropout=0, mlp_dropout=0):
        """
        Building the full model based on previous variables
        """
        
        inputs = keras.Input(shape=(self.lagDays, self.n_inputs))
        # loop of transformers
        x = transformer_encoder(inputs, head_size, num_heads, ff_dim, n_kernel, n_strides)
        for _ in range(num_transformer_blocks-1):
            x = transformer_encoder(x, head_size, num_heads, ff_dim, n_kernel, n_strides)

        # final MLP to regression
        x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
        for _ in range(n_hidden_layers):
            x = layers.Dense(n_hidden_neurons, activation="relu")(x)
            x = layers.Dropout(mlp_dropout)(x)

        outputs = layers.Dense(self.n_outputs, activation="relu")(x)
        self.model = keras.Model(inputs, outputs)
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        return self.model

    def train(self, verbose=0, show_graph=False):
        '''
        It trains the model, previously build by 'build_model' or 
        'bayesian_optimization'

        Input:
            verbose -> 
                '1': Show details
                '0': Do not show details
            show_graph -> 
                True: See training graph
                False: Do not see training graph
        '''
        
        self.history=self.model.fit(
            self.x_train, 
            self.y_train, 
            epochs=50, 
            verbose=verbose)
            #callbacks = self.my_callback)
            
        if show_graph:
            plt.figure(figsize=(10,5), dpi=500)
            plt.figure()
            plt.plot(self.history.history['mae'])
            plt.plot(self.history.history['loss'])
            plt.legend(['MAE','MSE'])
            plt.show()

    def predict(self):
        '''
        It returns a numpy array with the predictions with the shape:
        pred.shape = (len(x_test),)
        '''
        self.pred = np.array(self.model.predict(self.x_test))
        return self.pred
