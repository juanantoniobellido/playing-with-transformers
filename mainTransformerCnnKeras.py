#%%
import pandas as pd
from utils.splitDataByYear import splitDataByYear
from models.transformerConvolutionkeras import transformer

df = pd.read_csv("data/example-data.csv", sep=';')
df['date'] = pd.to_datetime(df['date'])
df.info()

xTrain, xTest, yTrain, yTest = splitDataByYear(
    df = df, 
    station = 'COR06',
    yearTestStart = 2014,
    varListInputs = ['tx', 'tn', 'rs'], 
    varListOutputs = ['et0'])

model = transformer(xTrain, xTest, yTrain, yTest)

model.build_model(
    head_size = 2, 
    num_heads = 2, 
    ff_dim = 16, 
    num_transformer_blocks = 2, 
    n_hidden_layers = 2, 
    n_hidden_neurons = 5, 
    n_kernel= 5, 
    n_strides= 2, 
    dropout=0, 
    mlp_dropout=0)

model.train()
yPred = model.predict()




# %%
