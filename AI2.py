import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import MinMaxScaler 



df = pd.read_csv('C:\\Users\\theek\\OneDrive\\Desktop\\AI_Python\\Car_Purchasing_Data.csv')

irrelevent = ['Customer Name', 'Customer e-mail', 'Country', 'Car Purchase Amount']

input_df = df.drop(columns=irrelevent)

print(input_df.head())


output_df = df[['Car Purchase Amount']]

print(output_df.head())

scaler = MinMaxScaler()


sinput = scaler.fit_transform(input_df)

scaled_input_df = pd.DataFrame(sinput, columns=input_df.columns)
print("Scaled Input Data:")
print(scaled_input_df.head)

output_df = df[['Car Purchase Amount']]
print("Output Data:")
print(output_df.head())

