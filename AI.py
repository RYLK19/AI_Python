import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('C:\\Users\\theek\\OneDrive\\Desktop\\AI_Python\\Car_Purchasing_Data.csv')

print("First 5 Rows:")
print(df.head())

print("Last 5 Rows")
print(df.tail())

rows,columns = df.shape

print("Numbers of Row:",rows)
print("Numbers of Columns:",columns)

print("summary:")
print(df.info())

print("null values:")
print(df.isnull().sum())

print("Describes min max/ only nunbers:")
print(df.describe())



sns.pairplot(df, vars=['Age', 'Annual Salary', 'Credit Card Debt', 'Net Worth', 'Gender'])
plt.suptitle('Pair Plot of Age and Annual Salary', y=1.0)
plt.show()

