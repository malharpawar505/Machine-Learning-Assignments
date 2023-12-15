import pandas as pd
import numpy as np

import matplotlib as plt
from sklearn.model_selection import train_test_split
df=pd.read_csv("electricity.csv")
df.describe()
print(np.quantile(df['Consumption'], [0.25,0.5,0.75]))
print(np.quantile(df['Production'], [0.25,0.5,0.75]))

#2
print(np.mean(df['Nuclear']))

#3
df.isnull().sum()

#4
df.duplicated().sum()

#5
plt.figure(figsize=(20,5))
plt.plot(df.index,df['Production'],'b')
plt.plot(df.index,df['Consumption'],'r')
plt.title(' Production vs Consumption ')
plt.show()

#6
df["DateTime"]=pd.to_datetime(df['DateTime'])
df['year']=df['DateTime'].dt.year
print(df)

df_year=df.groupby("year").mean()
plt.figure(figsize=(20,5))
plt.plot(df_year.index,df_year['Consumption'],'r')
plt.plot(df_year.index,df_year['Production'],'b')

plt.grid()
plt.show()