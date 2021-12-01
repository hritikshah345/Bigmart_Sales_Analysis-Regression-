# -*- coding: utf-8 -*-
"""
Created on Tue May 25 14:35:52 2021

@author: Hritik
"""

#BIGMART Sales prediction Analysis
import math
import numpy as np
np.__version__
import pandas as pd
pd.__version__
import os, inspect
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import warnings
%matplotlib inline
warnings.filterwarnings('ignore')

#Loading the dataset

data = pd.read_csv(r"F:\quants\download\Downloads\archive (2)\Train.csv")
data.columns
data.describe()


data.apply(lambda x: len(x.unique()))
data.value_counts
data
data.isnull().sum()
data.dtypes.index

#For categorical attributes
categorical_col = []
for x in df.dtypes.index:
    if data.dtypes[x] == 'object':
        categorical_col.append(x)
#as we dont need any Identifiers        
categorical_col.remove('Item_Identifier')
categorical_col.remove('Outlet_Identifier')

categorical_col

for column in categorical_col:
    print(col)
    print(data[column].value_counts())
    print()
    
data[categorical_col].value_counts()
np.unique(data[:, 2])

temp_mean = np.nanmean(data['Item_Weight'], axis = 0).round(3)
temp_mean
np.isnan(data["Item_Weight"])
Item_weight_mean = data.pivot_table(values = "Item_Weight", index = "Item_Identifier")
Item_weight_mean

missing_values = data['Item_Weight'].isnull()
missing_values


#filling the missing values
for i, item in enumerate(data["Item_Weight"]):
    if missing_values[i]:
        if item in Item_weight_mean:
            data['Item_Weight'][i] = Item_weight_mean.loc[item]['Item_Weight']
        else:
            data['Item_Weight'][i] = 0
            
np.isnan(data["Item_Weight"]).sum()


Outlet_size_MV = data.pivot_table(values = 'Outlet_Size', columns = "Outlet_Type", aggfunc=(lambda x: x.mode()[0]))
Outlet_size_MV


miss_values_OS = data["Outlet_Size"].isnull()
miss_values_OS

data.loc[miss_values_OS, "Outlet_Size"] = data.loc[miss_values_OS, "Outlet_Type"].apply(lambda x: Outlet_size_MV[x])            
#Item visibility            
data["Item_Visibility"].unique()
sum(data['Item_Visibility']==0)

data['Item_Visibility'] = np.where(data['Item_Visibility'] == 0, np.mean(data['Item_Visibility']), data["Item_Visibility"])


data['Item_Fat_Content'] = data['Item_Fat_Content'].replace({'LF': 'Low Fat', 'reg': 'Regular', 'low fat': 'Low Fat'})
np.unique(data['Item_Fat_Content'])

#Creating New attribute******
data['Item_Name'] = data['Item_Identifier'].apply(lambda x: x[:2])
np.unique(data['Item_Name'])
data['Item_Name'] = data['Item_Name'].replace({'DR': 'Drinks', 'FD': 'Food', 'NC': 'Non Consumable'})

data.loc[df['Item_Name'] == 'Non Consumable', 'Item_Fat_Content'] = 'Non_Edible'
np.unique(data['Item_Fat_Content'])


#Year size minimizing
np.unique(data['Outlet_Establishment_Year'])
data['Outlet_Years'] = 2013 - data['Outlet_Establishment_Year'] 

data.columns

##Data Visualization
sns.distplot(data['Item_Visibility'])
sns.distplot(data['Item_Weight'])
sns.distplot(data['Item_MRP'])
sns.distplot(data['Item_Outlet_Sales'])

#Log transformation for sales 
data['Item_Outlet_Sales'] = np.log(1 + data['Item_Outlet_Sales'])
sns.distplot(data['Item_Outlet_Sales'])

#Catergorical Attributes
sns.countplot(data['Item_Fat_Content'])
sns.countplot(data['Item_Type'])

# As we require bigger plot size so that Item_Type is Visible
plt.figure(figsize=(12, 8))
plt.xticks(rotation = 90, fontsize = 13)
plt.yticks(fontsize = 13)
sns.countplot(data['Item_Type'])
plt.show()

sns.countplot(data['Outlet_Establishment_Year'])
sns.countplot(data['Outlet_Size'])
sns.countplot(data['Outlet_Location_Type'])


plt.figure(figsize=(12, 8))
plt.xticks(rotation = 90, fontsize = 13)
plt.yticks(fontsize = 13)
sns.countplot(data['Outlet_Type'])
plt.show()


#Corelation Matrix
corr = data.corr()
sns.heatmap(corr, annot = True, cmap = 'coolwarm')
#Label Encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['Outlet'] = le.fit_transform(data['Outlet_Identifier'])
data.columns
data.head()
categorical_col = ['Item_Fat_Content', 'Item_Type', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type', 'Item_Name']
for col in categorical_col:
    data[col] = le.fit_transform(data[col])
    
#Onhot encoding
data = pd.get_dummies(data, columns = ['Item_Fat_Content', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type', 'Item_Name'])
data.head()


#Input split

X = data.drop(columns = ['Outlet_Establishment_Year', 'Item_Identifier', 'Outlet_Identifier', 'Item_Outlet_Sales'])
y = data['Item_Outlet_Sales']

#Model training
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
def train(model, X, y):
    model.fit(X, y)
    pred = model.predict(X)
    cv_score = cross_val_score(model, X, y, scoring = 'neg_mean_squared_error')
    cv_score = np.abs(np.mean(cv_score))
    print("Model report")
    print("MSE:", mean_squared_error(y, pred))
    print("Cv_Score:", cv_score)
    
from sklearn.linear_model import LinearRegression, Ridge, Lasso
model = LinearRegression(normalize=True)
train(model, X, y)
coef = pd.Series(model.coef_, X.columns).sort_values()
coef.plot(kind = 'bar')

#Ridge model
model = Ridge(normalize=True)
train(model, X, y)
coef = pd.Series(model.coef_, X.columns).sort_values()
coef.plot(kind = 'bar')

#DecisionTreeregressor model
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
train(model, X, y)
coef = pd.Series(model.feature_importances_, X.columns).sort_values(ascending=True)
coef.plot(kind = 'bar')

#linear regression model gives best results