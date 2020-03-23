#import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

#load the data
tdata = pd.read_excel('Train_dataset.xlsx',index_col=False)
tdata.shape

#checking for null values
tdata.isnull().sum()

#droping the unwanted data
tdata = tdata.drop(['people_ID','Designation','Name','Insurance','salary'],axis=1)
tdata.shape
tdata.isnull().sum()

#filling the nan values with its average value
tdata['Children'].fillna(0,inplace=True)
tdata['Occupation'].fillna('No Occupation',inplace=True)
tdata['Occupation'].unique()
tdata['Mode_transport'].isnull().sum()
tdata['Mode_transport'].fillna("None",inplace=True)
tdata['Mode_transport'].unique()
tdata['comorbidity'].unique()
tdata['comorbidity'].fillna("None",inplace=True)
tdata['cardiological pressure'].fillna('None',inplace=True)
tdata['cardiological pressure'].unique()
D_mean = np.mean(tdata['Diuresis'])
tdata['Diuresis'].fillna(np.trunc(D_mean),inplace=True)
P_mena = np.mean(tdata['Platelets'])
tdata['Platelets'].fillna(np.trunc(P_mena),inplace=True)
H_mean = np.mean(tdata['HBB'])
tdata['HBB'].fillna(np.trunc(H_mean),inplace=True)
d_mean = np.mean(tdata['d-dimer'])
tdata['d-dimer'].fillna(np.trunc(d_mean),inplace=True)
A_mean = np.mean(tdata['Heart rate'])
tdata['Heart rate'].fillna(np.trunc(A_mean),inplace=True)
c_mean = np.mean(tdata['HDL cholesterol'])
tdata['HDL cholesterol'].fillna(np.trunc(c_mean),inplace=True)
tdata['FT/month'].fillna(0,inplace=True)

tdata.isnull().sum()

#converting the all cateogries in binary data using Label encoder

from sklearn.preprocessing import LabelEncoder
Lb= LabelEncoder()
tdata['Region'] = Lb.fit_transform(tdata['Region'])
tdata['Gender'] = Lb.fit_transform(tdata['Gender'])
tdata['Married'] = Lb.fit_transform(tdata['Married'])
tdata['Occupation'] = Lb.fit_transform(tdata['Occupation'])
tdata['Mode_transport'] = Lb.fit_transform(tdata['Mode_transport'])
tdata['comorbidity'] = Lb.fit_transform(tdata['comorbidity'])
tdata['Pulmonary score'] = Lb.fit_transform(tdata['Pulmonary score'])
tdata['cardiological pressure'] = Lb.fit_transform(tdata['cardiological pressure'])


x=tdata.iloc[:,:-1].values
y=tdata['Infect_Prob'].values


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)



#model selection
from sklearn.linear_model import LinearRegression
svr_model  = LinearRegression()
svr_model.fit(x_train,y_train)

prediction = svr_model.predict(x_test)
#spliting the data