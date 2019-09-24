# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 16:45:30 2019

@author: Shriyash Shende
"""

import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns
import statsmodels.formula.api as st
from scipy import stats

sta = pd.read_csv('C:\\Users\\Good Guys\\Desktop\\pRACTICE\\EXCELR PYTHON\\Assignment\\Multilinear Regression\\50_Startups.csv')

sta.describe()
sta.columns
mode = stats.mode(sta['State'])
print(mode.mode)
print(mode.count)

sns.boxplot(sta['R&D Spend'])
sns.boxplot(sta['Administration'])
sns.boxplot(sta['Marketing Spend'])
sns.boxplot(sta['Profit'])

sta.corr()
sns.pairplot(sta)
sns.distplot(sta['R&D Spend'])
sns.distplot(sta['Administration'])
sns.distplot(sta['Marketing Spend'])
sns.distplot(sta['Profit'])

new_sta = sta.drop(['Administration'], axis = 1)

data_with_dummies = pd.get_dummies(new_sta, drop_first = True)
data_with_dummies.columns

ml1 = st.ols("data_with_dummies['Profit'] ~ data_with_dummies['R&D Spend'] + data_with_dummies['Marketing Spend'] + data_with_dummies['State_Florida'] + data_with_dummies['State_New York'] ",data= data_with_dummies).fit()
ml1.summary()
ml2 = st.ols("data_with_dummies['Profit'] ~ data_with_dummies['R&D Spend']",data= data_with_dummies).fit()
ml2.summary()
ml3 = st.ols("data_with_dummies['Profit'] ~ data_with_dummies['Marketing Spend']",data= data_with_dummies).fit()
ml3.summary()

ml3 = st.ols("data_with_dummies['Profit'] ~ data_with_dummies['Marketing Spend']  + data_with_dummies['R&D Spend']",data= data_with_dummies).fit()
ml3.summary()

import statsmodels.api as sm

sm.graphics.influence_plot(ml1)
sta_new = data_with_dummies.drop(data_with_dummies.index[[46,49]],axis=0)
ml4 = st.ols("data_with_dummies['Profit'] ~ data_with_dummies['Marketing Spend']  + data_with_dummies['R&D Spend'] + data_with_dummies['State_Florida'] + data_with_dummies['State_New York']",data= data_with_dummies).fit()
ml4.summary()

new_sta2 = sta_new.drop(['State_New York'], axis = 1)
new_sta2 = sta_new.drop(['State_Florida'], axis = 1)

ml5 = st.ols("new_sta2['Profit'] ~ new_sta2['Marketing Spend']  + new_sta2['R&D Spend']",data= new_sta2).fit()
ml5.summary()



###Exponential Transformation
ml6 = st.ols("np.log(new_sta2['Profit']) ~ new_sta2['Marketing Spend']  + new_sta2['R&D Spend']", data = new_sta2).fit()
ml6.summary()

###Polynomail Transformation
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree = 2)  #Degree 2
X1 = X.drop(['State_Florida'],axis=1)
X1 = X1.drop(['State_New York'],axis=1)

X_poly = poly.fit_transform(X1)
from sklearn.linear_model import LinearRegression
lireg = LinearRegression()
lireg.fit(X_poly, Y)
score0 = cross_val_score(lireg,X1,Y,cv = 2)
score0
score0.mean()#86.00


#NOW BY USING SKLEARN RIDGE
from sklearn.linear_model import Ridge
new_order_data = data_with_dummies[['R&D Spend','Marketing Spend','State_Florida','State_New York','Profit']]
X = new_order_data.drop(["Profit"],axis=1)
Y = new_order_data["Profit"]



from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.25)


clf = Ridge(alpha = 1.0)
clf.fit(X_train,y_train)
clf.score(X_train,y_train) #96.50
clf.score(X_test,y_test)  #90.50

#Cross Validation
from sklearn.model_selection import cross_val_score
score = cross_val_score(clf,X,Y,cv = 2)
score
score.mean() #84.2


from sklearn import linear_model
clf1 = linear_model.Lasso(alpha = 0.1)
clf1.fit(X_train, y_train)
clf1.score(X_train, y_train) #96.50
clf1.score(X_test,y_test) #90.35

score1 = cross_val_score(clf1,X,Y,cv = 2)
score1
score1.mean() #83.45


from sklearn.linear_model import ElasticNet
clf2 = ElasticNet(random_state=0)
clf2.fit(X,Y)
clf2.score(X_train, y_train)#96.32
clf2.score(X_test,y_test)#91.34

score2 = cross_val_score(clf2,X,Y,cv = 2)
score2
score2.mean() #85.50
