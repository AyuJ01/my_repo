# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 02:04:14 2018

@author: Ayushi
"""
import pandas as pd
df = pd.read_table('Dota2data.txt', sep=',',header=None)


features=df.iloc[:,0:10]
labels=df.iloc[:,10].values

#Label Encoding
from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder() 
for i in range(0,10):
        features[i]=labelencoder.fit_transform(features[i])

features = features.iloc[:,:].values

"""
#Building the optimal model using backward elimination
import numpy as np
import statsmodels.formula.api as sm

features = np.append(arr = np.ones((15000,1)).astype(int),values = features,axis=1)

features_opt = features[:,[0,1,2,3,4,5,6,7,8,9,10]]
regressor_OLS = sm.OLS(endog=labels,exog = features_opt).fit()
regressor_OLS.summary()


features_opt = features[:,[0,1,2,3,4,6,7,8,9,10]]
regressor_OLS = sm.OLS(endog=labels,exog = features_opt).fit()
regressor_OLS.summary()


features_opt = features[:,[0,2,3,4,6,7,8,9,10]]
regressor_OLS = sm.OLS(endog=labels,exog = features_opt).fit()
regressor_OLS.summary()


features_opt = features[:,[0,2,3,4,6,7,9,10]]
regressor_OLS = sm.OLS(endog=labels,exog = features_opt).fit()
regressor_OLS.summary()


features_opt = features[:,[0,2,3,4,6,7,9,10]]
regressor_OLS = sm.OLS(endog=labels,exog = features_opt).fit()
regressor_OLS.summary()


features_opt = features[:,[0,2,3,4,6,7,9]]
regressor_OLS = sm.OLS(endog=labels,exog = features_opt).fit()
regressor_OLS.summary()


features_opt = features[:,[0,2,3,6,7,9]]
regressor_OLS = sm.OLS(endog=labels,exog = features_opt).fit()
regressor_OLS.summary()


features_opt = features[:,[0,2,3,6,7]]
regressor_OLS = sm.OLS(endog=labels,exog = features_opt).fit()
regressor_OLS.summary()



features_opt = features[:,[0,2,6,7]]
regressor_OLS = sm.OLS(endog=labels,exog = features_opt).fit()
regressor_OLS.summary()

feat.isnull().any()
#new features
feat = pd.DataFrame(features)
features = feat.iloc[:,[0,2,6,7]].values
"""
#Train_tst_split
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.2, random_state = 0)

#Using Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(features_train,labels_train)

labels_pred=classifier.predict(features_test)
Score=classifier.score(features_test,labels_test)


#Using KNN
from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=9,p=2)
classifier.fit(features_train,labels_train)
labels_pred=classifier.predict(features_test)
Score_knn=classifier.score(features_test,labels_test)

#fitting random forest Classifier to the training set         #best sscore=53.2   #without ols
from sklearn.ensemble import RandomForestClassifier as rf
classifier = rf(n_estimators = 50,criterion = 'entropy',random_state = 0)

classifier.fit(features_train,labels_train)

score_rf = classifier.score(features_test,labels_test)

#fitting decision tree regressor to dataset

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy',random_state=0)
classifier.fit(features_train,labels_train)


score_dt = classifier.score(features_test,labels_test)



#Building the optimal model using backward elimination
import numpy as np
import statsmodels.formula.api as sm

features = np.append(arr = np.ones((15000,1)).astype(int),values = features,axis=1)

features_opt = features[:,[0,1,2,3,4,5,6,7,8,9,10]]
regressor_OLS = sm.OLS(endog=labels,exog = features_opt).fit()
regressor_OLS.summary()


features_opt = features[:,[0,1,2,3,4,6,7,8,9,10]]
regressor_OLS = sm.OLS(endog=labels,exog = features_opt).fit()
regressor_OLS.summary()


features_opt = features[:,[0,2,3,4,6,7,8,9,10]]
regressor_OLS = sm.OLS(endog=labels,exog = features_opt).fit()
regressor_OLS.summary()


features_opt = features[:,[0,2,3,4,6,7,9,10]]
regressor_OLS = sm.OLS(endog=labels,exog = features_opt).fit()
regressor_OLS.summary()


features_opt = features[:,[0,2,3,4,6,7,9,10]]
regressor_OLS = sm.OLS(endog=labels,exog = features_opt).fit()
regressor_OLS.summary()


features_opt = features[:,[0,2,3,4,6,7,9]]
regressor_OLS = sm.OLS(endog=labels,exog = features_opt).fit()
regressor_OLS.summary()


features_opt = features[:,[0,2,3,6,7,9]]
regressor_OLS = sm.OLS(endog=labels,exog = features_opt).fit()
regressor_OLS.summary()


features_opt = features[:,[0,2,3,6,7]]
regressor_OLS = sm.OLS(endog=labels,exog = features_opt).fit()
regressor_OLS.summary()



features_opt = features[:,[0,2,6,7]]
regressor_OLS = sm.OLS(endog=labels,exog = features_opt).fit()
regressor_OLS.summary()