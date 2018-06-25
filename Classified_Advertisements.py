# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 20:44:01 2018

@author: Ayushi
"""
#Read training data
with open('Advertisement_training_data.json', 'r') as myfile:
    #data=myfile.read().replace('}', '},') 
    strs = myfile.read()
    count = strs.count("}") - 1
    strs = strs.replace('}', '},', count)
strs = strs.strip("ï»¿")
strs = '['+strs+']'
with open("Output.json", "w") as text_file:
    text_file.write(strs)
    
#import json

#file = open(filename, encoding="utf8")
#read testing data
with open('Advertisement_test_data.json', 'r', encoding="utf8") as myfile:
    #data=myfile.read().replace('}', '},') 
    strs = myfile.read()
    count = strs.count("}") - 1
    strs = strs.replace('}', '},', count)
strs = strs.strip("ï»¿")
strs = '['+strs+']'
strs = strs.strip("ï»¿")

with open("Output2.json", "w", encoding="utf8") as text_file:
    strs = strs.replace("ï»¿","")
    text_file.write(strs)
    
    
#open Output2.json and delete "i>>?"

#read files
    
import pandas as pd
df_train = pd.read_json("Output.json")

df_test = pd.read_json("Output2.json")


#df_test = json.loads("text_file")

#visualize training data
#label encoding
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df_train["category"] = encoder.fit_transform(df_train["category"])


#split into training and test sets
labels_train1 = df_train.iloc[:,0:1]
features_train1 = df_train.iloc[:,1:]

#one hot encoding
features_train1 = pd.get_dummies(features_train1, columns=['city'])

features_train1 = pd.get_dummies(features_train1, columns=['section'])


#NLP
# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 20217):
    heading = re.sub('[^a-zA-Z0-9]', ' ', features_train1['heading'][i])
    heading = heading.lower()
    heading = heading.split()
    ps = PorterStemmer()
    heading = [ps.stem(word) for word in heading if not word in set(stopwords.words('english'))]
    heading = ' '.join(heading)
    corpus.append(heading)

# Creating the Bag of Words model  
from sklearn.feature_extraction.text import CountVectorizer
#cv = CountVectorizer()
cv = CountVectorizer(max_features = 8000)
features1 = cv.fit_transform(corpus).toarray()
df = pd.DataFrame(features1)
df1 = features_train1.iloc[:,1:]
new_fea = pd.concat([df1,df],axis=1)
features_train = new_fea.iloc[:,:].values

labels_train = labels_train1.iloc[:,:].values


#fitting the logistic regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(features_train,labels_train)
score = classifier.score(features_train,labels_train)


#Visualize testing data
df_test['category']=0


#split into training and test sets
labels_test1 = df_test.iloc[:,-1:]
features_test1 = df_test.iloc[:,0:-1]

#one hot encoding
features_test1 = pd.get_dummies(features_test1, columns=['city'])

features_test1 = pd.get_dummies(features_test1, columns=['section'])


#NLP
# Cleaning the texts
corpus1 = []
for i in range(0, 15370):
    heading = re.sub('[^a-zA-Z0-9]', ' ', features_test1['heading'][i])
    heading = heading.lower()
    heading = heading.split()
    ps = PorterStemmer()
    heading = [ps.stem(word) for word in heading if not word in set(stopwords.words('english'))]
    heading = ' '.join(heading)
    corpus1.append(heading)

# Creating the Bag of Words model
#cv1 = CountVectorizer()
cv1 = CountVectorizer(max_features = 8000)

features2 = cv1.fit_transform(corpus1).toarray()
df2 = pd.DataFrame(features2)
df3 = features_test1.iloc[:,1:]
new_fea2 = pd.concat([df3,df2],axis=1)
features_test = new_fea2.iloc[:,:].values

labels_test = labels_test1.iloc[:,:].values

#pred the logistic regression
labels_pred = classifier.predict(features_test)
labels_test = labels_pred
labels_test = labels_pred.reshape(-1,1)
most  = labels_test.mode()
lab = pd.DataFrame(labels_test)

#showing the top five categories
k=lab[0].value_counts()[0:5]



