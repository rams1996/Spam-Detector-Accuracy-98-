# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 11:50:07 2018

@author: ramse
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB 
df=pd.read_csv('spam.csv',encoding='latin-1')
df=df.iloc[:,0:2]
df.columns=['RESULT','MESSAGE']
print(len(df[df.RESULT=='spam']))
df.loc[df["RESULT"]=='spam', "RESULT"]=0
df.loc[df["RESULT"]=='ham', "RESULT"]=1
df_x=df["MESSAGE"]
df_y=df["RESULT"]
cv1=TfidfVectorizer(stop_words='english')
x_train,x_test,y_train,y_test=train_test_split(df_x,df_y,test_size=0.2,random_state=1)
x_traincv=cv1.fit_transform(x_train)
a=x_traincv.toarray()
mnb=MultinomialNB()
y_train=y_train.astype('int')
mnb.fit(x_traincv,y_train)
x_testcv=cv1.transform(x_test)
pred=mnb.predict(x_testcv)
actual=np.array(y_test)
def check_accuracy(actual,pred):
    count=0
    for i in range(len(pred)):
        if pred[i]==actual[i]:
            count=count+1
    accuracy=count/len(actual)
    return accuracy
            
print(df.head())
print(check_accuracy(actual,pred))
"""ACCURACY WAS 98%"""