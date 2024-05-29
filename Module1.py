import os
import pandas as pd
import sklearn.metrics as s1
import sklearn.tree as s2
import sklearn.model_selection as s3
import sklearn.naive_bayes as s4
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

a=r"C:\Users\shamb\OneDrive\Documents\Major Project - EasyMed\Databases\Training.csv"
data = pd.read_csv(a)
del data['Unnamed: 133']
data[data.isnull().any(axis=1)].head()
data = data.sample(frac = 1)
y_train = data['prognosis']
del data['prognosis']
x_train = data.copy()

xtraining,xvalid,ytraining,yvalid = s3.train_test_split(x_train,y_train,test_size=0.7)
print(len(xvalid))

data

from sklearn.linear_model import LogisticRegression
import pickle

#specifying the initial learners
model1 = s2.DecisionTreeClassifier(max_leaf_nodes=50,random_state=7)
model2 = LogisticRegression(multi_class='multinomial',solver='sag')
model3 = RandomForestClassifier(max_leaf_nodes=8, n_estimators=40,random_state=1)

#training the initial learners
model1.fit(xtraining.values,ytraining.values)
model2.fit(xtraining.values,ytraining.values)
model3.fit(xtraining.values,ytraining.values)

preds1 = model1.predict(xvalid.values)
preds2 = model2.predict(xvalid.values)
preds3 = model3.predict(xvalid.values)

predss1 = model1.predict(xtraining.values)
predss2 = model2.predict(xtraining.values)
predss3 = model3.predict(xtraining.values)

print(round(s1.accuracy_score(preds1,yvalid.values)*100,2), round(s1.accuracy_score(predss1,ytraining.values)*100,2))
print(round(s1.accuracy_score(preds2,yvalid.values)*100,2), round(s1.accuracy_score(predss2,ytraining.values)*100,2))
print(round(s1.accuracy_score(preds3,yvalid.values)*100,2), round(s1.accuracy_score(predss3,ytraining.values)*100,2))

print(len(xvalid.values[0]))

filename = 'model1.sav'
pickle.dump(model1, open(filename, 'wb'))

set(y_train)