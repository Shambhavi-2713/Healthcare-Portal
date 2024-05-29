import os
import pandas as pd
import sklearn.metrics as s1
import sklearn.tree as s2
import sklearn.model_selection as s3 
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

a = r"C:\Users\shamb\OneDrive\Documents\Major Project - EasyMed\Databases\diabetes.csv"
data = pd.read_csv(a)
data[data.isnull().any(axis=1)].head()
train, test = s3.train_test_split(data, test_size=0.2)

y_train = train['Outcome']
del train['Outcome']
x_train = train.copy()
y_test = test['Outcome']
del test['Outcome']
x_test = test.copy()

xtraining,xvalid,ytraining,yvalid = s3.train_test_split(x_train,y_train,test_size=0.5)

#specifying the initial learners
model1 = RandomForestClassifier(max_leaf_nodes=5, n_estimators=200)
model2 = LogisticRegression(max_iter=200)
model3 = s2.DecisionTreeClassifier(max_leaf_nodes=10,random_state=5)

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

print(s1.accuracy_score(preds1,yvalid.values)*100)
print(s1.accuracy_score(preds2,yvalid.values)*100)
print(s1.accuracy_score(preds3,yvalid.values)*100)
print(s1.accuracy_score(predss1,ytraining.values)*100)
print(s1.accuracy_score(predss2,ytraining.values)*100)
print(s1.accuracy_score(predss3,ytraining.values)*100)

filename = 'model1.sav'
pickle.dump(model1, open(filename, 'wb'))

filename = 'model2.sav'
pickle.dump(model2, open(filename, 'wb'))
filename = 'model3.sav'
pickle.dump(model3, open(filename, 'wb'))

test_preds1 = model1.predict(x_test.values)
test_preds2 = model2.predict(x_test.values)
test_preds3 = model3.predict(x_test.values)

#making a new dataset for training our final model by stacking the predictions on the validation data
train_stack = np.column_stack((preds1,preds2,preds3))

#making the final test set for our final model by stacking the predictions on the test data
test_stack = np.column_stack((test_preds1,test_preds2,test_preds3))

final_model = RandomForestClassifier(max_leaf_nodes=5, n_estimators=195)

#training the final model on the stacked predictions
final_model.fit(train_stack,yvalid.values)

final_predictions = final_model.predict(test_stack)
s1.accuracy_score(y_test,final_predictions)*100

final_predictions = final_model.predict(train_stack)
s1.accuracy_score(yvalid,final_predictions)*100

filename = 'model_final.sav'
pickle.dump(final_model, open(filename, 'wb'))