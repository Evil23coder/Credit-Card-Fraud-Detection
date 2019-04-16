import pickle

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
data=pd.read_csv("G:\\Credit_Card_Fraud_Detection\\resources\\dataset\\v1.0\\dataset_v1.0.csv")
target=data['isFraud']
target=np.array(target)
del data['isFraud']
train=data
train=np.array(train)
print("Train:",train)
X_train,X_test,y_train,y_test=train_test_split(train,target,random_state=7)
print("Shape of Training Features:",X_train.shape)
print("Shape of Testing Features:",X_test.shape)
print("Shape of Training Target:",y_train.shape)
print("Shape of Testing Target:",y_test.shape)
logistic_regression=LogisticRegression()
print("Training the Model")
logistic_regression.fit(X_train,y_train)
print("Testing the model")
prediction=logistic_regression.predict(X_test)
print("Accuracy of model is ",np.mean(prediction==y_test))
FILENAME='credit_model_logreg99.sav'
pickle.dump(logistic_regression,open(FILENAME,'wb'))
