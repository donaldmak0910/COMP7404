from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from BoostedRandomForest import BoostedRandomForest
import pandas as pd
import numpy as np

""" Read data to be used """
# In this example, a dataset about spamming is used
data = pd.read_csv('spambase.csv',sep=",")
data.head()

""" Some preprocessing on data """
# Number of features
m = data.shape[1]
# Remove unwanted features
X = data.iloc[:,0:48]
y = data.iloc[:,(m-1):]

# Turn data into onehot format
X_onehot = pd.get_dummies(X)

""" Splitting training and testing data """
X_train, X_test, y_train, y_test = train_test_split(X_onehot, y, test_size=0.25, random_state=33)

""" Create a BRF classifier """
brf = BoostedRandomForest(verbose=True)

""" Train BRF classifier """
brf.fit(X_train, y_train)

""" Give prediction """
pred = brf.ensemble_predict(X_test)

""" Calculate accuracy """
acc = accuracy_score(y_test, pred)
print(acc)