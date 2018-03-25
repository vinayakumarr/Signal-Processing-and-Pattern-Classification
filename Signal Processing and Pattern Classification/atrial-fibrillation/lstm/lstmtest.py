from __future__ import print_function
from sklearn.cross_validation import train_test_split
import pandas as pd
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM, SimpleRNN, GRU
from keras.datasets import imdb
from keras.utils.np_utils import to_categorical
from sklearn.metrics import (precision_score, recall_score,
                             f1_score, accuracy_score,mean_squared_error,mean_absolute_error)
from sklearn import metrics
from sklearn.preprocessing import Normalizer
import h5py
from keras import callbacks


testdata = pd.read_csv('dataset/testdata.csv', header=None)


C = testdata.iloc[:,0]
T = testdata.iloc[:,1:15002]

scaler = Normalizer().fit(T)
testT = scaler.transform(T)

y_test = np.array(C)


X_test = np.reshape(testT, (testT.shape[0], 1, testT.shape[1]))


batch_size = 2

# 1. define the network
model = Sequential()
model.add(LSTM(32,input_dim=15000))  # try using a GRU instead, for fun
model.add(Dropout(0.1))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

from sklearn.metrics import confusion_matrix
model.load_weights("logs/lstm1layer/checkpoint-07.hdf5")
y_pred = model.predict_classes(X_test)
print(y_pred)
print(confusion_matrix(y_test, y_pred))
loss, accuracy = model.evaluate(X_test, y_test)
print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))
y_prob = model.predict_proba(X_test)
np.savetxt("gru.txt", y_prob)


import os
for file in os.listdir("logs/lstm1layer/"):
  model.load_weights("logs/lstm1layer/"+file)
  y_pred = model.predict_classes(X_test)
  model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
  loss, accuracy = model.evaluate(X_test, y_test)
  print(file)
  print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))
  print("---------------------------------------------------------------------------------")
  accuracy = accuracy_score(y_test, y_pred)
  recall = recall_score(y_test, y_pred , average="binary")
  precision = precision_score(y_test, y_pred , average="binary")
  f1 = f1_score(y_test, y_pred, average="binary")

  print("accuracy")
  print("%.3f" %accuracy)
  print("precision")
  print("%.3f" %precision)
  print("recall")
  print("%.3f" %recall)
  print("f1score")
  print("%.3f" %f1)


