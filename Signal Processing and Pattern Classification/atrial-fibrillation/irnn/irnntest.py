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
from keras.initializations import normal, identity
from keras.optimizers import RMSprop


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
model.add(SimpleRNN(4,init=lambda shape, name: normal(shape, scale=0.001, name=name),inner_init=lambda shape, name: identity(shape, scale=1.0, name=name),input_dim=15000,activation='relu',))
model.add(Dropout(0.1))
model.add(Dense(1))
model.add(Activation('sigmoid'))

import os
for file in os.listdir("logs/irnn1layer/"):
  model.load_weights("logs/irnn1layer/"+file)
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

