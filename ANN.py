# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 14:51:09 2022

@author: tom97
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("Churn_Modelling.csv")

X = data.iloc[:,3:13]
Y  = data.iloc[:,-1]

geography = pd.get_dummies(X["Geography"],drop_first =True)
gender = pd.get_dummies(X["Gender"],drop_first =True)

X = pd.concat([X,geography,gender], axis = 1)
X = X.drop(["Geography","Gender"],axis = 1)


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,random_state = 0)


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU,PReLU,ELU
from keras.layers import Dropout

model = Sequential()

#first layer
model.add(Dense(units = 6,kernel_initializer = 'he_uniform',activation = 'relu',input_dim = 11))
#second layer
model.add(Dense(units = 6,kernel_initializer = 'he_uniform',activation = 'relu'))
#output layer
model.add(Dense(units = 1,kernel_initializer = 'glorot_uniform',activation = 'sigmoid'))


model.compile(optimizer = 'Adamax',loss = 'binary_crossentropy',metrics = ["accuracy"])

model_fit = model.fit(X_train,Y_train,validation_split = 0.33,batch_size = 10,epochs= 100)

# summarize history for accuracy
plt.plot(model_fit.history['accuracy'])
plt.plot(model_fit.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# Test Validations
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix,accuracy_score
matrix = confusion_matrix(Y_test, y_pred)
score = accuracy_score(y_pred,Y_test)


# summarize history for loss
plt.plot(model_fit.history['loss'])
plt.plot(model_fit.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
