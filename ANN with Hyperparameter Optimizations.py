# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 18:32:38 2022

@author: tom97
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("Data/Churn_Modelling.csv")

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


from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense,Activation,Embedding,Flatten,LeakyReLU,BatchNormalization,Dropout
from keras.activations import relu,sigmoid


def cr_model(layers,activation):
    model = Sequential()
    for i, nodes in enumerate(layers):
        if i == 0:
            model.add(Dense(nodes,input_dim = X_train.shape[1]))
            model.add(Activation(activation))
            model.add(Dropout(0.3))
        else:
            model.add(Dense(nodes))
            model.add(Activation(activation))
            model.add(Dropout(0.3))

    model.add(Dense(units = 1, kernel_initializer = 'glorot_uniform',activation = 'sigmoid'))

    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model


model = KerasClassifier(build_fn = cr_model, verbose = 0)


layers = [[20], [40,20], [45,30,15]]
activations = ['sigmoid','relu']
param_grid = dict(layers = layers, activation = activations, batch_size = [128,256],epochs = [30])
grid = GridSearchCV(estimator = model, param_grid = param_grid, cv = 5)


grid_result = grid.fit(X_train, Y_train)

[grid_result.best_score_,grid_result.best_params_]


