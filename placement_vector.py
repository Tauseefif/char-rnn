#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 17:40:07 2019

@author: ti
"""
import numpy as np
import itertools
import tensorflow as tf
from tensorflow import keras
num_Patches = 5
# Create data
data = np.array(list(itertools.permutations(np.arange(0,num_Patches,1))))

X, y = data[:,:-1], data[:,-1]

# Convert into one hot encoding

data = [tf.keras.utils.to_categorical(x, num_classes=num_Patches) for x in X]
X = np.array(data)
y = tf.keras.utils.to_categorical(y, num_classes=num_Patches)

# define model
model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(75, input_shape=(X.shape[1], X.shape[2])))
model.add(tf.keras.layers.Dense(num_Patches, activation='softmax'))
print(model.summary())

	
# compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit model
model.fit(X, y, epochs=100, verbose=2)


# save the model to file
model.save('model.h5')
A = list([2,3,4,0])
A = [tf.keras.utils.to_categorical(x, num_classes=num_Patches) for x in A]
A = np.array(A)
# predict character
yhat = model.predict_classes(A, verbose=0)
print("yhat is",yhat)