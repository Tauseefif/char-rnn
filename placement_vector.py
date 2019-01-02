#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 17:40:07 2019

@author: ti
"""
import numpy as np
import itertools
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import load_model
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
num_Patches = 5
# Create data
data = np.array(list(itertools.permutations(np.arange(0,num_Patches,1))))

X, y = data[:,:-1], data[:,-1]

# Convert into one hot encoding

data = [to_categorical(x, num_classes=num_Patches) for x in X]
X = np.array(data)
y = to_categorical(y, num_classes=num_Patches)

# define model
model = Sequential()
model.add(LSTM(75, input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(num_Patches, activation='softmax'))
print(model.summary())

	
# compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit model
model.fit(X, y, epochs=100, verbose=2)


# save the model to file
model.save('model.h5')
A = list([2,3,4,0])
# predict character
def generate_seq(model, seq_length, seed_vec, n):
	in_vec = seed_vec
	# generate a fixed number of characters
	for _ in range(n):
         A = pad_sequences([in_vec], maxlen=seq_length, truncating='pre')
         # one hot encode
         A = to_categorical(A, num_classes=num_Patches)
         # predict character
         yhat = model.predict_classes(A, verbose=0)
         # reverse map integer to character
         # append to input
         in_vec.append(yhat[0])
	return in_vec

print(generate_seq(model, 4, A, 10))