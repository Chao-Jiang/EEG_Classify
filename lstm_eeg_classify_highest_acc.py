
"""
Created on Sat Jan  7 13:27:58 2017

@author: kevinchangwang

LSTM classification of EEG sequences
"""
#import modules
from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import SimpleRNN, LSTM, GRU
from keras.optimizers import SGD, Nadam, Adam, RMSprop
from keras.utils import np_utils
import scipy.io 
import numpy as np
np.random.seed(1337)
'''
Importing data
'''
print('Loading Data...')
data = scipy.io.loadmat('/home/kevinchangwang/Downloads/sp1s_aa_1000Hz.mat')
y_test = np.loadtxt('/home/kevinchangwang/Downloads/sp1s_aa_1000Hz_test.mat')
'''
Preprocessing the data
'''
#reshape the x variables to the correct input dimensions for LSTM and convert to float 32
print('Processing Data...')
x_train = data['x_train'].reshape((316,500,28))
x_train = x_train.astype('float32')

x_test = data['x_test'].reshape((100,500,28))
x_test = x_test.astype('float32')

#reshpe  y data to the correct dimensions and convert to float 32

y_train = data['y_train'].reshape(316,1)
y_train = y_train.astype('float32')

y_test = y_test.reshape(100,1)
y_test = y_test.astype('float32')

'''
Build model
'''
print('Building Model...')
#define the model as sequential
model = Sequential()
model.add(SimpleRNN(120, input_shape=(500, 28)))
model.add(Dense(100, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(100, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(100, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation = 'sigmoid'))

model.summary() 

# Optimizer settings
optim = RMSprop(lr = 0.001, clipvalue = 0.5)

model.compile(loss = 'binary_crossentropy', optimizer = optim, metrics = ['accuracy'])

'''
Fitting the model
'''
print('Fitting the Model...')

model.fit(x_train, y_train, nb_epoch=20, batch_size=4)

print('Calculating the score...')
score, acc = model.evaluate(x_test, y_test,
                            batch_size=1)
print('Test score:', score)
print('Test accuracy:', acc)
