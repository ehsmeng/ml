#!/usr/local/bin/python2.7 -u

'''
export THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32
'''

import math
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
import matplotlib.pyplot as plt

class Deep_Autoencoder_Pretrain:
    def __init__(self, train_data, hidden_sizes):
        self.m_train_data = train_data
        self.m_hidden_sizes = hidden_sizes
        self.m_chatty = False

    def train_no_pretrain(self):
        if self.m_chatty:
            print "Create model"
        self.m_keras_model = Sequential()
        sh = np.shape(train_data)[1]
        self.m_keras_model.add(Dense(self.m_hidden_sizes[0], input_dim=sh))
        self.m_keras_model.add(Activation('sigmoid'))
        for ldim in self.m_hidden_sizes[1:]:
            self.m_keras_model.add(Dense(ldim))
            self.m_keras_model.add(Activation('sigmoid'))
        for ldim in (self.m_hidden_sizes[::-1])[1:]:
            self.m_keras_model.add(Dense(ldim))
            self.m_keras_model.add(Activation('sigmoid'))
        self.m_keras_model.add(Dense(sh))
        self.m_keras_model.add(Activation('sigmoid'))
        if self.m_chatty:
            print "Compiling model"
        self.m_keras_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
        if self.m_chatty:
            print "Training model"
        verbose = 1 if self.m_chatty else 0
        self.m_keras_model.fit(self.m_train_data, self.m_train_data, nb_epoch=2, batch_size=8, verbose=verbose)
        self.m_keras_model.fit(self.m_train_data, self.m_train_data, nb_epoch=2, batch_size=16, verbose=verbose)
        self.m_keras_model.fit(self.m_train_data, self.m_train_data, nb_epoch=4, batch_size=32, verbose=verbose)
        self.m_keras_model.fit(self.m_train_data, self.m_train_data, nb_epoch=8, batch_size=64, verbose=verbose)
        self.m_keras_model.fit(self.m_train_data, self.m_train_data, nb_epoch=8, batch_size=128, verbose=verbose)
        self.m_keras_model.fit(self.m_train_data, self.m_train_data, nb_epoch=8, batch_size=256, verbose=verbose)
        self.m_keras_model.fit(self.m_train_data, self.m_train_data, nb_epoch=20, batch_size=512, verbose=verbose)

if __name__ == "__main__":
    print "Creating nonsense train data"
    train_data = []
    for i in range(10000):
        a = []
        for j in range(100):
            a.append(math.sin(0.03 * j))
        train_data.append(a)
    train_data = np.array(train_data)

    au = Deep_Autoencoder_Pretrain(train_data, [50, 25, 10])
    au.m_chatty = True
    au.train_no_pretrain()

    print "READY."
