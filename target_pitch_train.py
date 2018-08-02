#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Bidirectional, LSTM, BatchNormalization, Input, Layer, concatenate, InputLayer
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.utils import Sequence
import os
import glob
import struct
import numpy as np
import sys
import random
import math

from target_train import DoubleRelu


class VoiceGeneratorTargetPitch(Sequence):
    def __init__(self, dir_path, val_file, batch_size, length=None, train=True):
        self.length = length
        self.batch_size = batch_size
        self.train = train
        
        self.index = 0
        input_files = self.get_input_files(dir_path)
        if self.train:
            self.input_files = input_files[:int(len(input_files) * (1.0 - val_file)) + 1]
            random.shuffle(self.input_files)
        else:
            self.input_files = input_files[int(len(input_files) * (1.0 - val_file)) + 1:]
        
        self.get_input_voices()

    @staticmethod
    def get_input_files(dir_path):
        input_voices_list = []
        input_voices = glob.glob(dir_path + '/*.voice')
        return sorted(input_voices)

    def get_input_voices(self):
        MAX_SIZE = 512

        self.data_array = None
        self.lab_data = None

        data_array2 = []
        lab_data2 = []
        
        max_array_size = 0
        
        while (self.length is None) or (len(data_array2) < self.length):
        
            if self.index >= len(self.input_files):
                self.index = 0
                if self.train:
                    random.shuffle(self.input_files)
                if self.length is None:
                    break
            input_voice = self.input_files[self.index]
            
            data_array = []
            lab_data = []
            file_data = open(input_voice, 'rb').read()
            for loop in range(len(file_data) // (4 * 257)):
                data_array.append(list(struct.unpack('<256f', file_data[loop * 4 * 257:(loop + 1) * 4 * 257 - 4])))
                lab_data.append(list(struct.unpack('<f', file_data[(loop + 1) * 4 * 257 - 4:(loop + 1) * 4 * 257])))
            
            if len(data_array) > MAX_SIZE:
                max_array_size = MAX_SIZE
            elif max_array_size < len(data_array):
                max_array_size = len(data_array)
            
            for loop in range(len(data_array) // MAX_SIZE + 1):
                data_array2.append(data_array[loop * MAX_SIZE:(loop + 1) * MAX_SIZE])
                lab_data2.append(lab_data[loop * MAX_SIZE:(loop + 1) * MAX_SIZE])
            
            self.index += 1
        
        for loop in range(len(data_array2)):
            for loop2 in range(max_array_size - len(data_array2[loop])):
                data_array2[loop].append([0.0] * 256)

        for loop in range(len(lab_data2)):
            for loop2 in range(max_array_size - len(lab_data2[loop])):
                lab_data2[loop].append([0.0])
        
        self.data_array = data_array2
        self.lab_data = lab_data2

    def __len__(self):
        return int(math.ceil(len(self.data_array) / self.batch_size))
    
    def on_epoch_end(self):
        if self.length is not None:
            self.get_input_voices()

    def __getitem__(self, idx):
        inputs = self.data_array[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_inputs = np.array(inputs, dtype='float64')
        targets = self.lab_data[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_targets = np.array(targets, dtype='float64')
        return batch_inputs, batch_targets


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('arg error')
        exit()

    model = Sequential(name='pitch_model')
    model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=(None, 256), name='pitch_blstm0'))
    model.add(Dropout(0.3, name='pitch_dropout0'))
    for loop in range(10):
        model.add(Dense(256, name='pitch_dense' + str(loop)))
        model.add(BatchNormalization(name='pitch_bn' + str(loop)))
        model.add(DoubleRelu(name='pitch_dr' + str(loop)))
    model.add(Dense(128, name='pitch_dense10'))
    model.add(BatchNormalization(name='pitch_bn10'))
    model.add(DoubleRelu(name='pitch_dr10'))
    model.add(Dense(64, name='pitch_dense11'))
    model.add(BatchNormalization(name='pitch_bn11'))
    model.add(DoubleRelu(name='pitch_dr11'))
    model.add(Dense(32, name='pitch_dense12'))
    model.add(BatchNormalization(name='pitch_bn12'))
    model.add(DoubleRelu(name='pitch_dr12'))
    model.add(Dense(16, name='pitch_dense13'))
    model.add(BatchNormalization(name='pitch_bn13'))
    model.add(DoubleRelu(name='pitch_dr13'))
    model.add(Dense(8, name='pitch_dense14'))
    model.add(BatchNormalization(name='pitch_bn14'))
    model.add(DoubleRelu(name='pitch_dr14'))
    for loop in range(1):
        model.add(Bidirectional(LSTM(4, return_sequences=True), name='pitch_blstm_f' + str(loop)))
        model.add(Dropout(0.3, name='pitch_dropout_f' + str(loop)))
    model.add(Dense(1, name='pitch_dense_f'))
    model.add(Activation('relu', name='pitch_relu_f'))
    model.summary()
    
    model.compile(loss='mean_squared_error', optimizer='adam')

    cp = ModelCheckpoint(filepath=sys.argv[2], monitor='val_loss', save_best_only=True)
    
    if len(sys.argv) >= 4 and int(sys.argv[3]) >= 0:
        es = EarlyStopping(monitor='val_loss', patience=int(sys.argv[3]), verbose=0, mode='auto')
        callbacks = [es, cp]
    else:
        callbacks = [cp]

    if len(sys.argv) >= 5 and int(sys.argv[4]) >= 0:
        length = int(sys.argv[4])
    else:
        length = None

    if len(sys.argv) >= 6:
        batch_size = int(sys.argv[5])
    
    model.fit_generator(VoiceGeneratorTargetPitch(sys.argv[1], 0.1, batch_size, length, train=True),
        shuffle=True,
        epochs=100000,
        verbose=1,
        callbacks=callbacks,
        validation_data=VoiceGeneratorTargetPitch(sys.argv[1], 0.1, batch_size, train=False)
    )
