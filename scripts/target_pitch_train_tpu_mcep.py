#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Activation, Bidirectional, LSTM, InputLayer, Reshape, BatchNormalization, Input, Layer
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import Sequence, CustomObjectScope
import os
import glob
import struct
import numpy as np
import sys
import random
import math
import tensorflow as tf
from tensorflow.contrib.tpu.python.tpu import keras_support

from scripts.layers import RandomLayer


def pitch_loss(y_true, y_pred):
    switch_vals = K.switch(K.less(y_true, -1.0), K.zeros_like(y_true), K.ones_like(y_true))
    vals = K.mean(K.square(y_pred - y_true) * switch_vals, axis=-1)
    return vals


class MargeLayer(Layer):
    def call(self, inputs):
        PITCH_MAX = 200.0
        PITCH_MIN = 16.0
        
        pitches = inputs[1][:, ::2, :]
        pitches = (pitches + 1.0) * 0.5 * (PITCH_MAX - PITCH_MIN) + PITCH_MIN
        pitches = K.switch(K.less_equal(inputs[0][:, :, 128:129], 0.0), K.zeros_like(pitches), pitches)
        
        pitches0 = K.concatenate([K.zeros_like(pitches)[:, :1, :], pitches, K.zeros_like(pitches)[:, :1, :]], axis=1)
        pitches1 = K.concatenate([K.zeros_like(pitches)[:, :2, :], pitches], axis=1)
        pitches2 = K.concatenate([pitches, K.zeros_like(pitches)[:, :2, :]], axis=1)
        pitches_concat = K.concatenate([pitches0, pitches1, pitches2])
        pitches_concat = pitches_concat[:, 1:-1, :]
        
        return K.concatenate([inputs[0][:, :, :128], pitches_concat, inputs[0][:, :, 131:]])

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class PartsModelCheckpoint(ModelCheckpoint):
    def __init__(self, model, **kwargs):
        super(PartsModelCheckpoint, self).__init__(**kwargs)
        self.model = model
    
    def set_model(self, model):
        pass


class VoiceGeneratorTargetPitchTpu(Sequence):
    def __init__(self, dir_path, val_file, batch_size, length=None, train=True, max_size=None):
        self.length = length
        if self.length is None or self.length < 0:
            self.length = None
        self.batch_size = batch_size
        self.train = train
        
        self.index = 0
        input_files = self.get_input_files(dir_path)
        if self.train:
            self.input_files = input_files[:int(len(input_files) * (1.0 - val_file)) + 1]
            random.shuffle(self.input_files)
        else:
            self.input_files = input_files[int(len(input_files) * (1.0 - val_file)) + 1:]
        
        self.max_size = max_size
        
        self.reverse = False
        
        self.get_input_voices()

    @staticmethod
    def get_input_files(dir_path):
        input_voices_list = []
        input_voices = glob.glob(dir_path + '/*_nor.voice')
        for input_voice in input_voices:
            name, _ = os.path.splitext(input_voice)
            name = name[:-4]
            input_voices_list.append(name)
        return sorted(input_voices_list)

    def get_input_voices(self):
        MAX_SIZE = 256
        PITCH_MAX = 200.0
        PITCH_MIN = 16.0

        self.data_array = None
        self.lab_data = None
        self.lab_data_mcep = None

        data_array2 = []
        lab_data2 = []
        lab_data_mcep2 = []
        
        max_array_size = 0
        
        while (self.length is None) or (len(data_array2) < self.length):
        
            if self.index >= len(self.input_files):
                self.index = 0
                if self.train:
                    random.shuffle(self.input_files)
                if self.reverse == False:
                    self.reverse = True
                else:
                    self.reverse = False
                    if self.length is None:
                        break
            
            if self.reverse == False:
                rev_str = '_nor'
            else:
                rev_str = '_rev'
            
            input_voice = self.input_files[self.index] + rev_str + '.voice'
            
            file_data = open(input_voice, 'rb').read()
            data_array = [None for _ in range(len(file_data) // (4 * 129))]
            for loop in range(len(file_data) // (4 * 129)):
                data_array[loop] = list(struct.unpack('<128f', file_data[loop * 4 * 129:(loop + 1) * 4 * 129 - 4]))
                data_array[loop].extend([0.0 for _ in range(11)])
                if struct.unpack('<f', file_data[(loop + 1) * 4 * 129 - 4:(loop + 1) * 4 * 129])[0] > 0.0:
                    data_array[loop][128] = 1.0
                else:
                    data_array[loop][128] = -1.0
                if loop > 0:
                    if struct.unpack('<f', file_data[loop * 4 * 129 - 4:loop * 4 * 129])[0] > 0.0:
                        data_array[loop][129] = 1.0
                    else:
                        data_array[loop][129] = -1.0
                if loop + 1 < len(file_data) // (4 * 129):
                    if struct.unpack('<f', file_data[(loop + 2) * 4 * 129 - 4:(loop + 2) * 4 * 129])[0] > 0.0:
                        data_array[loop][130] = 1.0
                    else:
                        data_array[loop][130] = -1.0
            
            file_data = open(self.input_files[self.index] + rev_str + '.mcep', 'rb').read()
            for loop in range(len(file_data) // (4 * 21 * 8)):
                for loop2 in range(8):
                    if loop < len(data_array):
                        data_array[loop][131 + loop2] = list(struct.unpack('<f', file_data[(loop * 8 + loop2) * 4 * 21:(loop * 8 + loop2) * 4 * 21 + 4]))[0]
            lab_data_mcep = [None for _ in range(len(file_data) // (4 * 21 * 8) * 8)]
            for loop in range(len(file_data) // (4 * 21 * 8)):
                for loop2 in range(8):
                    lab_data_mcep[loop * 8 + loop2] = list(struct.unpack('<20f', file_data[(loop * 8 + loop2) * 4 * 21 + 4:(loop * 8 + loop2 + 1) * 4 * 21]))
            
            name, _ = os.path.splitext(os.path.basename(input_voice))
            dir_name = os.path.dirname(input_voice)
            file_data = open(dir_name + '/' + name + '.pitch', 'rb').read()
            lab_data = [None for _ in range(len(file_data) // (4 * 2) * 2)]
            for loop in range(len(file_data) // (4 * 2)):
                for loop2 in range(2):
                    lab_data[loop * 2 + loop2] = [(list(struct.unpack('<f', (file_data[(loop * 2 + loop2) * 4:(loop * 2 + loop2 + 1) * 4])))[0] - PITCH_MIN) / (PITCH_MAX - PITCH_MIN) * 2.0 - 1.0]
            
            while len(data_array) * 8 > len(lab_data_mcep) or len(data_array) * 2 > len(lab_data):
                data_array.pop()
            while len(data_array) * 8 < len(lab_data_mcep):
                for _ in range(8):
                    lab_data_mcep.pop()
            while len(data_array) * 2 < len(lab_data):
                for _ in range(2):
                    lab_data.pop()
            
            if self.max_size is None:
                if len(data_array) > MAX_SIZE:
                    max_array_size = MAX_SIZE
                elif max_array_size < len(data_array):
                    max_array_size = len(data_array)
                for loop in range(len(data_array) // MAX_SIZE + 1):
                    data_array2.append(data_array[loop * MAX_SIZE:(loop + 1) * MAX_SIZE])
                    lab_data2.append(lab_data[loop * 2 * MAX_SIZE:(loop + 1) * 2 * MAX_SIZE])
                    lab_data_mcep2.append(lab_data_mcep[loop * 8 * MAX_SIZE:(loop + 1) * 8 * MAX_SIZE])
            else:
                max_array_size = self.max_size
                for loop in range(len(data_array) // max_array_size + 1):
                    data_array2.append(data_array[loop * max_array_size:(loop + 1) * max_array_size])
                    lab_data2.append(lab_data[loop * 2 * max_array_size:(loop + 1) * 2 * max_array_size])
                    lab_data_mcep2.append(lab_data_mcep[loop * 8 * max_array_size:(loop + 1) * 8 * max_array_size])
            
            self.index += 1
        
        for loop in range(len(data_array2)):
            data_array2[loop].extend([[0.0 for _ in range(139)] for _ in range(max_array_size - len(data_array2[loop]))])

        for loop in range(len(lab_data2)):
            lab_data2[loop].extend([[0.0] for _ in range(max_array_size * 2 - len(lab_data2[loop]))])
        
        for loop in range(len(lab_data_mcep2)):
            lab_data_mcep2[loop].extend([[0.0 for _ in range(20)] for _ in range(max_array_size * 8 - len(lab_data_mcep2[loop]))])
        
        self.data_array = data_array2
        self.lab_data = lab_data2
        self.lab_data_mcep = lab_data_mcep2
        
        self.max_size = max_array_size

    def __len__(self):
        return len(self.data_array) // self.batch_size
    
    def on_epoch_end(self):
        if self.length is not None:
            self.get_input_voices()

    def __getitem__(self, idx):
        inputs = self.data_array[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_inputs = np.array(inputs, dtype='float')
        targets = self.lab_data[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_targets = np.array(targets, dtype='float')
        targets_mcep = self.lab_data_mcep[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_targets_mcep = np.array(targets_mcep, dtype='float')
        return batch_inputs, [batch_targets, batch_targets_mcep]


def target_pitch_train_tpu_main(gen_targets_dir, model_file_path, target_model_file_path, early_stopping_patience=None, length=None, batch_size=1, period=1, retrain_file=None, retrain_do_compile=False, base_model_file_path='pitch_common.h5', optimizer=tf.train.AdamOptimizer(), epochs=100000, loss_weights=[0.5, 0.5]):
    def init_uninitialized_variables():
        sess = K.get_session()
        uninitialized_variables = set([i.decode('ascii') for i in sess.run(tf.report_uninitialized_variables())])
        init_list = [v for v in tf.global_variables() if v.name.split(':')[0] in uninitialized_variables]
        init_op = tf.variables_initializer(init_list)
        sess.run(init_op)
    
    gen = VoiceGeneratorTargetPitchTpu(gen_targets_dir, 0.1, batch_size, length, train=True)
    val_gen = VoiceGeneratorTargetPitchTpu(gen_targets_dir, 0.1, batch_size, train=False, max_size=gen[0][0].shape[1])
    
    shape0 = gen[0][0].shape[1]
    
    with CustomObjectScope({'RandomLayer': RandomLayer}):
        if retrain_file is None:
            model = load_model(base_model_file_path)
            config = model.get_config()
            config['layers'][0]['config']['batch_input_shape'] = (None, shape0, 139)
            config['layers'][3]['config']['rate'] = 0.1
            config['layers'][6]['config']['target_shape'] = (shape0 * 2, 64)
            config['layers'][8]['config']['rate'] = 0.1
            model = Sequential.from_config(config)
            model.load_weights(base_model_file_path, by_name=True)
            #for layer in model.layers:
            #    layer.trainable = False
            #model.add(Class2ValueLayer(name='pitch_c2v_f'))
            model.add(Bidirectional(LSTM(8, return_sequences=True), name='pitch_blstm_a0'))
            model.add(Dense(16, name='pitch_dense_a0'))
            model.add(BatchNormalization(name='pitch_bn_a0'))
            model.add(RandomLayer(0.1, name='pitch_rand_a0'))
            model.add(Bidirectional(LSTM(8, return_sequences=True), name='pitch_blstm_a1'))
            model.add(Dense(1, name='pitch_dense_a1'))
            model.add(Activation('tanh', name='pitch_tanh_f'))
            model.summary()
            model.compile(loss='mse', optimizer=tf.train.AdamOptimizer())
        else:
            model = load_model(retrain_file)
            if retrain_do_compile:
                model.compile(loss='mse', optimizer=tf.train.AdamOptimizer())
        
        tpu_grpc_url = 'grpc://' + os.environ['COLAB_TPU_ADDR']
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(tpu_grpc_url)
        strategy = keras_support.TPUDistributionStrategy(tpu_cluster_resolver)
        model = tf.contrib.tpu.keras_to_tpu_model(model, strategy=strategy)
        init_uninitialized_variables()
        
        target_model = load_model(target_model_file_path)
        config = target_model.get_config()
        config['layers'][0]['config']['batch_input_shape'] = (None, shape0, 139)
        config['layers'][3]['config']['rate'] = 0.0
        config['layers'][6]['config']['target_shape'] = (shape0 * 2, 64)
        config['layers'][8]['config']['rate'] = 0.0
        config['layers'][11]['config']['target_shape'] = (shape0 * 4, 32)
        config['layers'][13]['config']['rate'] = 0.0
        config['layers'][16]['config']['target_shape'] = (shape0 * 8, 16)
        config['layers'][18]['config']['rate'] = 0.0
        target_model = Sequential.from_config(config)
        target_model.load_weights(target_model_file_path, by_name=True)
        for layer in target_model.layers:
            layer.trainable = False
        target_model.summary()
        target_model.compile(loss='mse', optimizer=tf.train.AdamOptimizer())
        target_model = tf.contrib.tpu.keras_to_tpu_model(target_model, strategy=strategy)
        init_uninitialized_variables()
        
        input = Input(shape=(shape0, 139))
        output0 = model(input)
        validity = MargeLayer()([input, output0])
        output1 = target_model(validity)
        combined_model = Model(input, [output0, output1])
        combined_model.summary()
        combined_model.compile(loss=[pitch_loss, 'mse'], loss_weights=loss_weights, optimizer=optimizer)
        
        cp = PartsModelCheckpoint(model, filepath=model_file_path, monitor='val_loss', save_best_only=True, period=period)
        
        if early_stopping_patience is not None:
            es = EarlyStopping(monitor='val_loss', patience=early_stopping_patience, verbose=0, mode='auto')
            callbacks = [es, cp]
        else:
            callbacks = [cp]

        combined_model.fit_generator(gen,
            shuffle=True,
            epochs=epochs,
            verbose=1,
            callbacks=callbacks,
            validation_data=val_gen
        )


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('arg error')
        exit()

    early_stopping_patience = None
    if len(sys.argv) >= 4 and int(sys.argv[3]) >= 0:
        early_stopping_patience = int(sys.argv[3])
    
    length = None
    if len(sys.argv) >= 5 and int(sys.argv[4]) >= 0:
        length = int(sys.argv[4])
    
    batch_size = 1
    if len(sys.argv) >= 6 and int(sys.argv[5]) >= 1:
        batch_size = int(sys.argv[5])
    
    target_pitch_train_main(sys.argv[1], sys.argv[2], early_stopping_patience, length, batch_size)
    

