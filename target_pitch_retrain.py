#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tensorflow.python.keras.models import load_model
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import sys

from target_train import DoubleRelu
from target_pitch_train import VoiceGeneratorTargetPitch


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('arg error')
        exit()

    model = load_model(sys.argv[1], custom_objects={'DoubleRelu': DoubleRelu})

    cp = ModelCheckpoint(filepath=sys.argv[3], monitor='val_loss', save_best_only=True)
    
    if len(sys.argv) >= 5 and int(sys.argv[4]) >= 0:
        es = EarlyStopping(monitor='val_loss', patience=int(sys.argv[4]), verbose=0, mode='auto')
        callbacks = [es, cp]
    else:
        callbacks = [cp]

    if len(sys.argv) >= 6 and int(sys.argv[5]) >= 0:
        length = int(sys.argv[5])
    else:
        length = None

    if len(sys.argv) >= 7:
        batch_size = int(sys.argv[6])
    
    do_compile = False
    if len(sys.argv) >= 8:
        if sys.argv[7] != 'False' and sys.argv[7] != 'false':
            try:
                do_compile = int(sys.argv[7]) > 0
            except ValueError:
                do_compile = True
    
    if do_compile:
        model.compile(loss='mean_squared_error', optimizer='adam')
    
    model.fit_generator(VoiceGeneratorTargetPitch(sys.argv[2], 0.1, batch_size, length, train=True),
        shuffle=True,
        epochs=100000,
        verbose=1,
        callbacks=callbacks,
        validation_data=VoiceGeneratorTargetPitch(sys.argv[2], 0.1, batch_size, train=False)
    )
