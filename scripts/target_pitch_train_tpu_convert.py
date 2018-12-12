#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tensorflow.keras.models import Sequential, load_model


def target_pitch_train_tpu_convert_main(input_file, output_file):
    input_model = load_model(input_file)
    
    config = input_model.get_config()
    
    config['layers'][0]['config']['batch_input_shape'] = (None, None, 128)
    config['layers'][0]['config']['layer']['config']['kernel_initializer'] = 'zeros'
    config['layers'][1]['config']['target_shape'] = (-1, 64)
    config['layers'][2]['config']['layer']['config']['kernel_initializer'] = 'zeros'
    config['layers'][3]['config']['target_shape'] = (-1, 32)
    config['layers'][4]['config']['layer']['config']['kernel_initializer'] = 'zeros'
    config['layers'][5]['config']['target_shape'] = (-1, 16)
    config['layers'][6]['config']['layer']['config']['kernel_initializer'] = 'zeros'
    config['layers'][7]['config']['target_shape'] = (-1, 64)
    config['layers'][8]['config']['kernel_initializer'] = 'zeros'
    
    model = Sequential.from_config(config)
    
    model.set_weights(input_model.get_weights())
    
    model.save(output_file)


if __name__ == '__main__':
    pass

