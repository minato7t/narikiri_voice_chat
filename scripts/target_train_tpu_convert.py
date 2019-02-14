#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tensorflow.keras.models import Sequential, load_model
import json


def target_train_tpu_convert_main(input_file, output_weight_file, output_json_file):
    input_model = load_model(input_file)
    
    input_model.save_weights(output_weight_file)
    
    config = input_model.get_config()
    
    config['layers'][0]['config']['batch_input_shape'] = (None, None, 129)
    config['layers'][0]['config']['layer']['config']['kernel_initializer'] = 'zeros'
    config['layers'][1]['config']['layer']['config']['kernel_initializer'] = 'zeros'
    config['layers'][2]['config']['target_shape'] = (-1, 64)
    config['layers'][3]['config']['layer']['config']['kernel_initializer'] = 'zeros'
    config['layers'][4]['config']['target_shape'] = (-1, 32)
    config['layers'][5]['config']['layer']['config']['kernel_initializer'] = 'zeros'
    config['layers'][6]['config']['target_shape'] = (-1, 16)
    config['layers'][7]['config']['layer']['config']['kernel_initializer'] = 'zeros'
    config['layers'][8]['config']['kernel_initializer'] = 'zeros'
    
    model = Sequential.from_config(config)
    json_dic = json.loads(model.to_json())
    del json_dic['config']['layers'][0]['config']['layer']['config']['time_major']
    del json_dic['config']['layers'][0]['config']['layer']['config']['zero_output_for_mask']
    del json_dic['config']['layers'][1]['config']['layer']['config']['time_major']
    del json_dic['config']['layers'][1]['config']['layer']['config']['zero_output_for_mask']
    del json_dic['config']['layers'][3]['config']['layer']['config']['time_major']
    del json_dic['config']['layers'][3]['config']['layer']['config']['zero_output_for_mask']
    del json_dic['config']['layers'][5]['config']['layer']['config']['time_major']
    del json_dic['config']['layers'][5]['config']['layer']['config']['zero_output_for_mask']
    del json_dic['config']['layers'][7]['config']['layer']['config']['time_major']
    del json_dic['config']['layers'][7]['config']['layer']['config']['zero_output_for_mask']
    with open(output_json_file, 'w') as f:
        json.dump(json_dic, f)


if __name__ == '__main__':
    pass

