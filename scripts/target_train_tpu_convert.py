#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.utils import CustomObjectScope
import json

from scripts.layers import RandomLayer


def target_train_tpu_convert_main(input_file, output_weight_file, output_json_file):
    with CustomObjectScope({'RandomLayer': RandomLayer}):
        input_model = load_model(input_file)
        
        input_model.save_weights(output_weight_file)
        
        config = input_model.get_config()
        
        config['layers'][0]['config']['batch_input_shape'] = (None, None, 139)
        config['layers'][0]['config']['layer']['config']['kernel_initializer'] = 'zeros'
        config['layers'][1]['config']['kernel_initializer'] = 'zeros'
        config['layers'][4]['config']['layer']['config']['kernel_initializer'] = 'zeros'
        config['layers'][5]['config']['kernel_initializer'] = 'zeros'
        config['layers'][6]['config']['target_shape'] = (-1, 64)
        config['layers'][9]['config']['layer']['config']['kernel_initializer'] = 'zeros'
        config['layers'][10]['config']['kernel_initializer'] = 'zeros'
        config['layers'][11]['config']['target_shape'] = (-1, 32)
        config['layers'][14]['config']['layer']['config']['kernel_initializer'] = 'zeros'
        config['layers'][15]['config']['kernel_initializer'] = 'zeros'
        config['layers'][16]['config']['target_shape'] = (-1, 16)
        config['layers'][19]['config']['layer']['config']['kernel_initializer'] = 'zeros'
        config['layers'][20]['config']['kernel_initializer'] = 'zeros'
        
        model = Sequential.from_config(config)
        json_dic = json.loads(model.to_json())
        del json_dic['config']['layers'][0]['config']['layer']['config']['time_major']
        del json_dic['config']['layers'][0]['config']['layer']['config']['zero_output_for_mask']
        json_dic['config']['layers'][2]['class_name'] = 'BatchNormalization'
        json_dic['config']['layers'][2]['config']['axis'] = 2
        del json_dic['config']['layers'][4]['config']['layer']['config']['time_major']
        del json_dic['config']['layers'][4]['config']['layer']['config']['zero_output_for_mask']
        json_dic['config']['layers'][7]['class_name'] = 'BatchNormalization'
        json_dic['config']['layers'][7]['config']['axis'] = 2
        del json_dic['config']['layers'][9]['config']['layer']['config']['time_major']
        del json_dic['config']['layers'][9]['config']['layer']['config']['zero_output_for_mask']
        json_dic['config']['layers'][12]['class_name'] = 'BatchNormalization'
        json_dic['config']['layers'][12]['config']['axis'] = 2
        del json_dic['config']['layers'][14]['config']['layer']['config']['time_major']
        del json_dic['config']['layers'][14]['config']['layer']['config']['zero_output_for_mask']
        json_dic['config']['layers'][17]['class_name'] = 'BatchNormalization'
        json_dic['config']['layers'][17]['config']['axis'] = 2
        del json_dic['config']['layers'][19]['config']['layer']['config']['time_major']
        del json_dic['config']['layers'][19]['config']['layer']['config']['zero_output_for_mask']
        with open(output_json_file, 'w') as f:
            json.dump(json_dic, f)


if __name__ == '__main__':
    pass

