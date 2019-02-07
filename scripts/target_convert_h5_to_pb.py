#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from keras import backend as K
import os
import sys
from keras.models import model_from_json


def convert_h5_to_pb_main(weight_file_path, json_file_path, pb_file_path, output_layer_name):
    graph = tf.Graph()
    with graph.as_default(), tf.Session() as sess:
        K.set_session(sess)

        json_string = open(json_file_path, 'r').read()
        
        model = model_from_json(json_string)
        model.load_weights(weight_file_path, by_name=True)
        
        converted_graph = tf.graph_util.convert_variables_to_constants(sess, graph.as_graph_def(), [output_layer_name])
        dir_name, file_name = os.path.split(pb_file_path)
        tf.train.write_graph(converted_graph, dir_name, file_name, as_text=False)


if __name__ == '__main__':
    convert_h5_to_pb_main(sys.argv[1], sys.argv[2], 'target_dense_f/BiasAdd')
