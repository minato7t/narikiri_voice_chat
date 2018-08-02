#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras import backend as K
import os
import sys

from target_train import DoubleRelu


if __name__ == '__main__':
    graph = tf.Graph()
    with graph.as_default(), tf.Session() as sess:
        K.set_session(sess)
        
        model = load_model(sys.argv[1], custom_objects={'DoubleRelu': DoubleRelu})
        converted_graph = tf.graph_util.convert_variables_to_constants(sess, graph.as_graph_def(), ['pitch_relu_f/Relu'])
        dir_name, file_name = os.path.split(sys.argv[2])
        tf.train.write_graph(converted_graph, dir_name, file_name, as_text=False)
