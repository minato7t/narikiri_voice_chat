#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K


class RandomLayer(Layer):
    def __init__(self, rate, **kwargs):
        super(RandomLayer, self).__init__(**kwargs)
        self.rate = rate

    def call(self, x, training=None):
        mask = K.random_uniform(K.shape(x)[:-1], 0.0, 1.0)
        mask = K.expand_dims(mask, -1)
        mask = K.repeat_elements(mask, K.int_shape(x)[-1], -1)
        rand_x = K.switch(K.less(mask, self.rate), K.random_normal(K.shape(x), 0.0, 1.0), x)
        return K.in_train_phase(rand_x, x, training=training)

    def get_config(self):
        config = {'rate': self.rate}
        base_config = super(RandomLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
