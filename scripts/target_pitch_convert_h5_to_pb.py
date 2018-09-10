#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

from target_convert_h5_to_pb import convert_h5_to_pb_main


if __name__ == '__main__':
    convert_h5_to_pb_main(sys.argv[1], sys.argv[2], 'pitch_relu_f/Relu')

