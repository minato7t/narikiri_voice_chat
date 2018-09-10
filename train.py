#!/usr/bin/env python
# -*- coding: utf-8 -*-


import sys
import os
import shutil

from scripts.make_nvm import make_nvm_main
from scripts.gen_targets import gen_targets_main
from scripts.target_train import target_train_main
from scripts.target_pitch_train import target_pitch_train_main
from scripts.target_convert_h5_to_pb import convert_h5_to_pb_main
from scripts.make_nvz import make_nvz_main


def train_main(targets_dir='targets', nvz_file='outputs/target.nvz', retrain_flag=False, outputs_dir='outputs'):
    retrain_step = 0
    if retrain_flag and os.path.isfile(outputs_dir + '/retrain.dat'):
        retrain_step = int(open(outputs_dir + '/retrain.dat', 'r').read())
    else:
        if os.path.isdir(outputs_dir):
            shutil.rmtree(outputs_dir)
        if os.path.isdir('gen_targets'):
            shutil.rmtree('gen_targets')

    if retrain_step < 1:
        make_nvm_main(targets_dir, outputs_dir + '/target.nvm')
        open(outputs_dir + '/retrain.dat', 'w').write('1')

    if retrain_step < 2:
        gen_targets_main(targets_dir, 'gen_targets')
        open(outputs_dir + '/retrain.dat', 'w').write('2')

    if retrain_step < 3:
        if not os.path.isfile(outputs_dir + '/target.h5'):
            target_train_main('gen_targets', outputs_dir + '/target.h5', 20, -1, 32)
        else:
            target_train_main('gen_targets', outputs_dir + '/target.h5', 20, -1, 32, outputs_dir + '/target.h5')
        open(outputs_dir + '/retrain.dat', 'w').write('3')

    if retrain_step < 4:
        if not os.path.isfile(outputs_dir + '/pitch.h5'):
            target_pitch_train_main('gen_targets', outputs_dir + '/pitch.h5', 20, -1, 32)
        else:
            target_pitch_train_main('gen_targets', outputs_dir + '/pitch.h5', 20, -1, 32, outputs_dir + '/pitch.h5')
        open(outputs_dir + '/retrain.dat', 'w').write('4')

    if retrain_step < 5:
        convert_h5_to_pb_main(outputs_dir + '/target.h5', outputs_dir + '/target.pb', 'target_dense_f/BiasAdd')
        open(outputs_dir + '/retrain.dat', 'w').write('5')

    if retrain_step < 6:
        convert_h5_to_pb_main(outputs_dir + '/pitch.h5', outputs_dir + '/pitch.pb', 'pitch_relu_f/Relu')
        open(outputs_dir + '/retrain.dat', 'w').write('6')

    if retrain_step < 7:
        make_nvz_main(nvz_file, outputs_dir + '/target.nvm', outputs_dir + '/target.pb', outputs_dir + '/pitch.pb')
        open(outputs_dir + '/retrain.dat', 'w').write('7')


if __name__ == '__main__':
    targets_dir = 'targets'
    if len(sys.argv) > 1:
        targets_dir = sys.argv[1]
    nvz_file = 'outputs/target.nvz'
    if len(sys.argv) > 2:
        nvz_file = sys.argv[2]
    retrain_flag = False
    if len(sys.argv) > 3:
        if sys.argv[3] != 'False' and sys.argv[3] != 'false' and sys.argv[3] != '0':
            retrain_flag = True
    outputs_dir = 'outputs'
    if len(sys.argv) > 4:
        outputs_dir = sys.argv[4]

    train_main(targets_dir, nvz_file, retrain_flag, outputs_dir)
