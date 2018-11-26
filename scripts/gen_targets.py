#!/usr/bin/env python
# -*- coding: utf-8 -*-

from keras.models import load_model
from keras import backend as K
from keras.layers import concatenate, Layer
import os
import glob
import shutil
import subprocess
import struct
import numpy as np
import sys


def gen_targets_main(input_voices_dir='targets', gen_dir_name='gen_targets', zip_file_name=None):
    zip_name = None
    if zip_file_name is not None:
        zip_name, _ = os.path.splitext(zip_file_name)

    input_voices = glob.glob(input_voices_dir + '/**/*', recursive=True)
    del_files = glob.glob(input_voices_dir + '/**/.*', recursive=True)
    for input_voice in input_voices:
        if os.path.isdir(input_voice):
            del_files.append(input_voice)
    for del_file in del_files:
        input_voices.remove(del_file)

    if os.path.isdir('tmp') == False:
        os.mkdir('tmp')
    if os.path.isdir(gen_dir_name) == False:
        os.mkdir(gen_dir_name)
    
    model_voice = load_model('voice.h5')
    while model_voice.layers[-1].name != 'voice_blstm_out':
        model_voice.pop()
    
    for input_voice in input_voices:
        
        name, _ = os.path.splitext(input_voice)
        name = ('_' + name).replace('/', '_').replace('\\', '_')
        
        if os.name == 'nt':
            subprocess.call('sox "' + input_voice + '" -e signed-integer -c 1 -b 16 -r 16000 -L tmp\\tmp.raw', shell=True)
        else:
            subprocess.call('sox "' + input_voice + '" -e signed-integer -c 1 -b 16 -r 16000 -L tmp/tmp.raw', shell=True)
        for cut_loop in range(16):
            if os.name == 'nt':
                subprocess.call('bin\\sptk\\x2x +sf < tmp\\tmp.raw | bin\\sptk\\bcut -s ' + str(cut_loop * 800 // 16) + ' > tmp\\tmp.bcut', shell=True)
                subprocess.call('bin\\sptk\\mfcc -l 800 -f 16 -m 12 -n 20 -a 0.97 -E < tmp\\tmp.bcut | bin\\sptk\\delta -m 12 -d -0.5 0 0.5 -d 0.25 0 -0.5 0 0.25 > tmp\\tmp.mfcc', shell=True)
                subprocess.call('bin\\sptk\\pitch -a 1 -H 600 -p 800 -t1 0.2 < tmp\\tmp.bcut > tmp\\tmp.pitch', shell=True)
                subprocess.call('bin\\sptk\\frame -l 1024 -p 800 < tmp\\tmp.bcut | bin\\sptk\\window -l 1024 -L 1024 | bin\\sptk\\mcep -l 1024 -m 39 -a 0.42 -e 0.001 > "' + gen_dir_name + '\\' + name + '_' + str(cut_loop) + '.mcep"', shell=True)
            else:
                subprocess.call('x2x +sf < tmp/tmp.raw | bcut -s ' + str(cut_loop * 800 // 16) + ' > tmp/tmp.bcut', shell=True)
                subprocess.call('mfcc -l 800 -f 16 -m 12 -n 20 -a 0.97 -E < tmp/tmp.bcut | delta -m 12 -d -0.5 0 0.5 -d 0.25 0 -0.5 0 0.25 > tmp/tmp.mfcc', shell=True)
                subprocess.call('pitch -a 1 -H 600 -p 800 -t1 0.2 < tmp/tmp.bcut > tmp/tmp.pitch', shell=True)
                subprocess.call('frame -l 1024 -p 800 < tmp/tmp.bcut | window -l 1024 -L 1024 | mcep -l 1024 -m 39 -a 0.42 -e 0.001 > "' + gen_dir_name + '/' + name + '_' + str(cut_loop) + '.mcep"', shell=True)
            
            mfcc = []
            mfcc_data = open('tmp/tmp.mfcc', 'rb').read()
            for loop in range(len(mfcc_data) // (4 * 39)):
                mfcc.append(list(struct.unpack('<39f', mfcc_data[loop * 4 * 39:(loop + 1) * 4 * 39])))
            mfcc_np = np.array([mfcc], dtype='float64')
            result_np = model_voice.predict(mfcc_np)
            
            pitch_data = open('tmp/tmp.pitch', 'rb').read()
            write_file = open(gen_dir_name + '/' + name + '_' + str(cut_loop) + '.voice', 'wb')
            for loop in range(result_np.shape[1]):
                result = result_np[0, loop, :]
                for val in result:
                    write_file.write(struct.pack('<f', val))
                write_file.write(pitch_data[loop * 4:(loop + 1) * 4])
            write_file.close()
        
    if zip_name is not None:
        shutil.make_archive(zip_name, 'zip', root_dir=gen_dir_name)


if __name__ == '__main__':
    input_voices_dir = 'targets'
    gen_dir_name = 'gen_targets'
    zip_file_name = None
    if len(sys.argv) >= 2:
        input_voices_dir = sys.argv[1]
    if len(sys.argv) >= 3:
        gen_dir_name = sys.argv[2]
    if len(sys.argv) >= 4:
        zip_file_name = sys.argv[3]

    gen_targets_main(input_voices_dir, gen_dir_name, zip_file_name)
