#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import concatenate, Layer
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
        
        subprocess.call('sox "' + input_voice + '" -e signed-integer -c 1 -b 16 -r 16000 -L tmp/tmp.raw', shell=True)
        for cut_loop in range(16):
            for reverse in [False, True]:
                if reverse == False:
                    subprocess.call('x2x +sf < tmp/tmp.raw | bcut -s ' + str(cut_loop * 800 // 16) + ' > tmp/tmp.bcut', shell=True)
                    subprocess.call('frame -l 800 -p 100 < tmp/tmp.bcut | mfcc -l 800 -f 16 -m 12 -n 20 -a 0.97 -E | delta -m 12 -d -0.5 0 0.5 -d 0.25 0 -0.5 0 0.25 > tmp/tmp.mfcc', shell=True)
                    subprocess.call('pitch -a 2 -H 600 -p 400 < tmp/tmp.bcut > "' + gen_dir_name + '/' + name + '_' + str(cut_loop) + '_nor.pitch"', shell=True)
                    subprocess.call('frame -l 512 -p 100 < tmp/tmp.bcut | window -l 512 -L 512 | mcep -l 512 -m 20 -a 0.42 -e 1 > "' + gen_dir_name + '/' + name + '_' + str(cut_loop) + '_nor.mcep"', shell=True)
                else:
                    subprocess.call('x2x +sf < tmp/tmp.raw | sopr -m -1.0 | bcut -s ' + str(cut_loop * 800 // 16) + ' > tmp/tmp.bcut', shell=True)
                    subprocess.call('frame -l 800 -p 100 < tmp/tmp.bcut | mfcc -l 800 -f 16 -m 12 -n 20 -a 0.97 -E | delta -m 12 -d -0.5 0 0.5 -d 0.25 0 -0.5 0 0.25 > tmp/tmp.mfcc', shell=True)
                    subprocess.call('pitch -a 2 -H 600 -p 400 < tmp/tmp.bcut > "' + gen_dir_name + '/' + name + '_' + str(cut_loop) + '_rev.pitch"', shell=True)
                    subprocess.call('frame -l 512 -p 100 < tmp/tmp.bcut | window -l 512 -L 512 | mcep -l 512 -m 20 -a 0.42 -e 1 > "' + gen_dir_name + '/' + name + '_' + str(cut_loop) + '_rev.mcep"', shell=True)
                
                pitch_error = False
                if reverse == False:
                    pitch_data = open(gen_dir_name + '/' + name + '_' + str(cut_loop) + '_nor.pitch', 'rb').read()
                else:
                    pitch_data = open(gen_dir_name + '/' + name + '_' + str(cut_loop) + '_rev.pitch', 'rb').read()
                for loop in range(len(pitch_data) // 4):
                    if struct.unpack('<f', pitch_data[loop * 4:(loop + 1) * 4])[0] > 16000.0 / 60.0:
                        pitch_error = True
                        break
                if pitch_error:
                    if reverse == False:
                        os.remove(gen_dir_name + '/' + name + '_' + str(cut_loop) + '_nor.pitch')
                        os.remove(gen_dir_name + '/' + name + '_' + str(cut_loop) + '_nor.mcep')
                    else:
                        os.remove(gen_dir_name + '/' + name + '_' + str(cut_loop) + '_rev.pitch')
                        os.remove(gen_dir_name + '/' + name + '_' + str(cut_loop) + '_rev.mcep')
                    continue
                
                result_list = []
                mfcc_data = open('tmp/tmp.mfcc', 'rb').read()
                for loop in range(len(mfcc_data) // (4 * 39 * 8)):
                    mfcc = []
                    for loop2 in range(8):
                        mfcc.append(list(struct.unpack('<39f', mfcc_data[(loop * 8 + loop2) * 4 * 39:(loop * 8 + loop2 + 1) * 4 * 39])))
                    mfcc_np = np.array([mfcc], dtype='float64')
                    result_np = model_voice.predict(mfcc_np)
                    result_list.append(result_np[0, 0, :])
                
                if reverse == False:
                    write_file = open(gen_dir_name + '/' + name + '_' + str(cut_loop) + '_nor.voice', 'wb')
                else:
                    write_file = open(gen_dir_name + '/' + name + '_' + str(cut_loop) + '_rev.voice', 'wb')
                for result in result_list:
                    for val in result:
                        write_file.write(struct.pack('<f', val))
                    write_file.write(pitch_data[loop * 2 * 4:(loop * 2 + 1) * 4])
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
