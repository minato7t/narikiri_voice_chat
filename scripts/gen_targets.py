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
import pyworld as pw


def gen_targets_main(input_voices_dir='targets', gen_dir_name='gen_targets', zip_file_name=None, cut_loop_length=16, reverse_flag=True):
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
    
    for input_voice_index, input_voice in enumerate(input_voices):
    
        print('\r音声ファイルの解析中...(' + str(input_voice_index) + '/' + str(len(input_voices)) + ')', end='')
        
        name, _ = os.path.splitext(input_voice)
        name = ('_' + name).replace('/', '_').replace('\\', '_')
        
        subprocess.call('sox "' + input_voice + '" -e signed-integer -c 1 -b 16 -r 16000 -L tmp/tmp.raw', shell=True)
        for cut_loop in range(cut_loop_length):
            if reverse_flag:
                reverse_list = [False, True]
            else:
                reverse_list = [False]
            for reverse in reverse_list:
                if reverse == False:
                    subprocess.call('x2x +sf < tmp/tmp.raw | bcut -s ' + str(cut_loop * 800 // 16) + ' > tmp/tmp.bcut', shell=True)
                    subprocess.call('frame -l 800 -p 100 < tmp/tmp.bcut | mfcc -l 800 -f 16 -m 12 -n 20 -a 0.97 -E | delta -m 12 -d -0.5 0 0.5 -d 0.25 0 -0.5 0 0.25 > tmp/tmp.mfcc', shell=True)
                    subprocess.call('frame -l 512 -p 100 < tmp/tmp.bcut | window -l 512 -L 512 | mcep -l 512 -m 20 -a 0.42 -e 1 > "' + gen_dir_name + '/' + name + '_' + str(cut_loop) + '_nor.mcep"', shell=True)
                else:
                    subprocess.call('x2x +sf < tmp/tmp.raw | sopr -m -1.0 | bcut -s ' + str(cut_loop * 800 // 16) + ' > tmp/tmp.bcut', shell=True)
                    subprocess.call('frame -l 800 -p 100 < tmp/tmp.bcut | mfcc -l 800 -f 16 -m 12 -n 20 -a 0.97 -E | delta -m 12 -d -0.5 0 0.5 -d 0.25 0 -0.5 0 0.25 > tmp/tmp.mfcc', shell=True)
                    subprocess.call('frame -l 512 -p 100 < tmp/tmp.bcut | window -l 512 -L 512 | mcep -l 512 -m 20 -a 0.42 -e 1 > "' + gen_dir_name + '/' + name + '_' + str(cut_loop) + '_rev.mcep"', shell=True)
                
                data_raw = open('tmp/tmp.bcut', 'rb').read()
                data_struct = struct.unpack('<' + str(len(data_raw) // 4) + 'f', data_raw)
                data = np.array(data_struct, dtype='float') / 32768.0
                
                f0, _ = pw.harvest(data, 16000, 80.0, 1000.0, 6.25)
                pitch = [0.0 for _ in range(f0.shape[0])]
                for loop in range(f0.shape[0]):
                    if f0[loop] >= 80.0:
                        pitch[loop] = 16000.0 / f0[loop]
                    else:
                        pitch[loop] = 0.0
                
                result_list = []
                mfcc_data = open('tmp/tmp.mfcc', 'rb').read()
                mfcc = []
                for loop in range(len(mfcc_data) // (4 * 39 * 8)):
                    for loop2 in range(8):
                        mfcc.append(list(struct.unpack('<39f', mfcc_data[(loop * 8 + loop2) * 4 * 39:(loop * 8 + loop2 + 1) * 4 * 39])))
                if len(mfcc) <= 0:
                    if reverse == False:
                        os.remove(gen_dir_name + '/' + name + '_' + str(cut_loop) + '_nor.pitch')
                        os.remove(gen_dir_name + '/' + name + '_' + str(cut_loop) + '_nor.mcep')
                    else:
                        os.remove(gen_dir_name + '/' + name + '_' + str(cut_loop) + '_rev.pitch')
                        os.remove(gen_dir_name + '/' + name + '_' + str(cut_loop) + '_rev.mcep')
                    continue
                mfcc_np = np.array([mfcc], dtype='float64')
                result_np = model_voice.predict(mfcc_np)
                results = result_np[0, :, :]
                
                if reverse == False:
                    write_file = open(gen_dir_name + '/' + name + '_' + str(cut_loop) + '_nor.voice', 'wb')
                else:
                    write_file = open(gen_dir_name + '/' + name + '_' + str(cut_loop) + '_rev.voice', 'wb')
                for loop in range(results.shape[0]):
                    for val in results[loop, :]:
                        write_file.write(struct.pack('<f', val))
                    write_file.write(struct.pack('<f', pitch[loop * 8]))
                write_file.close()
    
    print('\r音声ファイルの解析完了')
    
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
