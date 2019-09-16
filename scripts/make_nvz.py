#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import zipfile


def make_nvz_main(output_file, nvm_file, target_file, pitch_file=None):
    if pitch_file is not None:
        files = [nvm_file, target_file, pitch_file]
        arc_names = ['target.nvm', 'target.pb', 'pitch.pb']
    else:
        files = [nvm_file, target_file]
        arc_names = ['target.nvm', 'target.pb']
    
    with zipfile.ZipFile(output_file, 'w', compression=zipfile.ZIP_DEFLATED) as new_zip:
        for file_loop in range(len(arc_names)):
            new_zip.write(files[file_loop], arcname=arc_names[file_loop])


if __name__ == '__main__':
    output_file = 'outputs/target.nvz'
    if len(sys.argv) > 1:
        output_file = sys.argv[1]
    nvm_file = 'outputs/target.nvm'
    target_file = 'outputs/target.pb'
    pitch_file = 'outputs/pitch.pb'
    if len(sys.argv) == 5:
        nvm_file = sys.argv[2]
        target_file = sys.argv[3]
        pitch_file = sys.argv[4]

    make_nvz_main(output_file, nvm_file, target_file, pitch_file)
