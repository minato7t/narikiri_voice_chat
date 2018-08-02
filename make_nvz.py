#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import subprocess
import struct
import sys
import zipfile


if __name__ == '__main__':
    output_file = 'outputs/target.nvz'
    if len(sys.argv) > 1:
        output_file = sys.argv[1]

    files = sys.argv[2:]
    if len(files) != 3:
        files = ['outputs/target.nvm', 'outputs/target.pb', 'outputs/pitch.pb']
    
    arc_names = ['target.nvm', 'target.pb', 'pitch.pb']

    if os.path.isdir('tmp') == False:
        os.mkdir('tmp')
    
    with zipfile.ZipFile(output_file, 'w', compression=zipfile.ZIP_DEFLATED) as new_zip:
        for file_loop in range(len(arc_names)):
            new_zip.write(files[file_loop], arcname=arc_names[file_loop])

