#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

from target_train import target_train_main


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('arg error')
        exit()

    early_stopping_patience = None
    if len(sys.argv) >= 5 and int(sys.argv[4]) >= 0:
        early_stopping_patience = int(sys.argv[4])
    
    length = None
    if len(sys.argv) >= 6 and int(sys.argv[5]) >= 0:
        length = int(sys.argv[5])
    
    batch_size = 1
    if len(sys.argv) >= 7 and int(sys.argv[6]) >= 1:
        batch_size = int(sys.argv[6])
    
    do_compile = False
    if len(sys.argv) >= 8:
        if sys.argv[7] != 'False' and sys.argv[7] != 'false':
            try:
                do_compile = int(sys.argv[7]) > 0
            except ValueError:
                do_compile = True
    
    target_train_main(sys.argv[2], sys.argv[3], early_stopping_patience, length, batch_size, sys.argv[1], do_compile)

