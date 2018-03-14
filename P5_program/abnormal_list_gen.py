# -*- coding: utf-8 -*-

import os

abnormal_list = os.listdir('./Dataset/train/abnormal1')

with open('abnormal.txt', 'w') as f:
    for line in abnormal_list:
        f.write(line+'\n')
