#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   export.py
@Time    :   2024/07/30 19:01:51
@Author  :   yangqinglin
@Version :   v1.0
@Email   :   yangql1@wedoctor.com
@Desc    :   None
'''

from llamafactory.train.tuner import export_model


def launch():
    export_model()
    
if __name__ == '__main__':
    launch()