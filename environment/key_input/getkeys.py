# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 12:03:44 2020

@author: analoganddigital   ( GitHub )
"""
import numpy as np
import win32api as wapi
import time

keyList = ["\b"]
for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ 123456789,.'£$/\\":
    keyList.append(char)


def key_check():
    keys = []
    for key in keyList:
        if wapi.GetAsyncKeyState(ord(key)):
            keys.append(key)
    return keys


def get_key(keys):
    # W,J,M,K,R,none R代替左shift，识破
    output = [0, 0, 0, 0, 0, 0]
    if 'W' in keys:
        output[0] = 1
    elif 'J' in keys:
        output[1] = 1
    elif 'M' in keys:
        output[2] = 1
    elif 'K' in keys:
        output[3] = 1
    elif 'R' in keys:
        output[4] = 1
    elif '8' in keys:  # 停止记录操作
        output = [1, 1, 1, 1, 1, 1]
    else:
        output[5] = 1

    return output


def get_action():
    action = -1
    while action == -1:
        keys = key_check()
        if 'W' in keys:
            action = 0
        elif 'A' in keys:
            action = 2
        elif 'S' in keys:
            action = 1
        elif 'D' in keys:
            action = 3
        elif 'J' in keys:
            action = 4
        elif 'K' in keys:
            action = 5
        elif 'L' in keys:
            action = 6
        elif 'I' in keys:
            action = 7
        elif 'M' in keys:
            action = 9
    return np.int64(action)


if __name__ == '__main__':
    while True:
        print(get_action())
