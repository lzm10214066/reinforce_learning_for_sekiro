# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 10:37:50 2020

@author: analoganddigital   ( GitHub )
"""

import ctypes
import time

SendInput = ctypes.windll.user32.SendInput

ESC = 0x01  # stop

# move
W = 0x11  # forward
A = 0x1E  # left
S = 0x1F  # back
D = 0x20  # right

# action
J = 0x24  # attack
K = 0x25  # block
L = 0x26  # dodge
I = 0x17  # jump

# view
M = 0x32

# backup
LSHIFT = 0x2A
R = 0x13
V = 0x2F
Q = 0x10
O = 0x18
P = 0x19
C = 0x2E

# C struct redefinitions 
PUL = ctypes.POINTER(ctypes.c_ulong)


class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]


class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]


class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]


class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),
                ("mi", MouseInput),
                ("hi", HardwareInput)]


class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]


# Actuals Functions
def PressKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput(0, hexKeyCode, 0x0008, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(1), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))


def ReleaseKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput(0, hexKeyCode, 0x0008 | 0x0002, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(1), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))


def press_release(key):
    PressKey(key)
    time.sleep(0.1)
    ReleaseKey(key)
    time.sleep(0.15)


def go_forward():
    press_release(W)


def go_back():
    press_release(S)


def go_left():
    press_release(A)


def go_right():
    press_release(D)


def attack():
    press_release(J)


def block():  # 格挡
    press_release(K)


def dodge():  # 闪避
    press_release(L)
    pass


def jump():
    #press_release(I)
    time.sleep(0.1)
    pass


def do_nothing():
    time.sleep(0.1)
    pass


def fix_view():
    press_release(M)
    pass


def pause():
    press_release(ESC)


if __name__ == '__main__':
    time.sleep(5)
    time1 = time.time()
    while (True):
        if abs(time.time() - time1) > 5:
            break
        else:
            press_release(W)
            press_release(S)
            press_release(A)
            press_release(D)
            press_release(J)
            press_release(K)
            press_release(L)
            press_release(I)
