# -*- coding: utf-8 -*-
from environment.utils.grabscreen import grab_screen
import cv2
import time
from environment.key_input.getkeys import key_check
import os
import os.path as osp

WIDTH = 96
HEIGHT = 86
LR = 1e-3
EPOCHS = 20
MODEL_NAME = 'model_sekiro_1/py-sekiro-{}-{}-{}-epochs.model'.format(LR, 'alexnetv2', EPOCHS)

# w j m k none

w = [1, 0, 0, 0, 0, 0]
j = [0, 1, 0, 0, 0, 0]
m = [0, 0, 1, 0, 0, 0]
k = [0, 0, 0, 1, 0, 0]
r = [0, 0, 0, 0, 1, 0]
n_choise = [0, 0, 0, 0, 0, 1]

# model = alexnet2(WIDTH, HEIGHT, LR, output = 6)
# model.load(MODEL_NAME)

window_size = (175, 0, 625, 450)  # 384,224  192,112 96,86


def main():
    last_time = time.time()
    for i in list(range(5))[::-1]:
        print(i + 1)
        time.sleep(1)
    paused = False
    save_image_root = 'images'
    os.makedirs(save_image_root, exist_ok=True)
    im_count = 0
    while (True):

        if not paused:
            # 800x600 windowed mode
            screen = grab_screen(region=(window_size))
            cv2.imshow('test', screen)
            cv2.waitKey(1)
            print('loop took {} ms\n'.format(time.time() - last_time) * 1000)
            last_time = time.time()
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
            #screen = cv2.resize(screen, (224, 224))
            cv2.imwrite(osp.join(save_image_root, 'screen_%05d.jpg' % im_count), screen)
            im_count += 1
        keys = key_check()

        # p pauses game and can get annoying.
        if 'T' in keys:
            if paused:
                paused = False
                time.sleep(1)
            else:
                paused = True
                '''
                ReleaseKey(A)
                ReleaseKey(W)
                ReleaseKey(D)
                '''
                time.sleep(1)
        if 'Y' in keys:
            if paused:
                paused = False
                time.sleep(1)
            else:
                paused = True
                '''
                directkeys.ReleaseKey(J)
                directkeys.ReleaseKey(W)
                directkeys.ReleaseKey(M)
                directkeys.ReleaseKey(K)
                '''
                time.sleep(1)
                break


main()
