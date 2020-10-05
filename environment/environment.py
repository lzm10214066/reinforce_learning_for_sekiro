from environment.key_input.directkeys import *
from environment.reward.get_reward import RewardReader
import torch
from environment.utils.grabscreen import grab_screen
import cv2
import numpy as np


class Environment:
    def __init__(self, config):
        self.config = config
        # self.action_map = {0: go_forward, 1: go_back, 2: go_left, 3: go_right, 4: attack, 5: block, 6: dodge, 7: jump,
        #                    8: do_nothing, 9: fix_view}

        self.action_map = {0: attack, 1: block, 2: dodge, 3: jump, 4: do_nothing, 5: fix_view}
        self.region = self.config.region
        self.original_size = (self.region[2] - self.region[0], self.region[3] - self.region[1])
        self.final_size = (self.original_size[0] // 2, self.original_size[1] // 2)

        self.reward_reader = RewardReader(self.config.reward.base_player, self.config.reward.base_boss)
        self.init_state = np.zeros([3, self.original_size[1] // 2, self.original_size[0] // 2], dtype=np.float32)

        # self.final_size = (self.config.final_size, self.config.final_size)

    def trans_img(self, img):
        img = cv2.resize(img, dsize=self.final_size)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img * 3.2 / 255.0 - 1.6
        img = img.transpose((2, 0, 1)).astype(np.float32)
        #limg = np.expand_dims(img, axis=0).astype(np.float32)
        return img

    def get_cur_img(self):
        return grab_screen(region=self.region)

    def get_cur_state(self, pre_state):
        cur_img = grab_screen(region=self.region)
        cur_img = self.trans_img(cur_img)
        #cur_state = np.concatenate([pre_state[1:], cur_img], axis=0)
        cur_state = cur_img
        return cur_state

    def step(self, action, state, mimi_f=False):
        if not mimi_f:
            self.action_map[action]()
        cur_img = grab_screen(region=self.region)
        cur_img = self.trans_img(cur_img)
        #next_state = np.concatenate([state[1:], cur_img], axis=0)
        next_state = cur_img

        reward = self.reward_reader.get_reward(action)
        return next_state, reward
