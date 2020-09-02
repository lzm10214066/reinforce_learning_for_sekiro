from environment.key_input.directkeys import *
from environment.reward.get_reward import RewardReader
import torch
from environment.utils.grabscreen import grab_screen
import cv2
import numpy as np


class Environment:
    def __init__(self, config):
        self.config = config
        self.action_map = {0: go_forward, 1: go_back, 2: go_left, 3: go_right, 4: attack, 5: block, 6: dodge, 7: jump,
                           8: do_nothing}

        self.reward_reader = RewardReader(self.config.base_player, self.config.base_boss)
        self.init_state = np.zeros([3, 224, 224])
        self.region = (5, 0, 1024, 576)
        self.final_size = (224, 224)

    def trans_img(self, img):
        img = cv2.resize(img, self.final_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img * 3.2 / 255.0 - 1.6
        # img = img.transpose((2, 0, 1))
        img = np.expand_dims(img, axis=0)
        img = torch.from_numpy(img)
        img = img.float()
        return img

    def step(self, state, action):
        self.action_map[action]()
        reward = self.reward_reader.get_reward()
        cur_img = grab_screen(region=self.region)
        cur_img = self.trans_img(cur_img)
        next_state = np.concatenate([state[1:], cur_img], dim=0)
        return next_state, reward
