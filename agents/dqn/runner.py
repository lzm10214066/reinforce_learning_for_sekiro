import torch
import torch.backends.cudnn as cudnn
from environment.environment import Environment
from agents.dqn.dqn_agent import DQNAgent
import collections

import os
import time
import numpy as np
from tensorboardX import SummaryWriter

Transition = collections.namedtuple("Transition",
                                    ["state", "action", "reward", "next_state", "done"])


class Runner:
    def __init__(self, config):
        self.config = config

    def run(self):
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        np.random.seed(0)
        cudnn.benchmark = True

        # define model
        agent = DQNAgent(self.config.agent.dqn)
        agent.cuda()
        env = Environment(self.config.env)

        if self.config.env.model_load_path is not None:
            agent.load_state_dict(torch.load(self.config.env.model_load_path))

        # define log out
        time_array = time.localtime(time.time())
        log_time = time.strftime("%Y_%m_%d_%H_%M_%S", time_array)
        # name log_dir with paras
        log_dir_name = '_'.join([self.config.env.reward.reward_option,
                                 str(self.config.agent.dqn.buffer_size),
                                 str(self.config.agent.ddpg.a_learning_rate), log_time])

        log_dir = os.path.join(self.config.log_root, self.config.env.log_dir, log_dir_name)
        if os.path.exists(log_dir) is False:
            os.makedirs(log_dir)

        save_dir = os.path.join(self.config.log_root, self.config.env.save_dir, log_dir_name)
        if os.path.exists(save_dir) is False:
            os.makedirs(save_dir)

        # # parms out
        # paras_path = os.path.join(log_dir, 'paras.yaml')
        # with open(paras_path, "w", encoding='utf-8') as f:
        #     yaml.dump(self.config, f)

        for current_episode in range(self.config.env.num_episode):
            episode_log = 'sekiro_episode_' + str(current_episode)
            log_path = os.path.join(log_dir, episode_log)
            tb_logger = SummaryWriter(log_path)

            cur_step = 0
            state = env.init_state

            while True:
                end = time.time()
                action = agent.select_action_with_explore(state)
                next_state, reward = env.step(action, state)
                trans = Transition(state=state, action=action, next_state=next_state, reward=reward, done=False)
                agent.buffer.add(trans)
                agent.update_qnet()
                cur_step += 1

                if cur_step % self.config.env.save_interval == 0:
                    save_path = os.path.join(save_dir, 'dqn_sekiro_step_' + str(cur_step) + '.pth')
                    agent.save_model(save_path)

                tb_logger.add_scalar('reward', reward, cur_step)
                dtime = time.time() - end
                print('step:', cur_step, 'reward:', reward, 'dtime:', dtime, )