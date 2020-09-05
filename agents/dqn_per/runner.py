import torch
import torch.backends.cudnn as cudnn
from environment.environment import Environment
from agents.dqn_per.dqn_per_agent import DQN_PER_Agent
import collections

import os
import time
import numpy as np
from tensorboardX import SummaryWriter
from environment.key_input.getkeys import key_check, get_action

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
        agent = DQN_PER_Agent(self.config.agent.dqn_per)
        if self.config.agent.dqn_per.agent_model_path is not None:
            agent.load_model(self.config.agent.dqn_per.agent_model_path)
        agent.cuda()
        env = Environment(self.config.env)

        # define log out
        time_array = time.localtime(time.time())
        log_time = time.strftime("%Y_%m_%d_%H_%M_%S", time_array)
        # name log_dir with paras
        log_dir_name = '_'.join([self.config.env.reward.reward_option,
                                 str(self.config.agent.dqn_per.buffer_size),
                                 str(self.config.agent.dqn_per.a_learning_rate), log_time])

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
            tb_step = self.config.agent.dqn_per.tb_step

            cur_step = 0
            state = env.init_state

            mimic_f = False
            pause_f = False
            while True:
                end = time.time()
                keys = key_check()
                if 'H' in keys:
                    pause_f = not pause_f
                    time.sleep(1)
                if pause_f:
                    print('\rlearning pause', end='')
                else:
                    action_info = 'people'
                    if 'P' in keys:
                        mimic_f = not mimic_f
                        time.sleep(1)
                    if mimic_f:
                        action = get_action()
                    else:
                        action_info = 'learning action'
                        action, action_info = agent.select_action_with_explore(state, action_info)

                    next_state, reward = env.step(action, state, mimic_f)

                    trans = Transition(state=state, action=action, next_state=next_state, reward=reward, done=False)
                    agent.buffer.add_T(trans)
                    agent.buffer.add_P(0)
                    agent.update_qnet(cur_step, tb_logger, tb_step)
                    cur_step += 1

                    if cur_step % self.config.env.save_interval == 0:
                        save_path = os.path.join(save_dir, 'dqn_sekiro_step_' + str(cur_step) + '.pth')
                        agent.save_model(save_path)

                    tb_logger.add_scalar('reward', reward, cur_step)

                    state = next_state

                    dtime = time.time() - end
                    if abs(reward) > -0.0001:
                        print('step:', cur_step, 'action:', action, 'reward:', reward, 'dtime:', dtime, action_info)
