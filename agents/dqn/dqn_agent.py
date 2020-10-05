import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam

from torchvision import models
from agents.dqn.replay_buffer import ReplayBuffer
from utils import load_state, load_state_simple
from models.fbresnet_nobn import fbresnet18

criterion = nn.MSELoss()


class DQNAgent(nn.Module):
    def __init__(self, config):
        super(DQNAgent, self).__init__()

        self.config = config
        self.action_dim = self.config.action_dim
        self.q_net = fbresnet18(num_classes=self.config.action_dim)
        #self.q_net = models.vgg16(pretrained=False, num_classes=self.config.action_dim)
        # for k in self.q_net.state_dict():
        #     print(k)

        if self.config.model_load_path is not None:
            load_state_simple(self.config.model_load_path, self.q_net, self.config.ignore_prefix)

        self.q_tar_net = fbresnet18(num_classes=self.config.action_dim)
        #self.q_tar_net = models.vgg16(pretrained=False, num_classes=self.config.action_dim)

        self.q_tar_net.eval()
        self.hard_update(self.q_tar_net, self.q_net)  # Make sure target is with the same weight

        self.q_net.cuda()
        self.q_tar_net.cuda()

        self.q_net_optim = Adam(self.q_net.parameters(), lr=self.config.a_learning_rate,
                                weight_decay=self.config.weight_decay)

        # Create replay buffer
        self.buffer = ReplayBuffer(self.config.buffer_size)

        # Hyper-parameters
        self.batch_size = self.config.policy_batch_size
        self.discount = float(self.config.discount)
        self.decay_epsilon = self.config.decay_epsilon
        self.epsilon = self.config.epsilon
        self.max_grad_norm = self.config.max_grad_norm
        self.tau = self.config.tau

        self.is_training = True

    def forward(self, *input):
        pass

    def update_qnet_full(self, total_i, tb_logger):
        if self.buffer.size() < self.batch_size:
            return

        print('replay_buffer_size:', self.buffer.size())
        for i in range(self.config.update_full_epoch):
            indices = np.arange(self.buffer.size())
            np.random.shuffle(indices)
            offset = 0
            while offset + self.config.policy_batch_size <= self.buffer.size():
                picked = indices[offset:offset + self.config.policy_batch_size]
                batch = [self.buffer.buffer[i] for i in picked]

                state_batch = np.array([_[0] for _ in batch])
                action_batch = np.array([_[1] for _ in batch])
                reward_batch = np.array([_[2] for _ in batch])
                next_state_batch = np.array([_[3] for _ in batch])
                terminal_batch = np.array([_[4] for _ in batch])

                state_batch = torch.from_numpy(state_batch).cuda()
                action_batch = torch.from_numpy(np.expand_dims(action_batch, axis=1)).cuda()
                reward_batch = torch.from_numpy(reward_batch.astype(np.float32)).cuda()
                next_state_batch = torch.from_numpy(next_state_batch).cuda()
                terminal_batch = torch.from_numpy(terminal_batch.astype(np.float32)).cuda()
                # Prepare for the target q batch
                with torch.no_grad():
                    next_q_index = torch.argmax(self.q_net(next_state_batch), dim=1, keepdim=True)
                    next_q_values = self.q_tar_net(next_state_batch).gather(index=next_q_index, dim=1).squeeze()

                target_q_batch = reward_batch + self.discount * (1 - terminal_batch) * next_q_values

                q_batch = self.q_net(state_batch).gather(index=action_batch, dim=1)

                value_loss = criterion(q_batch.squeeze(), target_q_batch).cuda()

                # Critic update
                self.q_net_optim.zero_grad()
                value_loss.backward()
                # nn.utils.clip_grad_norm_(self.q_net.parameters(), self.max_grad_norm)
                self.q_net_optim.step()

                # Target update
                self.soft_update(self.q_tar_net, self.q_net, self.tau)

                # build summary
                offset += self.config.policy_batch_size
                if i == self.config.update_full_epoch - 1 and offset + self.config.policy_batch_size >= self.buffer.size():
                    tb_logger.add_scalar('critic loss', value_loss, total_i)
                    tb_logger.add_scalar('q', q_batch[0], total_i)
                    tb_logger.add_scalar('next_q', next_q_values[0], total_i)
                    tb_logger.add_scalar('epsilon', self.epsilon, total_i)

                    print("policy_step:", total_i,
                          "\tvalue_loss:", value_loss.item(),
                          "\tq_predict:", q_batch[0].item(),
                          "\tnext_q_predict:", next_q_values[0].item(),
                          "\tepsilon:", self.epsilon)

    def update_qnet(self, total_i, tb_logger, tb_step):
        if self.buffer.size() < self.batch_size:
            return
        # Sample batch
        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = \
            self.buffer.sample_batch(self.batch_size)

        state_batch = torch.from_numpy(state_batch).cuda()
        action_batch = torch.from_numpy(np.expand_dims(action_batch, axis=1)).cuda()
        reward_batch = torch.from_numpy(reward_batch.astype(np.float32)).cuda()
        next_state_batch = torch.from_numpy(next_state_batch).cuda()
        terminal_batch = torch.from_numpy(terminal_batch.astype(np.float32)).cuda()
        # Prepare for the target q batch
        with torch.no_grad():
            next_q_index = torch.argmax(self.q_net(next_state_batch), dim=1, keepdim=True)
            next_q_values = self.q_tar_net(next_state_batch).gather(index=next_q_index, dim=1).squeeze()

        target_q_batch = reward_batch + self.discount * (1 - terminal_batch) * next_q_values

        q_batch = self.q_net(state_batch).gather(index=action_batch, dim=1)

        value_loss = criterion(q_batch.squeeze(), target_q_batch).cuda()

        # Critic update
        self.q_net_optim.zero_grad()
        value_loss.backward()
        # nn.utils.clip_grad_norm_(self.q_net.parameters(), self.max_grad_norm)
        self.q_net_optim.step()

        # Target update
        self.soft_update(self.q_tar_net, self.q_net, self.tau)

        # build summary
        if total_i % tb_step == 0:
            tb_logger.add_scalar('q_loss', value_loss, total_i)
            tb_logger.add_scalar('q', q_batch[0], total_i)
            tb_logger.add_scalar('next_q', next_q_values[0], total_i)
            tb_logger.add_scalar('epsilon', self.epsilon, total_i)

            print("policy_step:", total_i,
                  "\tvalue_loss:", value_loss.item(),
                  "\tq_predict:", q_batch[0].item(),
                  "\tnext_q_predict:", next_q_values[0].item(),
                  "\tepsilon:", self.epsilon)

    def select_action_with_explore(self, state, action_info=''):
        e = np.random.random_sample()
        if e <= self.epsilon:
            action = np.random.randint(0, self.action_dim)
            action_info += "  random"
        else:
            state = torch.from_numpy(np.expand_dims(state, axis=0)).cuda()
            # self.q_net.eval()
            with torch.no_grad():
                q_out = self.q_net(state)
                action = torch.argmax(q_out, dim=1).item()
            # self.q_net.train()
        self.epsilon = max(self.epsilon - self.config.decay_epsilon, 0)
        action = np.int64(action)
        return action, action_info

    def load_model(self, path):
        if path is None:
            print('the path is None')
            return
        self.load_state_dict(torch.load(path))
        print('load ', path)

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def seed(self, s):
        torch.manual_seed(s)
        if self.use_cuda:
            torch.cuda.manual_seed(s)

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )

    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
