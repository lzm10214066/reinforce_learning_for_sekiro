import numpy as np

import torch
import torch.nn as nn

from agents.dqn_per.priority_replay_buffer import PriorityReplayBuffer
from agents.dqn.dqn_agent import DQNAgent

criterion = nn.MSELoss()


class DQN_PER_Agent(DQNAgent):
    def __init__(self, config):
        super().__init__(config)

        self.buffer = PriorityReplayBuffer(self.config.buffer_size, self.config.alpha)

    def forward(self, *input):
        pass

    def update_qnet(self, total_i, tb_logger, tb_step):
        if self.buffer.size() < self.batch_size:
            return
        # Sample batch
        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch, sample_indices = \
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

        target_q_batch = reward_batch + float(self.discount) * (1 - terminal_batch) * next_q_values

        q_batch = self.q_net(state_batch).gather(index=action_batch, dim=1).squeeze()

        td_error = (target_q_batch - q_batch).detach().cpu().numpy()
        self.buffer.update_p(td_error, sample_indices=sample_indices)

        value_loss = criterion(q_batch, target_q_batch).cuda()

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

            # print("policy_step:", total_i,
            #       "\tvalue_loss:", value_loss.item(),
            #       "\tq_predict:", q_batch[0].item(),
            #       "\tnext_q_predict:", next_q_values[0].item(),
            #       "\tepsilon:", self.epsilon)

    def select_action_with_explore(self, state, action_info='', decay_epsilon=True, ):
        e = np.random.random_sample()
        if e <= self.epsilon:
            action = np.random.randint(0, self.action_dim - 1)
            action_info += "  random"
        else:
            state = torch.from_numpy(np.expand_dims(state, axis=0)).cuda()
            self.q_net.eval()
            with torch.no_grad():
                q_out = self.q_net(state)
                action = torch.argmax(q_out, dim=1).item()
            self.q_net.train()
        if decay_epsilon:
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
