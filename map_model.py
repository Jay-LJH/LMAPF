import numpy as np
import torch
import torch.optim as optim
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler
from alg_parameters import *
from map_net import MAP_ACNet
from util import print_once

class MapModel(object):

    def __init__(self, env_id, device, global_model=False):
        """initialization"""
        self.ID = env_id
        self.device = device
        self.network = MAP_ACNet().to(device)  # neural network
        if global_model:
            self.net_optimizer = optim.Adam(self.network.parameters(), lr=TrainingParameters.lr, 
                                            eps=TrainingParameters.opti_eps,weight_decay=TrainingParameters.weight_decay)
            # self.multi_gpu_net = torch.nn.DataParallel(self.network) # training on multiple GPU
            self.net_scaler = GradScaler()  # automatic mixed precision

    def set_weights(self, weights):
        """load global weights to local models"""
        self.network.load_state_dict(weights)

    def step(self, observation,hidden_state, num_agent):
        observation = torch.from_numpy(observation).to(self.device)
        ps, v, _, hidden_state= self.network(observation, hidden_state)
        actions = np.zeros(num_agent)
        ps = np.squeeze(ps.cpu().detach().numpy())
        v = v.cpu().detach().numpy()
        for i in range(num_agent):
            actions[i] = np.random.choice(range(ps.shape[1]), p=ps[i])
        return actions, ps, v,hidden_state

    def greedy_step(self, observation, hidden_state, num_agent):
        observation = torch.from_numpy(observation).to(self.device)
        ps, _, hidden_state= self.network(observation, hidden_state)

        actions = np.zeros(num_agent)
        ps = np.squeeze(ps.cpu().detach().numpy())
        v = v.cpu().detach().numpy()
        for i in range(num_agent):
            actions[i] = np.argmax(ps[i])
        return actions, ps, v,hidden_state

    def value(self, observation, hidden_state):
        observation = torch.from_numpy(observation).to(self.device)
        _, v, _, _= self.network(observation, hidden_state)
        v =  np.squeeze(v.cpu().detach().numpy())
        return v

    def train(self, observation, returns, old_v, action, old_ps,input_state):
        """train model by reinforcement learning PPO"""
        self.net_optimizer.zero_grad()

        # from numpy to torch
        observation = torch.from_numpy(observation).to(self.device)
        returns = torch.from_numpy(returns).to(self.device)
        old_v = torch.from_numpy(old_v).to(self.device)
        action = torch.from_numpy(action).to(self.device)
        action = torch.unsqueeze(action, -1)
        old_ps = torch.from_numpy(old_ps).to(self.device)
        input_state_h = torch.from_numpy(
            np.reshape(input_state[:, 0], (-1, CopParameters.NET_SIZE))).to(self.device)
        input_state_c = torch.from_numpy(
            np.reshape(input_state[:, 1], (-1, CopParameters.NET_SIZE))).to(self.device)
        input_state = (input_state_h, input_state_c)

        advantage = returns - old_v
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-6)

        with autocast():
            new_ps, new_v,  policy_sig, _ = self.network(observation, input_state)
            new_p = new_ps.gather(-1, action)
            old_p = old_ps.gather(-1, action)
            ratio = torch.exp(torch.log(torch.clamp(new_p, 1e-6, 1.0)) - torch.log(torch.clamp(old_p, 1e-6, 1.0)))

            entropy = torch.mean(-torch.sum(new_ps * torch.log(torch.clamp(new_ps, 1e-6, 1.0)), dim=-1, keepdim=True))

            # critic loss
            new_v = torch.squeeze(new_v)
            new_v_clipped = old_v + torch.clamp(new_v - old_v, - TrainingParameters.CLIP_RANGE,
                                                      TrainingParameters.CLIP_RANGE)
            value_losses1 = torch.square(new_v- returns)
            value_losses2 = torch.square(new_v_clipped - returns)
            critic_loss = torch.mean(torch.maximum(value_losses1, value_losses2))

            # actor loss
            ratio = torch.squeeze(ratio)
            policy_losses = advantage * ratio
            policy_losses2 = advantage * torch.clamp(ratio, 1.0 - TrainingParameters.CLIP_RANGE,
                                                     1.0 + TrainingParameters.CLIP_RANGE)
            policy_loss = torch.mean(torch.min(policy_losses, policy_losses2))

            # total loss
            all_loss = -policy_loss - entropy * TrainingParameters.ENTROPY_COEF + \
                TrainingParameters.VALUE_COEF * critic_loss

        clip_frac = torch.mean(torch.greater(torch.abs(ratio - 1.0), TrainingParameters.CLIP_RANGE).float())

        self.net_scaler.scale(all_loss).backward()
        self.net_scaler.unscale_(self.net_optimizer)

        # Clip gradient
        grad_norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), TrainingParameters.MAX_GRAD_NORM)

        self.net_scaler.step(self.net_optimizer)
        self.net_scaler.update()
        # for recording
        prop_policy=-policy_loss/ (all_loss+1e-6)
        prop_en=-entropy * TrainingParameters.ENTROPY_COEF/ (all_loss+1e-6)
        prop_v = TrainingParameters.VALUE_COEF * critic_loss / (all_loss+1e-6)

        stats_list = [all_loss.cpu().detach().numpy(), policy_loss.cpu().detach().numpy(),
                      entropy.cpu().detach().numpy(),
                      critic_loss.cpu().detach().numpy(),
                      clip_frac.cpu().detach().numpy(), grad_norm.cpu().detach().numpy(),
                      torch.mean(advantage).cpu().detach().numpy(),prop_policy.cpu().detach().numpy(),
                      prop_en.cpu().detach().numpy(),prop_v.cpu().detach().numpy()]  # for recording
        return stats_list
