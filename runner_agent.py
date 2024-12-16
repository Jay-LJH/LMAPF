import numpy as np
import ray
import torch
from map_model import MapModel
from alg_parameters import *
from CO_mapf_gym_agent import CO_MAPFEnv
from util import *

class Runner(object):
    """sub-process used to collect experience"""

    def __init__(self, env_id,file_name=None):
        """initialize model and environment"""
        self.ID = env_id
        set_global_seeds(env_id*123)
        self.env_map= CO_MAPFEnv(env_id,file_name=file_name)
        self.local_device = torch.device('cuda') if SetupParameters.USE_GPU_LOCAL else torch.device('cpu')
        self.local_map_model = MapModel(env_id, self.local_device)
        self.map_done = False
        self.num_agent = runParameters.N_AGENT
        self.num_node = runParameters.N_NODE
        self.map_reset()

    def map_run(self): # run one episode, global reset -map eval-mapupdate- agent eval -step-condition- map eval -update map-modify obs
        """run multiple steps and collect corresponding data """
        num_window=0
        num_episode=0
        one_episode_perf= None
        mb_obs, mb_vector,mb_rewards, mb_values,mb_ps,mb_hidden_state,mb_actions,mb_done = [], [], [], [], [], [],[],[]
        with torch.no_grad():           
            while num_window < CopParameters.NUM_WINDOW:
                mb_obs.append(self.map_obs)
                mb_vector.append(self.map_vector)
                mb_hidden_state.append(
                    [self.map_hidden_state[0].cpu().detach().numpy(), self.map_hidden_state[1].cpu().detach().numpy()])
                mb_done.append(self.map_done)
                map_action, ps, v, self.map_hidden_state= self.local_map_model.step(self.map_obs,self.map_vector,
                                                                           self.map_hidden_state,self.num_agent)            
                mb_values.append(v)
                mb_ps.append(ps)
                mb_actions.append(map_action)
                rewards=self.map_one_window(map_action)
                mb_rewards.append(rewards)
                num_window += 1
                if self.map_done:
                    one_episode_perf = self.env_map.calculate_info_episode()
                    self.map_reset()
                    num_episode+=1
        window_perf = self.env_map.calculate_info()
        window_perf['reward']=np.sum(mb_rewards)
        mb_obs = np.concatenate(mb_obs, axis=0)
        mb_vector = np.concatenate(mb_vector, axis=0)
        mb_actions = np.asarray(mb_actions, dtype=np.int64)
        mb_hidden_state = np.stack(mb_hidden_state)
        mb_rewards = np.concatenate(mb_rewards, axis=0)
        mb_values = np.squeeze(np.concatenate(mb_values, axis=0), axis=-1)
        mb_ps = np.stack(mb_ps)
        mb_done = np.asarray(mb_done, dtype=np.bool_)
        if self.map_done:
            last_values = np.zeros((mb_values.shape[1]), dtype=np.float32)
        else:
            last_values = self.local_map_model.value(self.map_obs,self.map_vector, self.map_hidden_state)

        # calculate advantages
        mb_advs = np.zeros_like(mb_rewards)
        last_gaelam =  0
        for t in reversed(range(num_window)):
            if t == num_window - 1:
                next_nonterminal = 1.0 - self.map_done
                next_values = last_values
            else:
                next_nonterminal = 1.0- mb_done[t + 1]
                next_values = mb_values[t + 1]
            delta = mb_rewards[t] + TrainingParameters.GAMMA * next_values * next_nonterminal - mb_values[t]
            mb_advs[t] = last_gaelam = delta + TrainingParameters.GAMMA * TrainingParameters.LAM * next_nonterminal * last_gaelam

        mb_returns = np.add(mb_advs, mb_values)
        return mb_obs, mb_vector,mb_returns, mb_values,mb_actions, mb_ps, mb_hidden_state,num_episode,window_perf,one_episode_perf
    
    def map_one_window(self,map_action=None): # run one episode
        self.map_done,rl_local_restart, rewards, self.map_obs,self.map_vector=self.env_map.joint_step(map_action)
        if rl_local_restart and not self.map_done:
            self.env_map.local_reset()
        return rewards

    def map_reset(self):
        self.env_map.global_reset()
        self.map_hidden_state = (
            torch.zeros((self.num_agent, CopParameters.NET_SIZE)).to(self.local_device),
            torch.zeros((self.num_agent, CopParameters.NET_SIZE)).to(self.local_device))
        self.map_obs,self.map_vector= self.env_map.observe_for_map()
        return

    def set_map_weights(self, weights):
        """load global weights to local models"""
        self.local_map_model.set_weights(weights)

@ray.remote(num_cpus=1, num_gpus=SetupParameters.NUM_GPU / (TrainingParameters.N_ENVS + 1))
class RLRunner(Runner):
    def __init__(self, meta_agent_id,file_name=None):
        super().__init__(meta_agent_id,file_name)

if __name__ == "__main__":
    import os
    if not os.path.exists("./h_maps"):
        os.makedirs("./h_maps")

    env = Runner(0)
    job_results = env.map_run()

    print("test")