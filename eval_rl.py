import numpy as np
from CO_mapf_gym_simple import CO_MAPFEnv
from util import set_global_seeds
import torch
from alg_parameters import *
from map_model import MapModel
import datetime
import sys
RUN_STEP=5120
EVAL_TIMES= 10
# recording path and collide times

class Runner(object):
    def __init__(self, env_id,path_name=None):
        self.ID = env_id
        set_global_seeds(env_id*123)
        self.env_map= CO_MAPFEnv(env_id,RUN_STEP,path_name)
        self.local_device = torch.device('cuda') if SetupParameters.USE_GPU_LOCAL else torch.device('cpu')
        self.local_map_model = MapModel(env_id, self.local_device)
        restore_path = 'models/LMAPF/26_26_3/final'
        map_net_path_checkpoint = restore_path + "/map_net_checkpoint.pkl"
        map_net_dict = torch.load(map_net_path_checkpoint)
        self.local_map_model.network.load_state_dict(map_net_dict['model'])

    def map_run(self,seed):
        with torch.no_grad():
            self.env_map.global_reset_fix(seed)
            map_hidden_state = (
                torch.zeros((runParameters.N_NODE, CopParameters.NET_SIZE)).to(self.local_device),
                torch.zeros((runParameters.N_NODE, CopParameters.NET_SIZE)).to(self.local_device))
            map_obs, map_vector= self.env_map.observe_for_map()
            map_done = False
            while map_done==False:
                map_action, _, _, map_hidden_state = self.local_map_model.step(map_obs, map_vector,map_hidden_state,runParameters.N_NODE)
                map_done, rl_local_restart, map_obs, map_vector = self.env_map.joint_step(map_action)
                if rl_local_restart and not map_done:
                    self.env_map.local_reset()
        throughput=self.env_map.all_finished_task/self.env_map.episode_len
        return throughput,self.env_map.global_path,self.env_map.collide_times



if __name__ == "__main__": 
    if len(sys.argv) > 1 :
        env = Runner(0,sys.argv[1])
    else:
        env = Runner(0)
    throughputs=[]
    print("start evaluation rl {} at {}".format(env.env_map.path,datetime.datetime.now()))
    for eval_time in range(EVAL_TIMES):
        throughput, paths, collide_times = env.map_run(eval_time*123)
        print('[{}] evaluation times:{}'.format(datetime.datetime.now(),eval_time))
        print('throughput:{}'.format(throughput))
        throughputs.append(throughput)

    throughput_std=np.std(throughputs)
    throughput_mean=np.mean(throughputs)
    print("mean throughput:{}, std throughput:{}".format(throughput_mean,throughput_std))
    print()