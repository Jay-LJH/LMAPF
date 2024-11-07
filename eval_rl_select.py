import os
from util import *
import numpy as np
from util import set_global_seeds
import torch
from alg_parameters import *
from map_model import MapModel
from node_selector import *
import sys
from CO_mapf_gym_select import CO_MAPFEnv
import datetime
RUN_STEP=2048
EVAL_TIMES=1
# recording path and collide times

class Runner(object):
    def __init__(self, env_id, selected_vertex,model_path=None,file_name=None):
        self.ID = env_id
        set_global_seeds(env_id*123)
        self.env_map= CO_MAPFEnv(env_id, selected_vertex, RUN_STEP,file_name)
        self.local_device = torch.device('cuda') if SetupParameters.USE_GPU_LOCAL else torch.device('cpu')
        self.local_map_model = MapModel(env_id, self.local_device)
        self.K = self.env_map.num_node
        restore_path = 'models/LMAPF/'+ model_path+'/final'
        map_net_path_checkpoint = restore_path + "/map_net_checkpoint.pkl"
        map_net_dict = torch.load(map_net_path_checkpoint)
        self.local_map_model.network.load_state_dict(map_net_dict['model'])

    def map_run(self,seed):
        with torch.no_grad():
            self.env_map.global_reset(True,seed)
            map_hidden_state = (
                torch.zeros((self.K, CopParameters.NET_SIZE)).to(self.local_device),
                torch.zeros((self.K, CopParameters.NET_SIZE)).to(self.local_device))
            map_obs, map_vector= self.env_map.observe_for_map()
            map_done = False
            rl_time = 0
            while map_done==False:
                before = datetime.datetime.now()
                map_action, _, _, map_hidden_state = self.local_map_model.step(map_obs, map_vector,map_hidden_state,self.K)
                rl_time += (datetime.datetime.now()-before).total_seconds()
                map_done, rl_local_restart, map_obs, map_vector = self.env_map.joint_step(map_action)
                if rl_local_restart and not map_done:
                    self.env_map.local_reset()
            throughput=self.env_map.all_finished_task/self.env_map.episode_len
        return throughput,rl_time
    
# args 1: path 2: method 3:probability
if __name__ == "__main__":   
    if len(sys.argv) > 1 :
        origin_path = sys.argv[1]
    else:
        origin_path = "Proportion_Maze_26_26_3"
    path = "maps/"+origin_path+".txt"
    print("path:{}".format(path))
    if len(sys.argv) > 2:
        model_path = sys.argv[2]
    else:
        model_path = "26*26_3"
    print("model_path:{}".format(model_path))
    dict = {"random":random_selector,"pibt":pibt_selector,"BC":BC_selector,"all":selector}
    if len(sys.argv) > 3:
        select = dict[sys.argv[3]]
    else:
        select = dict["all"]
    print("select:{}".format(select.__name__))
    if len(sys.argv) > 4:
        probability = float(sys.argv[4])
    else:
        probability = 1.0
    print("probability:{}".format(probability))
    print("[{}] begin eval rl select {} using {}".format(datetime.datetime.now(),origin_path,select.__name__))
    
    _,_,map = read_map(path)
    selected_vertex = select(map,probability,file_name=origin_path).select()
    print("[{}] finish selected_vertex".format(datetime.datetime.now()))
    env = Runner(0,selected_vertex,model_path,origin_path)
    throughputs=[]
    for eval_time in range(EVAL_TIMES):  # 0 wait ,1 right, 2 down, 3 left, 4 up
        throughput,rl_time = env.map_run(eval_time*123)
        print('[{}] evaluation times:{}'.format(datetime.datetime.now(),eval_time))
        print('throughput:{} rl time:{} pibt time: {}'.format(throughput,rl_time,env.env_map.pibt_time))
        throughputs.append(throughput)

    throughput_std=np.std(throughputs)
    throughput_mean=np.mean(throughputs)
    print("mean throughput:{}, std throughput:{}".format(throughput_mean,throughput_std))
    print()
    
