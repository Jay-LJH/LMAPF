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
import argparse
RUN_STEP = 5120
record = False
EVAL_TIMES=1 if record else 5
# recording path and collide times

class Runner(object):
    def __init__(self, env_id, selected_vertex,model_path=None,file_name=None):
        self.ID = env_id
        set_global_seeds(env_id*123)
        self.env_map= CO_MAPFEnv(env_id, selected_vertex, RUN_STEP,file_name)
        self.local_device = torch.device('cuda') if SetupParameters.USE_GPU_LOCAL else torch.device('cpu')
        self.local_map_model = MapModel(env_id, self.local_device)
        self.K = self.env_map.num_node
        restore_path = 'models/LMAPF/'+ model_path+"/final"
        map_net_path_checkpoint = restore_path + "/map_net_checkpoint.pkl"
        map_net_dict = torch.load(map_net_path_checkpoint)
        self.local_map_model.network.load_state_dict(map_net_dict['model'])

    def map_run(self,seed):
        with torch.no_grad():
            self.env_map.global_reset(True,seed)
            recordings = [[] for _ in range(self.env_map.num_agents)]
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
                for i in range(len(self.env_map.agent_poss)):
                    recordings[i].append(self.env_map.agent_poss[i])
                if rl_local_restart and not map_done:
                    self.env_map.local_reset()
            throughput=self.env_map.all_finished_task/self.env_map.episode_len        
        return throughput,rl_time,recordings

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process maze and model parameters')

    parser.add_argument('--map_path', 
                        type=str,
                        default='Proportion_Maze_26_26_3',
                        help='Origin path for the maze')
    
    parser.add_argument('--model_path',
                        type=str,
                        default='26_26_3',
                        help='Path for the model')
    
    parser.add_argument('--selector',
                        type=str,
                        choices=['random', 'pibt', 'BC', 'all'],
                        default='all',
                        help='Selector type to use')
    
    parser.add_argument('--probability',
                        type=float,
                        default=1.0,
                        help='probability for selecting the vertex (default: 1.0)')
    
    parser.add_argument('--record',
                        action='store_true',
                        help='Enable recording (optional)')
    
    parser.add_argument('--record_path',
                        type=str,
                        help='Suffix for recording path (optional)')
   
    parser.add_argument('--step',
                        type=int,
                        default=5120,
                        help='Number of max steps (default: 1000)')
    
    parser.add_argument('--eval_times',
                        type=int,
                        default=5,
                        help='Number of eval time (default: 5)')
    args = parser.parse_args()

    path = f"maps/{args.map_path}.txt"
    print(f"map path: {path}")  
    print(f"model_path: {args.model_path}")

    selector_dict = {
        "random": random_selector,
        "pibt": pibt_selector,
        "BC": BC_selector,
        "all": selector
    }
    select = selector_dict[args.selector]
    print(f"select: {select.__name__}")  
    print(f"probability: {args.probability}")
    print(f"step: {args.step}")
    
    record_path = None
    if args.record:
        if args.record_path:
            record_path = args.record_path
        else:
            record_path = f"./recordings/{args.map_path}_{select.__name__}_{args.probability}.path"
        print(f"record_path: {record_path}")    
    return args,path, args.model_path, select, args.probability, args.step, args.record,record_path,args.eval_times

if __name__ == "__main__":   
    args,map_path, model_path, select, probability, RUN_STEP,record, record_path,EVAL_TIMES = parse_arguments()
    print("[{}] begin eval rl select {} using {}".format(datetime.datetime.now(),args.map_path,select.__name__))
    _,_,map = read_map(map_path)
    selected_vertex = select(map,probability,file_name=args.map_path).select()
    print("[{}] finish selected_vertex".format(datetime.datetime.now()))
    env = Runner(0,selected_vertex,model_path,args.map_path)
    throughputs=[]
    for eval_time in range(EVAL_TIMES):  # 0 wait ,1 right, 2 down, 3 left, 4 up
        throughput,rl_time,recordings = env.map_run(eval_time*123)
        print('[{}] evaluation times:{}'.format(datetime.datetime.now(),eval_time))
        print('throughput:{} rl time:{} pibt time: {}'.format(throughput,rl_time,env.env_map.pibt_time))
        throughputs.append(throughput)
    if record:
        with open(record_path,"w") as f:
            for i in range(len(recordings)):
                recording = recordings[i]
                f.write("Agent "+str(i)+": ")
                for pos in recording:
                    f.write("("+str(pos[0])+","+str(pos[1])+")->")
                f.write("\n")
            
    throughput_std=np.std(throughputs)
    throughput_mean=np.mean(throughputs)
    print("mean throughput:{}, std throughput:{}".format(throughput_mean,throughput_std))
    print()
    
