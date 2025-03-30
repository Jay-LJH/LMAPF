import os
from util import *
import numpy as np
import torch
from alg_parameters import *
from map_model import MapModel
import sys
from CO_mapf_gym_file import CO_MAPFEnv
import datetime
import argparse
RUN_TIME = 10
record = False
EVAL_TIMES=1 if record else 10
# recording path and collide times

class Runner(object):
    def __init__(self, env_id, model_path=None,file_name=None):
        self.ID = env_id
        set_global_seeds(env_id*123)
        self.env_map= CO_MAPFEnv(env_id,file_name)
        self.local_device = torch.device('cuda') if SetupParameters.USE_GPU_LOCAL else torch.device('cpu')
        self.local_map_model = MapModel(env_id, self.local_device)
        self.K = runParameters.N_NODE
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
            map_obs= self.env_map.observe_for_map()
            map_done = False
            rl_time = 0
            init_time = datetime.datetime.now()
            while map_done==False and (datetime.datetime.now()-init_time).total_seconds() < RUN_TIME:
                before = datetime.datetime.now()
                map_action, _, _, map_hidden_state = self.local_map_model.step(map_obs,map_hidden_state,self.K)
                rl_time += (datetime.datetime.now()-before).total_seconds()
                map_done,reward, map_obs = self.env_map.joint_step(map_action)
                for i in range(len(self.env_map.agent_poss)):
                    recordings[i].append(self.env_map.agent_poss[i])
            total_time = (datetime.datetime.now()-init_time).total_seconds()                   
        return map_done, rl_time,recordings,total_time

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process maze and model parameters')

    parser.add_argument('--map_path', 
                        type=str,
                        default='Proportion_Maze_26_26_3',
                        help='Origin path for the maze')
    
    parser.add_argument('--model_path',
                        type=str,
                        default='26_26_3_003',
                        help='Path for the model')
    
    parser.add_argument('--record',
                        action='store_true',
                        help='Enable recording (optional)')
    
    parser.add_argument('--record_path',
                        type=str,
                        help='Suffix for recording path (optional)')
   
    parser.add_argument('--max_time',
                        type=int,
                        default=100,
                        help='max running time for each epoch (default: 1 sec)')
    
    parser.add_argument('--eval_times',
                        type=int,
                        default=10,
                        help='Number of eval time (default: 10)')
    args = parser.parse_args()

    path = f"maps/{args.map_path}.txt"
    print(f"map path: {path}")  
    print(f"model_path: {args.model_path}")
    RUN_TIME = args.max_time
    
    return args,path, args.model_path,args.record,args.record_path,args.eval_times

if __name__ == "__main__":   
    args,map_path, model_path, record,record_path,EVAL_TIMES = parse_arguments()
    _,_,map = read_map(map_path)
    env = Runner(0,model_path,args.map_path)
    successs=[]
    runtimes = []
    for eval_time in range(EVAL_TIMES):  # 0 wait ,1 right, 2 down, 3 left, 4 up
        success,rl_time,recordings,total_time = env.map_run(eval_time*123)
        print('[{}] evaluation times:{}'.format(datetime.datetime.now(),eval_time))
        print('success:{} rl time:{} pibt time: {} total time: {}'.format(success,rl_time,env.env_map.pibt_time,total_time))
        print("running steps: ",env.env_map.time_step)
        successs.append(success)
        if success:
            runtimes.append(total_time)
    if record:
        print("saving")
        with open(record_path,"w") as f:
            for i in range(len(recordings)):
                recording = recordings[i]
                f.write("Agent "+str(i)+": ")
                for pos in recording:
                    f.write("("+str(pos[0])+","+str(pos[1])+")->")
                f.write("\n")
            
    success_time=np.sum(successs)
    success_rate=np.mean(successs)
    print("success rate: ",success_rate)
    print("success time: ",success_time)
    print("average success time: ",np.mean(runtimes))
    print()
    
