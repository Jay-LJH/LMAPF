import numpy as np
from CO_mapf_gym import CO_MAPFEnv
from util import set_global_seeds
import sys
import datetime
from alg_parameters import *
RUN_STEP = 1000
RECORD = False
EVAL_TIMES=1 if RECORD else 5
# recording path and collide times

class Runner(object):
    def __init__(self, env_id,file_name=None):
        """initialize model and environment"""
        self.ID = env_id
        set_global_seeds(env_id*123)
        self.env_map= CO_MAPFEnv(env_id,RUN_STEP)

    def map_run(self,seed):
        self.env_map.global_reset_fix(seed)
        recordings = [[] for _ in range(self.env_map.num_agents)]        
        map_done = False
        while map_done==False:
            map_done, rl_local_restart,_,_,_ = self.env_map.joint_step()
            for i in range(len(self.env_map.agent_poss)):
                    recordings[i].append(self.env_map.agent_poss[i])
            if rl_local_restart and not map_done:
                self.env_map.local_reset()
        throughput=self.env_map.all_finished_task/self.env_map.episode_len
        return throughput

if __name__ == "__main__":

    throughputs=[]
    env = Runner(0)
    for eval_time in range(EVAL_TIMES):  # 0 wait ,1 right, 2 down, 3 left, 4 up
        throughput = env.map_run(eval_time*123)
        print('[{}] evaluation times:{}'.format(datetime.datetime.now(),eval_time))
        print('throughput:{}'.format(throughput))
        throughputs.append(throughput)
    throughput_std=np.std(throughputs)
    throughput_mean=np.mean(throughputs)
    print("mean throughput:{}, std throughput:{}\n".format(throughput_mean,throughput_std))
    print()