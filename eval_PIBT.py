import numpy as np
from CO_mapf_gym_pibt import CO_MAPFEnv
from util import set_global_seeds
import sys
import datetime
RUN_STEP=1000
RECORD = False
EVAL_TIMES=1 if RECORD else 5
# recording path and collide times

class Runner(object):
    def __init__(self, env_id,file_name=None):
        """initialize model and environment"""
        self.ID = env_id
        set_global_seeds(env_id*123)
        self.env_map= CO_MAPFEnv(env_id,RUN_STEP,file_name)

    def map_run(self,seed):
        self.env_map.global_reset_fix(seed)
        recordings = [[] for _ in range(self.env_map.num_agents)]        
        map_done = False
        while map_done==False:
            map_done, rl_local_restart = self.env_map.joint_step()
            for i in range(len(self.env_map.agent_poss)):
                    recordings[i].append(self.env_map.agent_poss[i])
            if rl_local_restart and not map_done:
                self.env_map.local_reset()
        throughput=self.env_map.all_finished_task/self.env_map.episode_len
        return throughput,self.env_map.global_path,self.env_map.collide_times,recordings

if __name__ == "__main__":
    if len(sys.argv) > 1 :
        env = Runner(0,sys.argv[1])
    else:
        env = Runner(0)
    if RECORD:
        if len(sys.argv) > 2:
            record_path = "./recordings"+sys.argv[2]+".txt"
        else:
            record_path = "./recordings/"+sys.argv[1]+"_pibt.txt"
        print("record_path:{}".format(record_path))
    
    throughputs=[]
    print('start evaluation {} at:{}'.format(env.env_map.path,datetime.datetime.now()))
    for eval_time in range(EVAL_TIMES):  # 0 wait ,1 right, 2 down, 3 left, 4 up
        throughput, paths, collide_times,recordings = env.map_run(eval_time*123)
        print('[{}] evaluation times:{}'.format(datetime.datetime.now(),eval_time))
        print('throughput:{}'.format(throughput))
        throughputs.append(throughput)
    if RECORD:
        with open(record_path,"w") as f:
            for i in range(len(recordings)):
                recording = recordings[i]
                f.write("Agent "+str(i)+": ")
                for pos in recording:
                    f.write("("+str(pos[0])+","+str(pos[1])+")->")
                f.write("\n")
    throughput_std=np.std(throughputs)
    throughput_mean=np.mean(throughputs)
    print("mean throughput:{}, std throughput:{}\n".format(throughput_mean,throughput_std))
    print()