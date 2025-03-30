import numpy as np
from CO_mapf_gym_pibt import CO_MAPFEnv
from util import set_global_seeds
import sys
import datetime
MAX_STEP = 1000
MAX_TIME = 10
RECORD = False
EVAL_TIMES=1 if RECORD else 10
# recording path and collide times

class Runner(object):
    def __init__(self, env_id,file_name=None):
        """initialize model and environment"""
        self.ID = env_id
        set_global_seeds(env_id*123)
        self.env_map= CO_MAPFEnv(env_id,file_name)

    def map_run(self,seed):
        self.env_map.global_reset_fix(seed)       
        map_done = False
        init_time = datetime.datetime.now()
        while map_done==False and (datetime.datetime.now()-init_time).total_seconds() < MAX_TIME:
            map_done = self.env_map.joint_step()
        return map_done,self.env_map.time_step, (datetime.datetime.now()-init_time).total_seconds()

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
    
    print('start evaluation pibt {} at:{}'.format(env.env_map.path,datetime.datetime.now()))
    successs=[]
    runtimes = []
    for eval_time in range(EVAL_TIMES):  # 0 wait ,1 right, 2 down, 3 left, 4 up
        done,run_step,total_time = env.map_run(eval_time*123)
        successs.append(done)
        if done:
            runtimes.append(total_time)
        print('[{}] evaluation times:{}'.format(datetime.datetime.now(),eval_time))
        print('success:{}  pibt time: {} total time: {}'.format(done,env.env_map.pibt_time,total_time))
        print("running steps: ",env.env_map.time_step)

    success_time=np.sum(successs)
    success_rate=np.mean(successs)
    print("success rate: ",success_rate)
    print("success time: ",success_time)
    print("average success time: ",np.mean(runtimes))
    print()