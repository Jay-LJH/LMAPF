import numpy as np
from CO_mapf_gym_pibt import CO_MAPFEnv
from util import set_global_seeds

RUN_STEP=5120
EVAL_TIMES=10
# recording path and collide times

class Runner(object):
    def __init__(self, env_id):
        """initialize model and environment"""
        self.ID = env_id
        set_global_seeds(env_id*123)
        self.env_map= CO_MAPFEnv(env_id,RUN_STEP)

    def map_run(self,seed):
        self.env_map.global_reset_fix(seed)
        map_done = False
        while map_done==False:
            map_done, rl_local_restart = self.env_map.joint_step()
            if rl_local_restart and not map_done:
                self.env_map.local_reset()
        throughput=self.env_map.all_finished_task/self.env_map.episode_len
        return throughput,self.env_map.global_path,self.env_map.collide_times

if __name__ == "__main__":
    import os
    if not os.path.exists("./h_maps"):
        os.makedirs("./h_maps")

    if not os.path.exists("./recordings"):
        os.makedirs("./recordings")

    map_location="./recordings/pibt_paths.npy"
    env = Runner(0)
    throughputs=[]
    for eval_time in range(EVAL_TIMES):  # 0 wait ,1 right, 2 down, 3 left, 4 up
        throughput, paths, collide_times = env.map_run(eval_time*123)
        print('evaluation times:{}'.format(eval_time))
        print('throughput:{}'.format(throughput))
        throughputs.append(throughput)
        with open(map_location, 'ab') as f:
            np.save(f, paths)
            np.save(f,collide_times)

    throughput_std=np.std(throughputs)
    throughput_mean=np.mean(throughputs)
    print("mean throughput:{}, std throughput:{}".format(throughput_mean,throughput_std))
    print("test")
