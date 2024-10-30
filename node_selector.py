from map_generator import *
from alg_parameters import *
from CO_mapf_gym_pibt import CO_MAPFEnv
import random
from util import set_global_seeds
import networkx as nx

RUN_STEP=5120
EVAL_TIMES=10

class selector:
    def __init__(self,map,probaility,seed=42):
        self.map = map
        self.probaility = probaility
        set_global_seeds(seed)

    # base class just return all nodes
    def select(self):
        return self.map.nodes
    
class random_selector(selector):
    def __init__(self,map,probability,seed=42):
        super().__init__(map,probability,seed)
        self.len = int(len(self.map.nodes)*probability)

    # select nodes randomly
    def select(self):
        return random.sample(self.map.nodes, len)

class pibt_selector(selector):
    # RAII, run pibt when init
    def __init__(self,map,probability,seed=42):
        super().__init__(map,probability,seed)
        self.len = int(len(self.map.nodes)*probability)
        self.visit_cnt = np.zeros((self.map.width,self.map.height))
        for i in range(EVAL_TIMES):
            print("pibt eval time: ",i)
            self.env = CO_MAPFEnv(i,RUN_STEP)
            self.env.global_reset_fix(seed+i*123)
            map_done = False
            while map_done==False:
                map_done, rl_local_restart = self.env_map.joint_step()
                for poss in self.env.agent_poss:
                    self.visit_cnt[poss] += 1
                if rl_local_restart and not map_done:
                    self.env_map.local_reset()

    def select(self):
        flatten = self.visit_cnt.flatten()
        indices = np.argsort(flatten)[-self.len:]
        indices = np.unravel_index(indices, self.visit_cnt.shape)
        return list(zip(indices[0],indices[1]))
    
class BC_selector(selector):
    def __init__(self,map,probability,seed=42):
        super().__init__(map,probability,seed)
        self.len = int(len(self.map.nodes)*probability)
        G = nx.Graph()
        

    def select(self):
        