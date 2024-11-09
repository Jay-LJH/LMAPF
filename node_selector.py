from map_generator import *
from alg_parameters import *
import random
from util import set_global_seeds
import networkx as nx
from map_generator import *
from CO_mapf_gym_pibt import CO_MAPFEnv
RUN_STEP=1000
EVAL_TIMES=5

class selector:
    def __init__(self,map,probaility,seed=42,file_name=None):
        self.map = map
        self.probaility = probaility
        set_global_seeds(seed)
        self.nodes = []
        for i in range(self.map.shape[0]):
            for j in range(self.map.shape[1]):
                if self.map[i][j] == 0:
                    self.nodes.append((i,j))

    # base class just return all nodes
    def select(self):
        return self.nodes
    
class random_selector(selector):
    def __init__(self,map,probability,seed=42,file_name=None):
        super().__init__(map,probability,seed)
        self.len = int(len(self.nodes)*probability)

    # select nodes randomly
    def select(self):
        return random.sample(self.nodes, self.len)

class pibt_selector(selector):
    # RAII, run pibt when init
    def __init__(self,map,probability,seed=42,file_name=None):
        super().__init__(map,probability,seed)
        self.len = int(len(self.nodes)*probability)
        self.visit_cnt = np.zeros_like(self.map)
        for i in range(EVAL_TIMES):
            print("pibt eval time: ",i)
            self.env = CO_MAPFEnv(i,RUN_STEP,file_name)
            self.env.global_reset_fix(seed+i*123)
            map_done = False
            while map_done==False:
                map_done, rl_local_restart = self.env.joint_step()
                for poss in self.env.agent_poss:
                    self.visit_cnt[poss] += 1
                if rl_local_restart and not map_done:
                    self.env.local_reset()

    def select(self):
        flatten = self.visit_cnt.flatten()
        indices = np.argsort(flatten)[-self.len:]
        indices = np.unravel_index(indices, self.visit_cnt.shape)
        return list(zip(indices[0],indices[1]))
    
class BC_selector(selector):
    def __init__(self,map,probability,seed=42,file_name=None):
        super().__init__(map,probability,seed)
        self.len = int(len(self.nodes)*probability)
        G = nx.Graph()
        for i in range(self.map.shape[0]):
            for j in range(self.map.shape[1]):
                if self.map[i][j] == 0:
                    G.add_node((i,j))
                    if i>0 and self.map[i-1][j] == 0:
                        G.add_edge((i,j),(i-1,j))
                    if j>0 and self.map[i][j-1] == 0:
                        G.add_edge((i,j),(i,j-1))
        self.G = G
        self.bc = nx.betweenness_centrality(G)

    def select(self):
        target = sorted(self.bc.items(), key=lambda item: item[1], reverse=True)[:self.len]
        return [key for key, _ in target]
    
if __name__ == "__main__":
    maze = Maze(50, 50,seed=42, obstacle_rate=0.5,pad=True)
    maze.serialize()
    print("map:")
    print(maze.matrix)
    print("random select:")
    print(random_selector(maze.matrix,0.2).select())
    print("bc select:")
    print(BC_selector(maze.matrix,0.2).select())