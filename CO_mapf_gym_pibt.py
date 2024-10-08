import gym
import numpy as np
from im_function_PIBT_4.build import lifelong_pibt_4
from alg_parameters import *
import os
from world_property import State
import itertools

MAPdirDict = {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1), 4: (-1, 0)}  #  0 wait ,1 right, 2 down, 3 left, 4 up
actionDict = {v: k for k, v in MAPdirDict.items()}

numbers = [0, 1, 2, 3, 4]
actions_combination_list = list(itertools.permutations(numbers, 3))


class CO_MAPFEnv(gym.Env):
    def __init__(self,env_id, episode_len,gap=EnvParameters.GAP):
        """initialization"""
        self.obstacle_gap = gap
        self.episode_len=episode_len
        self.env_id=env_id
        self.project_path = os.getcwd() + "/h_maps"
        self.num_agents=EnvParameters.N_AGENT
        self.world_high = EnvParameters.WORLD_HIGH
        self.world_wide = EnvParameters.WORLD_WIDE
        self.build_sorting_map()
        self.build_guide_map()

    def build_sorting_map(self):
        map_1 = np.zeros((self.world_high, self.world_wide),dtype=np.int32)
        map_2 = np.zeros((self.world_high, self.world_wide),dtype=np.int32)
        self.station_map = np.zeros((self.world_high, self.world_wide),dtype=np.int32)
        for row in range(3, self.world_high - 3, self.obstacle_gap):
            map_1[row, 1:self.world_wide - 1] = -2
        for col in range(2, self.world_wide - 2, self.obstacle_gap):
            map_2[2:self.world_high - 2, col] = -2
        self.total_map = map_1 + map_2
        self.total_map[self.total_map == -4] = -1
        for i in range(self.world_wide):
            if self.total_map[2, i] == -2: # induct
                self.total_map[0, i] = -3
                self.total_map[-1, i] = -3

        eject_station_id=10000
        induct_station_id=1000
        for i in range(self.world_high):
            for j in range(self.world_wide):
                if self.total_map[i,j]==-1:
                    self.station_map[i,j+1]= eject_station_id
                    self.station_map[i, j - 1] = eject_station_id
                    self.station_map[i-1, j] = eject_station_id
                    self.station_map[i+1, j] = eject_station_id
                    eject_station_id+=1
                    assert (self.total_map[i,j+1]==-2 and self.total_map[i, j - 1] == -2 and self.total_map[i-1, j] == -2 and self.total_map[i+1, j] == -2)
                if self.total_map[i,j]==-3:
                    self.station_map[i,j]= induct_station_id
                    induct_station_id+=1

        self.total_map[0, 0] = self.total_map[0, -1] = self.total_map[-1, 0] = self.total_map[-1, -1] = -1
        self.obstacle_map = np.zeros((self.world_high, self.world_wide),dtype=np.int32)
        self.eject_map = np.zeros((self.world_high, self.world_wide),dtype=np.int32)
        self.induct_map = np.zeros((self.world_high, self.world_wide),dtype=np.int32)
        self.obstacle_map[self.total_map == -1] = 1
        self.eject_map[self.total_map == -2] = 1
        self.induct_map[self.total_map == -3] = 1
        self.eject_induct_map = self.eject_map + self.induct_map

    def build_guide_map(self):
        self.node_poss = []
        self.obs_range = []
        self.node_index_dict={}
        self.nearby_node_dict = {}
        node_index = 0
        for i in range(self.world_high):
            for j in range(self.world_wide):
                if self.obstacle_map[i, j] != 1:
                    top_poss = max(i- CopParameters.FOV // 2, 0)
                    bottom_poss = min(i + CopParameters.FOV // 2 + 1, self.world_high)
                    left_poss = max(j - CopParameters.FOV // 2, 0)
                    right_poss = min(j + CopParameters.FOV // 2 + 1, self.world_wide)
                    FOV_top, FOV_left = max(CopParameters.FOV // 2 - i, 0), max(
                        CopParameters.FOV // 2 -j, 0)
                    FOV_bottom, FOV_right = FOV_top + (bottom_poss - top_poss), FOV_left + (right_poss - left_poss)
                    top_left = (i - CopParameters.FOV // 2, j - CopParameters.FOV // 2)
                    self.obs_range.append((FOV_top,FOV_bottom, FOV_left,FOV_right,top_poss,bottom_poss,left_poss,right_poss,top_left))
                    self.node_poss.append((i, j))
                    self.node_index_dict[(i,j)]=node_index
                    node_index+=1

        for node_index, node in enumerate(self.node_poss):
            self.nearby_node_dict[node_index] = []
            near_by_grid=[(node[0]-1,node[1]),(node[0]+1,node[1]),(node[0],node[1]-1),(node[0],node[1]+1)]
            for item in near_by_grid:
                if item in self.node_index_dict.keys():
                    self.nearby_node_dict[node_index].append(self.node_index_dict[item])

    def global_reset_fix(self, seed):
        self.rhcr=lifelong_pibt_4.RHCR_class_pibt_learn(seed, self.num_agents, self.world_high, self.world_wide,self.env_id,self.total_map, self.station_map,self.project_path)
        self.rhcr.update_start_goal(EnvParameters.H)
        self.agent_state=np.zeros((self.world_high,self.world_wide))
        self.agent_poss=self.rhcr.rl_agent_poss
        for index, poss in enumerate(self.agent_poss):
            self.agent_state[poss] += 1
        # set recording to 0
        self.time_step,self.local_time_step,self.all_finished_task=0,0,0
        self.goals_id= np.zeros(self.num_agents, dtype=np.int32)
        self.true_path=[[self.agent_poss[i]] for i in range(self.num_agents)]
        self.world=State(self.world_high,self.world_wide,self.node_poss)
        map_location=self.project_path+"/"+ str(self.env_id)+str(self.world_high)+str(self.world_wide)+"py_h_map.npy"
        try:
            with open(map_location, 'rb') as f:
                self.world.heuristic_map = np.load(f, allow_pickle=True).item()
                self.world.all_priority= np.load(f, allow_pickle=True).item()
                self.world.all_h_map = np.load(f, allow_pickle=True)
        except FileNotFoundError:
            heuristic_map=self.rhcr.obtaion_heuri_map()
            self.world.convert_all_heuri_map(heuristic_map,self.obstacle_map,map_location) #convert and ssave
        self.elapsed=np.zeros(self.num_agents)
        self.global_path = [[self.agent_poss[i]] for i in range(self.num_agents)]
        self.collide_times = []
        return

    def local_reset(self):
        succ=self.rhcr.update_system(self.true_path)
        self.rhcr.update_start_goal(EnvParameters.H)
        self.goals_id = np.zeros(self.num_agents,dtype=np.int32)
        self.local_time_step=0
        self.true_path=[[self.agent_poss[i]] for i in range(self.num_agents)]
        return True

    def joint_move(self):
        self.agent_state = np.zeros((self.world_high, self.world_wide))
        coll_times=self.rhcr.run_pibt() # run PIBT without the guidance form RL
        coll_map=np.zeros((self.world_high,self.world_wide))
        self.agent_poss=self.rhcr.rl_path

        # update final status
        for i in range(self.num_agents):
            self.elapsed[i] +=1
            self.agent_state[self.agent_poss[i]] += 1
            self.true_path[i].append(self.agent_poss[i])
            self.global_path[i].append(self.agent_poss[i])
            coll_map[self.agent_poss[i]] += coll_times[i]

            if self.agent_poss[i] == self.rhcr.rl_agent_goals[i][self.goals_id[i]]:
                self.goals_id[i]+=1
                self.all_finished_task+=1
                self.elapsed[i]=0
        assert len(np.argwhere(self.agent_state>1)) == 0
        assert len(np.argwhere(self.agent_state < 0)) == 0
        self.collide_times.append(coll_map)
        return

    def joint_step(self):
        self.time_step+=1
        self.local_time_step+=1
        self.joint_move()
        if self.time_step >= self.episode_len:
            done = True
        else:
            done = False
        if self.local_time_step>= EnvParameters.H:
            local_done=True
        else:
            local_done=False
        return done, local_done


    def get_action(self, direction):
        return actionDict[direction]




