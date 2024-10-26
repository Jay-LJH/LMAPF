import copy

import gym
import numpy as np
from im_function_PIBT_1.build import lifelong_pibt_1
from alg_parameters import *
from collections import deque
import os
import random
from world_property import State
import itertools
# 0 wait ,1 right, 2 down, 3 left, 4 up
MAPdirDict = {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1), 4: (-1, 0)} 
actionDict = {v: k for k, v in MAPdirDict.items()}

numbers = [0, 1, 2, 3, 4]
actions_combination_list = list(itertools.permutations(numbers, 3))


class CO_MAPFEnv(gym.Env):
    """map MAPF problems to a standard RL environment"""
    def __init__(self,env_id,path="maps/Maze_25_25.txt"):      
        self.induct_value = -3
        self.eject_value = -2
        self.obstacle_value = -1
        self.travel_value = 0
        self.env_id=env_id
        self.project_path = os.getcwd() + "/h_maps"
        self.num_agents=EnvParameters.N_AGENT
        self.world_high,self.world_wide,self.total_map=self.read_map(path)
        self.finished_task=0
        self.wait_map=np.zeros((self.world_high,self.world_wide))
        self.build_sorting_map()
        self.build_guide_map()
        
    def read_map(self,file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
        dimensions = lines[0].strip().split()
        rows = int(dimensions[0])
        cols = int(dimensions[1])

        map_data = []
        for line in lines[1:rows+1]:
            map_data.append(list(line.strip()))
        for i in range(rows):
            for j in range(cols):
                if map_data[i][j] == '@':
                    map_data[i][j] = self.obstacle_value
                elif map_data[i][j] == 'e':
                    map_data[i][j] = self.eject_value
                elif map_data[i][j] == 'i':
                    map_data[i][j] = self.induct_value
                else:
                    map_data[i][j] = self.travel_value
        return rows, cols, np.array(map_data,dtype=np.int32)
    
    def build_sorting_map(self):
        self.station_map = np.zeros((self.world_high, self.world_wide),dtype=np.int32)
        eject_station_id=10000
        induct_station_id=1000
        # -1: obstacle, -2: eject station, -3: induct station
        for i in range(self.world_high):
            for j in range(self.world_wide):
                if self.total_map[i, j] == self.eject_value:
                    self.station_map[i, j] = eject_station_id
                    eject_station_id+=1
                elif self.total_map[i, j] == self.induct_value:
                    self.station_map[i, j] = induct_station_id
                    induct_station_id+=1
        self.obstacle_map = np.zeros((self.world_high, self.world_wide),dtype=np.int32)
        self.eject_map = np.zeros((self.world_high, self.world_wide),dtype=np.int32)
        self.induct_map = np.zeros((self.world_high, self.world_wide),dtype=np.int32)
        self.obstacle_map[self.total_map == self.obstacle_value] = 1
        self.eject_map[self.total_map == self.eject_value] = 1
        self.induct_map[self.total_map == self.induct_value] = 1
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

    # rand: whether to use a fixed seed 
    # seed: the seed for random
    def global_reset(self,rand = False,seed=42):
        if rand:
            seed = random.randint(0, 100000)
        self.rhcr=lifelong_pibt_1.RHCR_maze(seed, EnvParameters.N_AGENT, 1,self.total_map, ".")
        self.rhcr.update_start_goal(EnvParameters.H)
        # indicate the number of robots in each grids
        self.agent_state=np.zeros((self.world_high,self.world_wide))  
        self.agent_poss=self.rhcr.rl_agent_poss
        for index, poss in enumerate(self.agent_poss):
            self.agent_state[poss] += 1
        # set recording to 0
        self.time_step,self.local_time_step,self.all_finished_task=0,0,0
        self.goals_id= np.zeros(self.num_agents, dtype=np.int32)
        self.true_path=[[self.agent_poss[i]] for i in range(self.num_agents)]
        self.uti_deque = deque(maxlen=CopParameters.UTIL_T)
        self.all_wait_map = np.zeros((self.world_high, self.world_wide))
        # handle heuristic value related things
        self.world=State(self.world_high,self.world_wide,self.node_poss) 
        heuristic_map=self.rhcr.get_heuri_map()
        self.world.convert_all_heuri_map(heuristic_map,self.obstacle_map) #convert and save
        self.elapsed=np.zeros(self.num_agents) # for PIBT, control the priority of robots
        return

    def local_reset(self):  # update goals
        succ=self.rhcr.update_system(self.true_path)
        self.rhcr.update_start_goal(EnvParameters.H)
        self.goals_id = np.zeros(self.num_agents,dtype=np.int32)
        self.local_time_step=0
        self.true_path=[[self.agent_poss[i]] for i in range(self.num_agents)]
        return True

    def joint_move(self,map_action):
        past_position = copy.copy(self.agent_poss)
        self.agent_state = np.zeros((self.world_high, self.world_wide))
        action = np.zeros(self.num_agents, dtype=np.int32)
        agent_on_grid = np.zeros(CopParameters.N_NODE)
        team_agent_on_grid = np.zeros(CopParameters.N_NODE)
        rewards = np.zeros((1, CopParameters.N_NODE), dtype=np.float32)
        team_rewards = np.zeros((1, CopParameters.N_NODE), dtype=np.float32)

        action_guidance=np.zeros((self.num_agents,CopParameters.MAP_ACTION),dtype=np.int32) # the bigger the number the high the priority
        # transfer action
        for i in range(self.num_agents):
            ax = self.agent_poss[i][0]
            ay = self.agent_poss[i][1]
            node_index = self.node_index_dict[(ax,ay)]
            act_order=actions_combination_list[int(map_action[node_index])]
            action_guidance[i,act_order[0]]=CopParameters.TOP_NUM
            action_guidance[i, act_order[1]] = CopParameters.TOP_NUM-1
            action_guidance[i, act_order[2]] = CopParameters.TOP_NUM - 2
        # solve conflict by PIBT
        coll_times=self.rhcr.run_pibt(action_guidance)
        self.agent_poss=self.rhcr.rl_path
        uti_map = np.zeros((5, self.world_high, self.world_wide))
        # update final status
        for i in range(self.num_agents):
            self.elapsed[i] +=1
            action[i]=self.get_action((self.agent_poss[i][0]-past_position[i][0],self.agent_poss[i][1]-past_position[i][1]))
            self.agent_state[self.agent_poss[i]] += 1
            uti_map[int(action[i]), self.agent_poss[i][0], self.agent_poss[i][1]] += 1
            self.true_path[i].append(self.agent_poss[i])
            prioirty=self.world.all_priority[self.rhcr.rl_agent_goals[i][self.goals_id[i]]][past_position[i]]
            node_index=self.node_index_dict[past_position[i]]
            agent_on_grid[node_index] += 1
            rewards[:, node_index] += CopParameters.COLL_R * coll_times[i]
            if self.agent_poss[i] in prioirty[0]:
                rewards[:, node_index] += CopParameters.BEST_R
            elif self.agent_poss[i] in prioirty[1]:
                rewards[:, node_index] += CopParameters.WORSE_R
            else:
                rewards[:, node_index] += CopParameters.SECOND_R
            if action[i] == 0:
                self.wait_map[self.agent_poss[i][0], self.agent_poss[i][1]] += 1
                self.all_wait_map[self.agent_poss[i][0], self.agent_poss[i][1]] += 1
            if self.agent_poss[i] == self.rhcr.rl_agent_goals[i][self.goals_id[i]]:
                self.finished_task+=1
                self.goals_id[i]+=1
                self.all_finished_task+=1
                self.elapsed[i]=0

        self.uti_deque.append(uti_map)
        for node_index in range(CopParameters.N_NODE):
            for neigbor in self.nearby_node_dict[node_index]:
                team_rewards[:,node_index]+=rewards[:,neigbor]
                team_agent_on_grid[node_index]+=agent_on_grid[neigbor]
        team_rewards=np.divide(team_rewards, team_agent_on_grid, out=np.zeros_like(team_rewards, dtype=np.float32), where=team_agent_on_grid!=0)
        rewards=rewards+CopParameters.TEAM_REWARD*team_rewards
        return rewards

    def joint_step(self,map_action):
        """execute joint action and obtain reward"""
        self.time_step+=1
        self.local_time_step+=1
        rewards=self.joint_move(map_action)
        actor_obs,actor_vec = self.observe_for_map()
        if self.time_step >= CopParameters.EPISODE_LEN:
            done = True
        else:
            done = False
        if self.local_time_step>= EnvParameters.H:
            local_done=True
        else:
            local_done=False
        return done, local_done, rewards, actor_obs,actor_vec

    def observe_for_map(self):
        map_obs = np.zeros((1, CopParameters.N_NODE,CopParameters.OBS_CHANNEL, CopParameters.FOV, CopParameters.FOV), dtype=np.float32)
        map_vector=np.expand_dims(np.eye(CopParameters.N_NODE,dtype=np.float32), axis=0)
        if self.time_step != 0:
            sum_util_map = np.sum(self.uti_deque, axis=0)
            sum_util_map = 40 * sum_util_map / self.num_agents
        else:
            sum_util_map = np.zeros((5, self.world_high, self.world_wide))
        all_first_map = np.zeros((EnvParameters.WORLD_HIGH, EnvParameters.WORLD_WIDE))
        all_worse_map = np.zeros((EnvParameters.WORLD_HIGH, EnvParameters.WORLD_WIDE))
        all_order_map= np.zeros((EnvParameters.WORLD_HIGH, EnvParameters.WORLD_WIDE))
        agents_order = [i for i in range(self.num_agents)]      
        agents_order.sort(key=lambda x: (self.world.heuristic_map[self.rhcr.rl_agent_goals[x][self.goals_id[x]]][
                                          self.agent_poss[x][0] * self.world_wide + self.agent_poss[x][1]], -self.elapsed[x],
                                      -self.rhcr.tie_breaker[x])) # the former has higher priority
        agent_first_map = np.zeros((1,CopParameters.N_NODE, CopParameters.FOV, CopParameters.FOV))
        agent_worse_map = np.zeros((1,CopParameters.N_NODE, CopParameters.FOV, CopParameters.FOV))
        for visible_ag, poss in enumerate(self.agent_poss):
            node_index=self.node_index_dict[poss]
            _, _, _, _,_,_,_,_,top_left = self.obs_range[node_index]
            all_order_map[poss[0], poss[1]] = agents_order.index(visible_ag)+1
            ag_goal = self.rhcr.rl_agent_goals[visible_ag][self.goals_id[visible_ag]]
            first_poss = self.world.all_priority[ag_goal][poss][0]
            worse_poss = self.world.all_priority[ag_goal][poss][1]
            for first in first_poss:
                all_first_map[first[0], first[1]] += 1
                agent_first_map[0,node_index, first[0]-top_left[0] ,first[1]-top_left[1]]=1 
            for worse in worse_poss:
                all_worse_map[worse[0], worse[1]] += 1
                agent_worse_map[0,node_index, worse[0]-top_left[0], worse[1]-top_left[1]] = 1
        
        all_order_map/= self.num_agents
        for node in range(CopParameters.N_NODE):
            FOV_top, FOV_bottom, FOV_left, FOV_right, top_poss, bottom_poss, left_poss, right_poss,_ = self.obs_range[node]
            obs_map = np.ones((1, CopParameters.FOV, CopParameters.FOV))
            induct_eject_map = np.zeros((1, CopParameters.FOV, CopParameters.FOV))
            agent_map = np.zeros((1,  CopParameters.FOV, CopParameters.FOV))
            first_map = np.zeros((1, CopParameters.FOV, CopParameters.FOV))
            worse_map = np.zeros((1, CopParameters.FOV, CopParameters.FOV))
            all_h_map = np.zeros((4, CopParameters.FOV, CopParameters.FOV))
            util_map = np.zeros((5, CopParameters.FOV, CopParameters.FOV))
            order_map = np.zeros((1, CopParameters.FOV, CopParameters.FOV))
            my_first_map = agent_first_map[:,node,:,:]
            my_worse_map = agent_worse_map[:,node,:,:]

            obs_map[:, FOV_top:FOV_bottom, FOV_left:FOV_right] = self.obstacle_map[top_poss:bottom_poss, left_poss:right_poss]
            induct_eject_map[:,FOV_top:FOV_bottom, FOV_left:FOV_right] = self.eject_induct_map[top_poss:bottom_poss,left_poss:right_poss]
            all_h_map[:,FOV_top:FOV_bottom, FOV_left:FOV_right] = self.world.all_h_map[:,top_poss:bottom_poss, left_poss:right_poss]
            util_map[:, FOV_top:FOV_bottom, FOV_left:FOV_right] = sum_util_map[:, top_poss:bottom_poss,
                                                                       left_poss:right_poss]
            agent_map[:,  FOV_top:FOV_bottom, FOV_left:FOV_right] = self.agent_state[top_poss:bottom_poss, left_poss:right_poss]
            first_map[:, FOV_top:FOV_bottom, FOV_left:FOV_right] = all_first_map[top_poss:bottom_poss,left_poss:right_poss]
            first_map-=my_first_map
            worse_map[:,  FOV_top:FOV_bottom, FOV_left:FOV_right] = all_worse_map[top_poss:bottom_poss,left_poss:right_poss]
            worse_map-=my_worse_map
            order_map[:, FOV_top:FOV_bottom, FOV_left:FOV_right] = all_order_map[top_poss:bottom_poss, left_poss:right_poss]
            ag_obs = np.concatenate([obs_map,induct_eject_map,agent_map,first_map,worse_map,all_h_map,util_map,order_map,my_first_map,my_worse_map], axis=0)
            map_obs[:, node, :, :, :] = ag_obs
        map_obs[:,:,3,:,:]/=4
        map_obs[:,:,4,:,:]/=4
        return map_obs,map_vector

    def calculate_info(self):
        perf_dict = {'throughput':0, "wait":0,"congestion":0}
        perf_dict['throughput']= self.finished_task/CopParameters.NUM_WINDOW
        perf_dict["wait"]=sum(sum(self.wait_map))
        perf_dict["congestion"]=np.std(self.wait_map)
        self.finished_task=0
        self.wait_map=np.zeros((self.world_high,self.world_wide))
        return perf_dict

    def calculate_info_episode(self):
        perf_dict = {'throughput':0, "wait":0,"congestion":0}
        perf_dict['throughput']= self.all_finished_task/CopParameters.EPISODE_LEN
        perf_dict["wait"]=sum(sum(self.all_wait_map))
        perf_dict["congestion"]=np.std(self.all_wait_map)
        return perf_dict

    def get_action(self, direction):
        return actionDict[direction]

if __name__ == '__main__':
    env=CO_MAPFEnv(1)
    print(env.obs_range)





