import numpy as np


MAPdirDict = {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1), 4: (-1, 0)}
actionDict = {v: k for k, v in MAPdirDict.items()}


class State(object):  # world property
    def __init__(self,world_high,world_wide,node_poss):
        """initialization"""
        self.world_high=world_high
        self.world_wide=world_wide
        self.heuristic_map = None
        self.all_priority = None
        self.all_h_map = None
        self.node_poss = node_poss

    # convert heuristic map to all_priority and all_h_map
    def convert_all_heuri_map(self,heuristic_map,obstacle_map):
        self.heuristic_map=heuristic_map
        self.all_priority = {}
        self.all_h_map = np.zeros((4,self.world_high,self.world_wide))
        for goal_poss in heuristic_map.keys():
            self.all_priority[goal_poss]={}
            for node, poss in enumerate(self.node_poss):
                self.all_priority[goal_poss][poss]=[[],[]]
                curr_h=heuristic_map[goal_poss][poss[0] * self.world_wide + poss[1]]
                for action in range(1, 5):
                    direction = MAPdirDict[action]
                    dx, dy = direction[0], direction[1]
                    if poss[0] + dx < self.world_high and poss[0] + dx >= 0 and poss[1] + dy < self.world_wide and poss[1] + dy >= 0 and \
                            obstacle_map[poss[0] + dx, poss[1] + dy] != 1:
                        h_value=heuristic_map[goal_poss][(poss[0] + dx) * self.world_wide + poss[1] + dy]
                        # if h_value==curr_h:
                        #     print('same')
                        if h_value==curr_h-1:
                            self.all_priority[goal_poss][poss][0].append((poss[0] + dx, poss[1] + dy))
                            self.all_h_map[action-1,poss[0],poss[1]]+=1
                        elif h_value==curr_h+1:
                            self.all_priority[goal_poss][poss][1].append((poss[0] + dx, poss[1] + dy))
                        else:
                            print('error')
        self.all_h_map/=np.max(self.all_h_map)
        # with open(map_location, 'wb') as f:
        #     np.save(f, heuristic_map)
        #     np.save(f, self.all_priority)
        #     np.save(f, self.all_h_map)
        return







