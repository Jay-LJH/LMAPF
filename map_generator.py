import random
import os
import numpy as np
from alg_parameters import *

class map:
    def __init__(self, width=runParameters.WORLD_WIDE, height=runParameters.WORLD_HIGH,seed=42,path=None):
        self.width = width
        self.height = height
        self.seed = seed 
        self.path = path    
        if path:
            self.load(path)
        else:
            self.matrix = np.full((height, width), -1, dtype=np.int32)
            self.generate_map()
            
    def __str__(self):
        return str(self.matrix)
    
    def print_map(self):
        for row in self.matrix:
            print("".join(["@" if cell == -1 else "." for cell in row]))
    def serialize(self):
        if self.path:
            path = self.path
        else:
            path = os.getcwd() + '/maps/' + self.__class__.__name__ +"_"+str(self.width)+"_"+str(self.height) +'.txt'
        with open(path, 'w') as f:
            f.write(f"{self.width} {self.height}\n")
            for row in self.matrix:
                line = ""
                for cell in row:
                    if cell == -1:
                        line += "@"
                    elif cell == -2:
                        line += "e"
                    elif cell == -3:
                        line += "i"
                    else:
                        line += "."
                f.write(line + '\n')
        config_path = os.getcwd() + '/maps/' + self.__class__.__name__ +"_"+str(self.width)+"_"+str(self.height) +'.config'
        with open(config_path, 'w') as f:
            f.write(f"N_NODE = {len(self.nodes)}\n")
            f.write(f"VEC_LEN = {len(self.nodes)}\n")
            f.write(f"N_AGENT = {int(len(self.nodes)*EnvParameters.AGENT_RATE)}\n")
            f.write(f"WORLD_HIGH = {self.height}\n")
            f.write(f"WORLD_WIDE = {self.width}\n")
            f.write(f"MAP_CLASS = {self.__class__.__name__}\n")

    def load(self, path):
        with open(path, 'r') as f:
            self.width, self.height = f.readline().split()
            self.width, self.height = int(self.width), int(self.height)
            print(self.width, self.height)
            self.matrix = np.zeros((self.height, self.width), dtype=np.int32)
            for i in range(self.height):
                line = f.readline()
                for j in range(self.width):
                    if line[j] == "@":
                        self.matrix[i,j]=-1
                    elif line[j] == "e":
                        self.matrix[i,j]=-2
                    elif line[j] == "i":
                        self.matrix[i,j]=-3
                    else:
                        self.matrix[i,j]=0
                
    def generate_map(self):
        print("generate_map at base map class, you should implement this method in child class")
        
# create a random maze
# 0: path, -1: wall, -2: eject, -3: induct
# algorithm: dfs, prim
class Maze(map):
    def __init__(self, width=runParameters.WORLD_WIDE, height=runParameters.WORLD_HIGH,
                 seed=42,obstacle_rate=0.5,function='prim',pad=True,path = None):
        self.totalpath = width * height * (1-obstacle_rate)
        self.function = function
        self.pad = pad
        super().__init__(width, height,seed,path=path)      
        
    def generate_map(self):
        if self.function == 'dfs':
            self.generate_maze_dfs(1, 1)
        elif self.function == 'prim':
            self.generate_maze_prim(1, 1)
        else:
            raise ValueError("Function not supported")
        self.nodes = []
        self.obstacles = []
        for i in range(self.height):
            for j in range(self.width):
                if self.matrix[i,j] == 0:
                    self.nodes.append((i,j))
                elif self.matrix[i,j] == -1:
                    self.obstacles.append((i,j))
    
    def generate_maze_dfs(self, x, y):
        directions = [(0, -2), (2, 0), (0, 2), (-2, 0)]
        self.matrix[y][x] = 0
        random.shuffle(directions)
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 < nx < self.width and 0 < ny < self.height and self.matrix[ny][nx] == 1:
                self.matrix[y + dy // 2][x + dx // 2] = 0
                self.generate_maze_dfs(nx, ny)
            
    def generate_maze_prim(self, x, y):
        self.matrix[y][x] = 0
        walls = self.get_neighbors((x,y), -1)
        path_count = 0
        while walls:
            wall = random.choice(walls)
            neighbors = self.get_neighbors(wall, 0)
            if len(neighbors) == 1:
                self.matrix[wall[0]][wall[1]] = 0
                walls += self.get_neighbors(wall, -1)
                path_count += 1
                if path_count >= self.totalpath:
                    break
            walls.remove(wall)
        if path_count < self.totalpath:
            if self.pad:
                walls = [(x, y) for x in range(1, self.width-1) for y in range(1, self.height-1) if self.matrix[y][x] == -1]
            else:
                walls = [(x, y) for x in range(0, self.width) for y in range(0, self.height) if self.matrix[y][x] == -1]
            while path_count < self.totalpath:
                wall = random.choice(walls)
                self.matrix[wall[1]][wall[0]] = 0
                path_count += 1
                walls.remove(wall)
    
    def create_induct(self, induct_cnt):
        paths = [(x, y) for x in range(1, self.width-1) for y in range(1, self.height-1) if self.matrix[y][x] == 0]
        for _ in range(induct_cnt):
            path = random.choice(paths)
            self.matrix[path[1]][path[0]] = -3
            
    def create_eject(self, eject_cnt):
        paths = [(x, y) for x in range(1, self.width-1) for y in range(1, self.height-1) if self.matrix[y][x] == 0]
        for _ in range(eject_cnt):
            path = random.choice(paths)
            self.matrix[path[1]][path[0]] = -2         
            
    def get_neighbors(self, cell, value):
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        neighbors = []
        for direction in directions:
            neighbor = (cell[0] + direction[0], cell[1] + direction[1])
            if self.pad:
                if 0 < neighbor[0] < self.height-1 and 0 < neighbor[1] < self.width-1:
                    if self.matrix[neighbor[0]][neighbor[1]] == value:
                        neighbors.append(neighbor)
            else:
                if 0 <= neighbor[0] < self.height and 0 <= neighbor[1] < self.width:
                    if self.matrix[neighbor[0]][neighbor[1]] == value:
                        neighbors.append(neighbor)
        return neighbors 

'''
    Proportion_Maze:
    create a maze with a proportion of the original maze
    width: width of the target maze
    height: height of the target maze
    proportion: the proportion of the original maze
    seed: random seed
    obstacle_rate: the rate of obstacles in the original maze
    function: maze generation algorithm (dfs, prim)
    pad: whether to pad the maze. If True, the maze will be padded with walls, otherwise it would be double wall
    path: where to read or save the maze
'''
class Proportion_Maze(Maze):
    def __init__(self, width=runParameters.WORLD_WIDE, height=runParameters.WORLD_HIGH,
                 proportion=2,seed=42,obstacle_rate=0.5,function='prim',pad=False,path = None):
        if pad:
            width -= 2
            height -= 2
        if width % proportion != 0 or height % proportion != 0:
            print("width and height should be divisible by proportion")
        self.proportion = proportion      
        super().__init__(width//proportion, height//proportion,seed,obstacle_rate,function,not pad,path)
        self.pad = pad
        self.scale_map()

    def scale_map(self):
        if self.pad: 
            new_matrix = np.full((self.height*self.proportion+2, self.width*self.proportion+2), -1, dtype=np.int32)
        else:
            new_matrix = np.full((self.height*self.proportion, self.width*self.proportion), -1, dtype=np.int32)
        for i in range(self.height):
            for j in range(self.width):
                if self.matrix[i,j] == 0:
                    if self.pad:
                        new_matrix[i*self.proportion+1:(i+1)*self.proportion+1, j*self.proportion+1:(j+1)*self.proportion+1] = 0
                    else:
                        new_matrix[i*self.proportion:(i+1)*self.proportion, j*self.proportion:(j+1)*self.proportion] = 0
        self.matrix = new_matrix
        self.height, self.width = self.matrix.shape
        self.nodes = []
        self.obstacles = []
        for i in range(self.height):
            for j in range(self.width):
                if self.matrix[i,j] == 0:
                    self.nodes.append((i,j))
                elif self.matrix[i,j] == -1:
                    self.obstacles.append((i,j))
        
# warehouse map
class Warehouse(map):
    def __init__(self, width=runParameters.WORLD_WIDE, height=runParameters.WORLD_HIGH
                 ,obstacle_gap = EnvParameters.GAP,path = None):
        self.obstacle_gap = obstacle_gap
        super().__init__(width, height,seed=None,path=path) 

    # 0: path, 1: wall, -2: eject, -3: induct
    def generate_map(self):
        map_1 = np.zeros((self.height, self.width),dtype=np.int32)
        map_2 = np.zeros((self.height, self.width),dtype=np.int32)
        self.station_map = np.zeros((self.height, self.width),dtype=np.int32)
        for row in range(3, self.height - 3, self.obstacle_gap):
            map_1[row, 1:self.width - 1] = -2
        for col in range(2, self.width - 2, self.obstacle_gap):
            map_2[2:self.height - 2, col] = -2 
        self.matrix = map_1 + map_2
        self.matrix[self.matrix == -4] = -1
        for i in range(self.width):
            if self.matrix[2, i] == -2: # induct
                self.matrix[0, i] = -3
                self.matrix[-1, i] = -3
        
        eject_station_id=10000
        induct_station_id=1000
        for i in range(self.height):
            for j in range(self.width):
                if self.matrix[i,j]==-1: 
                    self.station_map[i,j+1]= eject_station_id
                    self.station_map[i, j - 1] = eject_station_id
                    self.station_map[i-1, j] = eject_station_id
                    self.station_map[i+1, j] = eject_station_id
                    eject_station_id+=1
                if self.matrix[i,j]==-3:
                    self.station_map[i,j]= induct_station_id
                    induct_station_id+=1
                    
        self.matrix[0, 0] = self.matrix[0, -1] = self.matrix[-1, 0] = self.matrix[-1, -1] = -1
        self.obstacle_map = np.zeros((self.height, self.width),dtype=np.int32)
        self.eject_map = np.zeros((self.height, self.width),dtype=np.int32)
        self.induct_map = np.zeros((self.height, self.width),dtype=np.int32)
        self.obstacle_map[self.matrix == -1] = 1
        self.eject_map[self.matrix == -2] = 1
        self.induct_map[self.matrix == -3] = 1
        self.eject_induct_map = self.eject_map + self.induct_map
        
if __name__ == "__main__":
    maze = Maze(25, 25,seed=42, obstacle_rate=0.5,pad=True)
    maze.print_map()
    print(maze.matrix.shape)
    maze.serialize()
    print(len(maze.nodes))
    