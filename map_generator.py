import random
import os
import numpy as np
from alg_parameters import *

class map:
    def __init__(self, width=EnvParameters.WORLD_WIDE, height=EnvParameters.WORLD_HIGH,seed=42,path=None):
        self.width = width
        self.height = height
        self.seed = seed
        self.matrix = np.full((height, width), -1, dtype=np.int32)
        if path:
            self.load(path)
        else:
            self.generate_map()
            
    def __str__(self):
        return str(self.matrix)
    
    def print_map(self):
        for row in self.matrix:
            print("".join(["@" if cell == -1 else "." for cell in row]))
    def serialize(self):
        print(self.__class__.__name__)
        path = os.getcwd() + '/maps/' + self.__class__.__name__ + '.txt'
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
    def load(self, path):
        with open(path, 'r') as f:
            self.width, self.height = map(int, f.readline().split())
            self.matrix = []
            for _ in range(self.height):
                line = f.readline().strip()
                row = []
                for cell in line:
                    if cell == "@":
                        row.append(-1)
                    elif cell == "e":
                        row.append(-2)
                    elif cell == "i":
                        row.append(-3)
                    else:
                        row.append(0)
                self.matrix.append(row)
                
    def generate_map(self):
        print("generate_map at base map class, you should implement this method in child class")
        
# create a random maze
# 0: path, 1: wall, -2: eject, -3: induct
# algorithm: dfs, prim
class Maze(map):
    def __init__(self, width=EnvParameters.WORLD_WIDE, height=EnvParameters.WORLD_HIGH,seed=42,obstacle_rate=0.5,function='prim',path = None):
        self.totalpath = width * height * (1-obstacle_rate)
        self.function = function
        super().__init__(width, height,seed,path=path)      
        
    def generate_map(self):
        if self.function == 'dfs':
            self.generate_maze_dfs(1, 1)
        elif self.function == 'prim':
            self.generate_maze_prim(1, 1)
        else:
            raise ValueError("Function not supported")
    
    def generate_maze_dfs(self, x, y):
        directions = [(0, -2), (2, 0), (0, 2), (-2, 0)]
        self.matrix[y][x] = 0
        random.shuffle(directions)
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 < nx < self.width and 0 < ny < self.height and self.matrix[ny][nx] == 1:
                self.matrix[y + dy // 2][x + dx // 2] = 0
                self.generate_maze_dfs(nx, ny)
            
    def generate_maze_prim(self,x,y):
        self.matrix[y][x] = 0
        walls = self.get_neighbors((x,y), -1)
        path_count = 0
        while walls:
            wall = random.choice(walls)
            neighbors = self.get_neighbors(wall, 0)
            if len(neighbors) == 1:
                self.matrix[wall[0]][wall[1]] = 0
                walls += self.get_neighbors(wall, 1)
                path_count += 1
                if path_count >= self.totalpath:
                    break
            walls.remove(wall)
        if path_count < self.totalpath:
            walls = [(x, y) for x in range(1, self.width-1) for y in range(1, self.height-1) if self.matrix[y][x] == -1]
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
            if 0 < neighbor[0] < self.height-1 and 0 < neighbor[1] < self.width-1:
                if self.matrix[neighbor[0]][neighbor[1]] == value:
                    neighbors.append(neighbor)
        return neighbors   

if __name__ == "__main__":
    maze = Maze(EnvParameters.WORLD_HIGH, EnvParameters.WORLD_WIDE, seed=42, obstacle_rate=0.5)
    maze.print_map()
    maze.serialize()
    print(maze.matrix)
        
    