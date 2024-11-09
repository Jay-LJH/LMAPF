import time
import fire

from logdir import LogDir
from gym_maze.envs import MazeEnv
from gym_maze.utils import create_movie

def main(
    map_file,
    path_file,
    path_format="default", # one of ["default", "warehouse"]
    domain="kiva", # one of ["kiva", "default"]
):
    print("Show maze")
    print("map_file: ", map_file)
    print("path_file: ", path_file)
    logdir = LogDir(name="pygame_capture")
    env = MazeEnv(map_file, domain=domain, logdir=logdir)
    env.render()

    if path_format == "default":
        # Read in path file
        f = open(path_file)
        env.render()
        path_table = []
        new_line = f.readline()
        while new_line:
            tmp_traj = []
            items = new_line.split(' ')
            tmp_path = items[-1]
            # print(tmp_path)
            new_line = f.readline()
            steps = tmp_path.split("->")
            for step in steps:
                step = step.strip("()")
                tmp_cordinate = step.split(",")
                if (len(tmp_cordinate) >= 2):
                    x, y = float(tmp_cordinate[0]), float(tmp_cordinate[1])
                    # print(x, y)
                    tmp_traj.append((x, y))
            path_table.append(tmp_traj)
            # else:
            # print("Wrong coordinate: ", tmp_cordinate)
        # print(path_table)

    elif path_format == "warehouse":
        width, _ = env.maze_size
        with open(path_file, "r") as f:
            path_table = []
            lines = f.readlines()
            n_agent = int(lines[0])
            for line in lines[1:]:
                coordinates = line.split(";")[:-1]
                curr_traj = []
                for coordinate in coordinates:
                    tile_loc = float(coordinate.split(",")[0])
                    tile_loc_x = tile_loc // width
                    tile_loc_y = tile_loc % width
                    curr_traj.append((tile_loc_y, tile_loc_x))
                path_table.append(curr_traj)

    num_agents = len(path_table)
    # print(num_agents)
    for step in range(1000):
        tmp_state = []
        for i in range(num_agents):
            if step >= len(path_table[i]):
                tmp_state.append(path_table[i][-1])
            else:
                tmp_state.append(path_table[i][step])

        # print(len(tmp_state))
        env.add_agents(tmp_state)
        # action = env.action_space.sample()
        # # execute the action
        # obv, reward, done, _ = env.step(action)
        # print("reward is: {}".format(reward))
        env.render()
        time.sleep(0.2)


    # Create video
    create_movie(logdir.dir("frames"), logdir.file("capture"))


if __name__ == "__main__":
    fire.Fire(main)