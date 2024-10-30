import random
import numpy as np
import torch
import wandb
from alg_parameters import *


def set_global_seeds(i):
    """set seed for fair comparison"""
    torch.manual_seed(i)
    torch.cuda.manual_seed(i)
    torch.cuda.manual_seed_all(i)
    np.random.seed(i)
    random.seed(i)
    torch.backends.cudnn.deterministic = True

def write_to_wandb_map(step, mb_loss=None, window_perf_dict=None,perf_dict=None,recording=False):
    """record performance using wandb"""
    log_dict = {"map/step":step}
    mb_loss = np.nanmean(mb_loss, axis=0)
    for (val, name) in zip(mb_loss, RecordingParameters.MAP_LOSS_NAME):
        if name == 'grad_norm':
            log_dict['map/grad_norm'] = val
        else:
            log_dict['map/'+ name] = val
    log_dict['map/throughput'] =  window_perf_dict['throughput']
    log_dict['map/wait'] =  window_perf_dict['wait']
    log_dict['map/congestion'] =  window_perf_dict['congestion']
    log_dict['map/reward'] =  window_perf_dict['reward']
    if not recording:
        wandb.log(log_dict)
        return
    else:
        log_dict['map/ep_throughput'] = perf_dict['throughput']
        log_dict['map/ep_wait'] = perf_dict['wait']
        log_dict['map/ep_congestion'] = perf_dict['congestion']
        wandb.log(log_dict)
        return


def map_perf():
    perf_dict={'throughput':[], "wait":[],"congestion":[]}
    return perf_dict

def window_map_perf():
    perf_dict={'throughput':[], "wait":[],"congestion":[],"reward":[]}
    return perf_dict

def read_map(file_path):
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
                map_data[i][j] = EnvParameters.obstacle_value
            elif map_data[i][j] == 'e':
                map_data[i][j] = EnvParameters.eject_value
            elif map_data[i][j] == 'i':
                map_data[i][j] = EnvParameters.induct_value
            elif map_data[i][j] == '.':
                map_data[i][j] = EnvParameters.travel_value
    return rows, cols, np.array(map_data,dtype=np.int32)

def read_config(config_path):
        data_dict = {}
        with open(config_path, 'r') as file:
             for line in file:
                line = line.strip()
                if line and not line.startswith('#'):
                    key, value = line.split('=')
                    key = key.strip()
                    value = value.strip()
                    data_dict[key] = convert_if_number(value)

        for key in data_dict:
            setattr(runParameters, key, data_dict[key])
        return data_dict

def convert_if_number(s):
    try:
        return int(s)
    except ValueError:
        return s