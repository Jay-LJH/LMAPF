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

