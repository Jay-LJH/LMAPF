import os
import os.path as osp

import numpy as np
import ray
import setproctitle
import torch
import wandb

from alg_parameters import *
from map_model import MapModel
from runner import RLRunner
from util import set_global_seeds, map_perf, write_to_wandb_map, window_map_perf

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
ray.init(num_gpus=SetupParameters.NUM_GPU)
print("Welcome to LMAPF!\n")


def main():
    """main code"""
    if RecordingParameters.RETRAIN:
        restore_path = 'models/LMAPF/Maze24-10-241708/final'
        map_net_path_checkpoint = restore_path + "/map_net_checkpoint.pkl"
        map_net_dict = torch.load(map_net_path_checkpoint)

    if RecordingParameters.WANDB:
        if RecordingParameters.RETRAIN:
            wandb_id = '587ljdoa'
        else:
            wandb_id = wandb.util.generate_id()
        wandb.init(project="LMAPF")
        print('id is:{}'.format(wandb_id))
        print('Launching wandb...\n')
        wandb.define_metric("map/step")
        wandb.define_metric("map/*", step_metric="map/step")
    setproctitle.setproctitle(
        RecordingParameters.EXPERIMENT_PROJECT + RecordingParameters.EXPERIMENT_NAME + "@" + RecordingParameters.ENTITY)
    set_global_seeds(SetupParameters.SEED)
    if not os.path.exists("./h_maps"):
        os.makedirs("./h_maps")

    global_device = torch.device('cuda') if SetupParameters.USE_GPU_GLOBAL else torch.device('cpu')
    local_device = torch.device('cuda') if SetupParameters.USE_GPU_LOCAL else torch.device('cpu')
    global_map_model = MapModel(0, global_device, True)

    if RecordingParameters.RETRAIN:
        global_map_model.network.load_state_dict(map_net_dict['model'])
        global_map_model.net_optimizer.load_state_dict(map_net_dict['optimizer'])

    envs = [RLRunner.remote(i + 1) for i in range(TrainingParameters.N_ENVS)]

    if RecordingParameters.RETRAIN:
        curr_steps = map_net_dict["step"]
        curr_episodes = map_net_dict["episode"]
        map_update=map_net_dict["map_update"]
    else:
        curr_steps = curr_episodes = map_update=0

    update_done = True
    job_list = []
    last_model_t = -RecordingParameters.SAVE_INTERVAL - 1
    last_print_t = -RecordingParameters.PRINT_INTERVAL - 1

    # start training
    try:
        while curr_steps < TrainingParameters.N_MAX_STEPS:
            if update_done:
                map_update += 1
                for i, env in enumerate(envs):
                    job_list.append(env.map_run.remote())

            # get data from multiple processes
            done_id, job_list = ray.wait(job_list, num_returns=TrainingParameters.N_ENVS)
            update_done = True if job_list == [] else False
            done_len = len(done_id)
            job_results = ray.get(done_id)

            temp_step=int(done_len*CopParameters.NUM_WINDOW)
            curr_steps+=int(done_len*CopParameters.NUM_WINDOW)
            data_buffer = {"obs": [],"vector":[], "returns": [], "values": [], "action": [],"ps": [],"hidden_state":[]}
            perf_dict = map_perf()
            window_perf_dict = window_map_perf()
            for results in range(done_len):
                for i, key in enumerate(data_buffer.keys()):
                    data_buffer[key].append(job_results[results][i])
                curr_episodes += job_results[results][-3]
                for key in window_perf_dict.keys():
                    window_perf_dict[key].append(job_results[results][-2][key])
                if job_results[results][-1] is not None:
                    for key in job_results[results][-1].keys():
                        perf_dict[key].append(np.nanmean(job_results[results][-1][key]))

            for key in data_buffer.keys():
                data_buffer[key] = np.concatenate(data_buffer[key], axis=0)
            for key in window_perf_dict.keys():
                window_perf_dict[key] = np.nanmean(window_perf_dict[key])

            recording = False
            if perf_dict["throughput"] != []:
                recording=True
                for key in perf_dict.keys():
                    perf_dict[key] = np.nanmean(perf_dict[key])

            # training of reinforcement learning
            mb_loss = []
            inds = np.arange(temp_step)
            for _ in range(TrainingParameters.N_EPOCHS):
                np.random.shuffle(inds)
                for start in range(0, temp_step, CopParameters.MINIBATCH_SIZE):
                    end = start + CopParameters.MINIBATCH_SIZE
                    mb_inds = inds[start:end]
                    slices = (arr[mb_inds] for arr in
                              (data_buffer["obs"],data_buffer["vector"],data_buffer["returns"],
                               data_buffer["values"],data_buffer["action"],data_buffer["ps"],data_buffer["hidden_state"]))
                    mb_loss.append(global_map_model.train(*slices))

            data_buffer=None
            if global_device != local_device:
                net_weights = global_map_model.network.to(local_device).state_dict()
                global_map_model.network.to(global_device)
            else:
                net_weights = global_map_model.network.state_dict()
            net_weights_id = ray.put(net_weights)
            weight_job=[]
            for i, env in enumerate(envs):
                weight_job.append(env.set_map_weights.remote(net_weights_id))
            ray.get(weight_job)
            # record training result
            if RecordingParameters.WANDB:
                write_to_wandb_map(map_update, mb_loss,window_perf_dict, perf_dict,recording)

            if (curr_steps - last_print_t) / RecordingParameters.PRINT_INTERVAL >= 1.0:
                last_print_t = curr_steps
                print('episodes: {}, steps: {} \n'.format(
                    curr_episodes, curr_steps))

            # save model
            if (curr_steps - last_model_t) / RecordingParameters.SAVE_INTERVAL >= 1.0:
                last_model_t = curr_steps
                print('Saving Model !\n')
                model_path = osp.join(RecordingParameters.MODEL_PATH, '%.5i' % curr_steps)
                os.makedirs(model_path)
                path_checkpoint = model_path + "/map_net_checkpoint.pkl"
                net_checkpoint = {"model": global_map_model.network.state_dict(),
                                  "optimizer": global_map_model.net_optimizer.state_dict(),
                                  "map_update": map_update,
                                  "step": curr_steps,
                                  "episode": curr_episodes}
                torch.save(net_checkpoint, path_checkpoint)

    except KeyboardInterrupt:
        print("CTRL-C pressed. killing remote workers")
    finally:
        # save final model
        print('Saving Final Model !\n')
        model_path = RecordingParameters.MODEL_PATH + '/final'
        os.makedirs(model_path)
        path_checkpoint = model_path + "/map_net_checkpoint.pkl"
        net_checkpoint = {"model": global_map_model.network.state_dict(),
                          "optimizer": global_map_model.net_optimizer.state_dict(),
                          "map_update": map_update,
                          "step": curr_steps,
                          "episode": curr_episodes}
        torch.save(net_checkpoint, path_checkpoint)
        # killing
        for e in envs:
            ray.kill(e)
        if RecordingParameters.WANDB:
            wandb.finish()


if __name__ == "__main__":
    main()
