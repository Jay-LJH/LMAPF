import datetime

""" Hyperparameters"""


class CopParameters:
    MAP_ACTION=5  # 0 right, 1 down, 2 left, 3 up， 4:wait
    OUTPUT_ACTION=60
    OBS_CHANNEL=17
    NET_SIZE = 128
    NET_VEC= 32
    N_NODE = 314 
    EPISODE_LEN= 5120
    NUM_WINDOW= 32
    FOV=11
    MINIBATCH_SIZE = int(32)
    GAIN=0.01
    UTIL_T=15
    BEST_R=1
    SECOND_R=0
    WORSE_R=-1
    COLL_R=-0.3
    VEC_LEN= N_NODE
    TOP_NUM=3
    TEAM_REWARD=0.25


class EnvParameters:
    N_AGENT= 262
    N_ACTIONS = 5
    EPISODE_LEN = 5120
    WORLD_HIGH = 25
    WORLD_WIDE= 25
    OBSTACLE_RATE = 0.5
    AGENT_RATE = 0.83
    GAP=3
    H=5


class TrainingParameters:
    GAMMA = 0.95  # discount factor
    LAM = 0.95  # For GAE
    lr = 1e-5
    CLIP_RANGE = 0.2
    MAX_GRAD_NORM = 20
    ENTROPY_COEF = 0.01
    VALUE_COEF = 0.5
    POLICY_COEF = 1
    N_EPOCHS = 8
    N_ENVS = 1 # number of processes
    N_MAX_STEPS = 4e7  # maximum number of time steps used in training
    opti_eps=1e-8
    weight_decay=0


class SetupParameters:
    SEED = 42
    USE_GPU_LOCAL = True
    USE_GPU_GLOBAL = True
    NUM_GPU = 1


class RecordingParameters:
    RETRAIN = False
    WANDB = False
    ENTITY = 'JayLiu'  #记得换成自己的wandb account
    TIME = datetime.datetime.now().strftime('%d-%m-%y%H%M')
    EXPERIMENT_PROJECT = 'LMAPF'
    EXPERIMENT_NAME = 'PIBT_MARL'
    EXPERIMENT_NOTE = 'anything you want to note'
    SAVE_INTERVAL = 1e5  # interval of saving model
    PRINT_INTERVAL=1e4
    MODEL_PATH = './models' + '/' + EXPERIMENT_PROJECT + '/' + EXPERIMENT_NAME + TIME
    MAP_LOSS_NAME = ['all_loss', 'policy_loss', 'entropy', 'critic_loss', 'clipfrac',
                 'grad_norm', 'advantage',"prop_policy","prop_en","prop_v"]


dict1 = {key: value for key, value in vars(CopParameters).items() if not key.startswith('__')}
dict2 = {key: value for key, value in vars(EnvParameters).items() if not key.startswith('__')}
dict3 = {key: value for key, value in vars(TrainingParameters).items() if not key.startswith('__')}
dict4 = {key: value for key, value in vars(SetupParameters).items() if not key.startswith('__')}
dict5 = {"RETRAIN":RecordingParameters.RETRAIN,"MODEL_PATH":RecordingParameters.MODEL_PATH}
all_args = {**dict1, **dict2, **dict3, **dict4, **dict5}



