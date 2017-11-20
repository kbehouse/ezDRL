''' STATE Parameter '''
# STATE_FRAMES = 4
# STATE_SHAPE = (84,84)
STATE_FRAMES = 1                # NOTE: "ONE" state use 4 frames
STATE_SHAPE = (7,)

''' ACTION Parameter '''
ACTION_NUM = 4



''' NETWORK Parameter '''
NET_TYPE = "A3C"

MAX_GLOBAL_EP = 2000
MAX_EP_STEP = 300
TRAIN_RUN_STEPS = 5
N_WORKERS = 4 # multiprocessing.cpu_count()
LR_A = 1e-4  # learning rate for actor
LR_C = 2e-4  # learning rate for critic
GAMMA = 0.9  # reward discount
n_model = 1

ENTROPY_BETA = 0.01

N_S = 7              # 7
N_A = 2             # 2
A_BOUND = [-1,1]       # [-1,1]

# Net Parameter
NET_MAIN_SCOPE = 'Main_Net'
NET_TYPE = "A3C"
NET_OUTPUT_GRAPH = True
NET_LOG_DIR = './log'
