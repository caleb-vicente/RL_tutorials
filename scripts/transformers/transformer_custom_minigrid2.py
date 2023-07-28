import gym
import torch
from gym_minigrid.minigrid import *
from gym_minigrid.wrappers import FullyObsWrapper, ImgObsWrapper
from skimage.transform import resize
from collections import deque
import matplotlib.pyplot as plt
from tqdm import tqdm

from src import DQNAgent, DQNTrainer, DQNInference
from src import MultiHeadAttention
from src import FeatureExtractorMiniGrid


def prepare_state(x):
    ns = torch.from_numpy(x).float().permute(2, 0, 1).unsqueeze(dim=0)  #
    maxv = ns.flatten().max()
    ns = ns / maxv
    return ns


def get_minibatch(replay, size):
    batch_ids = np.random.randint(0, len(replay), size)
    batch = [replay[x] for x in batch_ids]
    state_batch = torch.cat([s for (s, a, r, s2, d) in batch], )
    action_batch = torch.Tensor([a for (s, a, r, s2, d) in batch]).long()
    reward_batch = torch.Tensor([r for (s, a, r, s2, d) in batch])
    state2_batch = torch.cat([s2 for (s, a, r, s2, d) in batch], dim=0)
    done_batch = torch.Tensor([d for (s, a, r, s2, d) in batch])
    return state_batch, action_batch, reward_batch, state2_batch, done_batch


def get_qtarget_ddqn(qvals, r, df, done):
    targets = r + (1 - done) * df * qvals
    return targets


def lossfn(pred, targets, actions):  # A
    loss = torch.mean(torch.pow(targets.detach() -
                                pred.gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze(), 2), dim=0)
    return loss


def update_replay(replay, exp, replay_size):
    r = exp[2]
    N = 1
    if r > 0:
        N = 50
    for i in range(N):
        replay.append(exp)
    return replay


action_map = {  # C
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 5,
}

env = ImgObsWrapper(gym.make('MiniGrid-DoorKey-5x5-v0', render_mode='rgb_array'))
s, _ = env.reset()
state = prepare_state(s)

# --------- Parameters transformer ----------------
ch_in = 3
conv1_ch = 16
last_conv_n_layers = 20
H = 28
W = 28
node_size = 64
lin_hid = 100
out_dim = 5
sp_coord_dim = 2
n_nodes = int(7 ** 2)  # 7 would be the output
n_heads = 3
GWFeatureExtractor = FeatureExtractorMiniGrid(ch_in, conv1_ch, last_conv_n_layers)
model = MultiHeadAttention(input_h=H, input_w=W, n_nodes=n_nodes, input_feature_size=last_conv_n_layers,
                           node_size=node_size, lin_hid=lin_hid, out_dim=out_dim, sp_coord_dim=sp_coord_dim,
                           feature_extractor=GWFeatureExtractor, n_heads=n_heads)

maxsteps = 400
env.max_steps = maxsteps
env.env.max_steps = maxsteps
epochs = 50000
agent = DQNAgent(env, model, lr=0.0005, gamma=0.99, epsilon_start=0.5, epsilon_end=0.5, epsilon_decay=1,
                 buffer_size=9000, batch_size=50, update_target_freq=100, flag_target=True, flag_double=True)

agent, all_total_rewards = DQNTrainer(agent, env, epochs)

checkpoint_path = agent.save(path='../../checkpoints')
print(f"Checkpoint path was saved in {checkpoint_path}")


# Inference -----------------------------------------
DQNInference(agent, env, episodes=1, steps=maxsteps, render=False)

