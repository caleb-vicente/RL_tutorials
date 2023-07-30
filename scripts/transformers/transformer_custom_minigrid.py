import gym
import torch
from gym_minigrid.minigrid import *
from gym_minigrid.wrappers import FullyObsWrapper, ImgObsWrapper
from skimage.transform import resize
from collections import deque
import matplotlib.pyplot as plt
from tqdm import tqdm

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
n_nodes = int(7**2) # 7 would be the output
n_heads = 3
GWFeatureExtractor = FeatureExtractorMiniGrid(ch_in, conv1_ch, last_conv_n_layers)
GWagent = MultiHeadAttention(input_h=H, input_w=W, n_nodes=n_nodes, input_feature_size=last_conv_n_layers,
                             node_size=node_size, lin_hid=lin_hid, out_dim=out_dim, sp_coord_dim=sp_coord_dim,
                             feature_extractor=GWFeatureExtractor, n_heads=n_heads)
TFeatureExtractor = FeatureExtractorMiniGrid(ch_in, conv1_ch, last_conv_n_layers)
Tnet = MultiHeadAttention(input_h=H, input_w=W, n_nodes=n_nodes, input_feature_size=last_conv_n_layers,
                          node_size=node_size, lin_hid=lin_hid, out_dim=out_dim, sp_coord_dim=sp_coord_dim,
                          feature_extractor=GWFeatureExtractor, n_heads=n_heads)
# --------- End Parameters transformer ---------------

loss_training_array = []
average_episode_lenght = []
episode_length = 0

# TODO: It doesnt look that is training properly, check that is working with the original code of the book
maxsteps = 400  # D
env.max_steps = maxsteps
env.env.max_steps = maxsteps
epochs = 50000
replay_size = 9000
batch_size = 50
lr = 0.0005
gamma = 0.99
replay = deque(maxlen=replay_size)  # E
opt = torch.optim.Adam(params=GWagent.parameters(), lr=lr)
eps = 0.5
update_freq = 100
for i in tqdm(range(epochs)):

    pred = GWagent(state)
    action = int(torch.argmax(pred).detach().numpy())
    if np.random.rand() < eps:  # F
        action = int(torch.randint(0, 5, size=(1,)).squeeze())
    action_d = action_map[action]
    state2, reward, truncated, terminated, info = env.step(action_d)
    done = False
    if truncated or terminated:
        done = True
    reward = -0.01 if reward == 0 else reward  # G
    state2 = prepare_state(state2)
    exp = (state, action, reward, state2, done)

    replay = update_replay(replay, exp, replay_size)

    episode_length += 1
    if done:
        average_episode_lenght.append(episode_length)
        episode_length = 0
        s, _ = env.reset()
        state = prepare_state(s)
    else:
        state = state2

    if len(replay) > batch_size:
        opt.zero_grad()

        state_batch, action_batch, reward_batch, state2_batch, done_batch = get_minibatch(replay, batch_size)

        q_pred = GWagent(state_batch).cpu()
        astar = torch.argmax(q_pred, dim=1)
        qs = Tnet(state2_batch).gather(dim=1, index=astar.unsqueeze(dim=1)).squeeze()

        targets = get_qtarget_ddqn(qs.detach(), reward_batch.detach(), gamma, done_batch)

        loss = lossfn(q_pred, targets.detach(), action_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(GWagent.parameters(), max_norm=1.0)  # H
        opt.step()

        loss_training_array.append(loss.item())
    if i % update_freq == 0:  # I
        Tnet.load_state_dict(GWagent.state_dict())

# Print training metrics ----------------------------
from src import moving_average

log_loss = [math.log(num) for num in loss_training_array]
log_loss_average = moving_average(log_loss, 100)
plt.plot(log_loss_average)
plt.title('Log Loss')
plt.savefig('../../video/log_loss_transformer.png')
# Calculate moving average
try:
    # moving_avg = moving_average(average_episode_lenght, 5)
    plt.plot(average_episode_lenght)
    plt.title('Average episode length')
    plt.savefig('../../video/average_episode_transformer.png')  # TODO: reset previous plot when printing this one
except Exception as e:
    print(e)

# End print training metrics ------------------------

# Save models ----------------------------------------
import os
import datetime
path = "../../checkpoints/"
timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
filename = f"DQNTransformer_agent_{timestamp}.pt"
filepath = path + filename

# Create directory if it doesn't exist
os.makedirs(path, exist_ok=True)

torch.save(GWagent, filepath)
print(f"Checkpoint saved in {filepath}")
# End save models ------------------------------------


# Inference -----------------------------------------
from src import convert_numpy_to_video

s, info = env.reset()
done = False
frames_list = []
inference_counter = 0

while not done:
    frames_list.append(env.render())
    state = prepare_state(s)
    pred = GWagent(state)
    action = int(torch.argmax(pred).detach().numpy())
    if np.random.rand() < eps:
        action = int(torch.randint(0, 5, size=(1,)).squeeze())
    action_d = action_map[action]
    s, reward, truncated, terminated, info = env.step(action_d)

    inference_counter += 1

    if terminated or truncated:
        done = True

convert_numpy_to_video(frames_list, "../../video/", inference_counter) # TODO: It is not printing the number of the frame in the video
# End Inference -------------------------------------


# state_, _ = env.reset()
# state = prepare_state(state_)
# GWagent(state)
# plt.imshow(env.render())
# plt.imshow(state[0].permute(1,2,0).detach().numpy())
# head, node = 2, 26
# plt.imshow(GWagent.att_map[0][head][node].view(7,7))