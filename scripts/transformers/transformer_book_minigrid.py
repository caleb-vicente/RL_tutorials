import torch
import torch.nn as nn
from gym_minigrid.minigrid import *
from gym_minigrid.wrappers import FullyObsWrapper, ImgObsWrapper
from skimage.transform import resize
from einops import rearrange
from collections import deque
from tqdm import tqdm

class MultiHeadRelationalModule(torch.nn.Module):
    def __init__(self):
        super(MultiHeadRelationalModule, self).__init__()
        self.conv1_ch = 16
        self.conv2_ch = 20
        self.conv3_ch = 24
        self.conv4_ch = 30
        self.H = 28
        self.W = 28
        self.node_size = 64
        self.lin_hid = 100
        self.out_dim = 5
        self.ch_in = 3
        self.sp_coord_dim = 2
        self.N = int(7 ** 2)
        self.n_heads = 3

        self.conv1 = nn.Conv2d(self.ch_in, self.conv1_ch, kernel_size=(1, 1), padding=0)  # A
        self.conv2 = nn.Conv2d(self.conv1_ch, self.conv2_ch, kernel_size=(1, 1), padding=0)
        self.proj_shape = (self.conv2_ch + self.sp_coord_dim, self.n_heads * self.node_size)
        self.k_proj = nn.Linear(*self.proj_shape)
        self.q_proj = nn.Linear(*self.proj_shape)
        self.v_proj = nn.Linear(*self.proj_shape)

        self.k_lin = nn.Linear(self.node_size, self.N)  # B
        self.q_lin = nn.Linear(self.node_size, self.N)
        self.a_lin = nn.Linear(self.N, self.N)

        self.node_shape = (self.n_heads, self.N, self.node_size)
        self.k_norm = nn.LayerNorm(self.node_shape, elementwise_affine=True)
        self.q_norm = nn.LayerNorm(self.node_shape, elementwise_affine=True)
        self.v_norm = nn.LayerNorm(self.node_shape, elementwise_affine=True)

        self.linear1 = nn.Linear(self.n_heads * self.node_size, self.node_size)
        self.norm1 = nn.LayerNorm([self.N, self.node_size], elementwise_affine=False)
        self.linear2 = nn.Linear(self.node_size, self.out_dim)

    def forward(self, x):
        N, Cin, H, W = x.shape
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        with torch.no_grad():
            self.conv_map = x.clone()  # C
        _, _, cH, cW = x.shape
        xcoords = torch.arange(cW).repeat(cH, 1).float() / cW
        ycoords = torch.arange(cH).repeat(cW, 1).transpose(1, 0).float() / cH
        spatial_coords = torch.stack([xcoords, ycoords], dim=0)
        spatial_coords = spatial_coords.unsqueeze(dim=0)
        spatial_coords = spatial_coords.repeat(N, 1, 1, 1)
        x = torch.cat([x, spatial_coords], dim=1)
        x = x.permute(0, 2, 3, 1)
        x = x.flatten(1, 2)

        K = rearrange(self.k_proj(x), "b n (head d) -> b head n d", head=self.n_heads)
        K = self.k_norm(K)

        Q = rearrange(self.q_proj(x), "b n (head d) -> b head n d", head=self.n_heads)
        Q = self.q_norm(Q)

        V = rearrange(self.v_proj(x), "b n (head d) -> b head n d", head=self.n_heads)
        V = self.v_norm(V)
        A = torch.nn.functional.elu(self.q_lin(Q) + self.k_lin(K))  # D
        A = self.a_lin(A)
        A = torch.nn.functional.softmax(A, dim=3)
        with torch.no_grad():
            self.att_map = A.clone()  # E
        E = torch.einsum('bhfc,bhcd->bhfd', A, V)  # F
        E = rearrange(E, 'b head n d -> b n (head d)')
        E = self.linear1(E)
        E = torch.relu(E)
        E = self.norm1(E)
        E = E.max(dim=1)[0]
        y = self.linear2(E)
        y = torch.nn.functional.elu(y)
        return y


def prepare_state(x): #A
    ns = torch.from_numpy(x).float().permute(2,0,1).unsqueeze(dim=0)#
    maxv = ns.flatten().max()
    ns = ns / maxv
    return ns

def get_minibatch(replay,size): #B
    batch_ids = np.random.randint(0,len(replay),size)
    batch = [replay[x] for x in batch_ids] #list of tuples
    state_batch = torch.cat([s for (s,a,r,s2,d) in batch],)
    action_batch = torch.Tensor([a for (s,a,r,s2,d) in batch]).long()
    reward_batch = torch.Tensor([r for (s,a,r,s2,d) in batch])
    state2_batch = torch.cat([s2 for (s,a,r,s2,d) in batch],dim=0)
    done_batch = torch.Tensor([d for (s,a,r,s2,d) in batch])
    return state_batch,action_batch,reward_batch,state2_batch, done_batch

def get_qtarget_ddqn(qvals,r,df,done): #C
    targets = r + (1-done) * df * qvals
    return targets


def lossfn(pred, targets, actions):  # A
    loss = torch.mean(torch.pow(targets.detach() - pred.gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze(), 2), dim=0)
    return loss


def update_replay(replay, exp, replay_size):  # B
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

env = ImgObsWrapper(gym.make('MiniGrid-DoorKey-5x5-v0', render_mode='rgb_array'))  # A
s, info = env.reset()
state = prepare_state(s)
GWagent = MultiHeadRelationalModule()  # B
Tnet = MultiHeadRelationalModule()  # C
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
    done=False
    state2, reward, truncated, terminated, info = env.step(action_d)
    if truncated or terminated:
        done=True
    reward = -0.01 if reward == 0 else reward  # G
    state2 = prepare_state(state2)
    exp = (state, action, reward, state2, done)

    replay = update_replay(replay, exp, replay_size)
    if done:
        s, info = env.reset()
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
    if i % update_freq == 0:  # I
        Tnet.load_state_dict(GWagent.state_dict())

# ------ Save checkpoint -------------
import datetime
import os
path = '../../video'
timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
filename = f"dqn_agent_{timestamp}.pt"
filepath = path + filename

# Create directory if it doesn't exist
os.makedirs(path, exist_ok=True)

torch.save(GWagent.state_dict(), filepath)
print(f"Checkpoint saved in {filepath}")

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

convert_numpy_to_video(frames_list, "../../video/")