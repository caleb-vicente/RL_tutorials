import numpy as np
from matplotlib import pyplot as plt
import torch
from torch import nn
import torchvision
from tqdm import tqdm

from src import OneHeadAttention
from src import FeatureExtractorConvolutional

# Import datasets
mnist_data = torchvision.datasets.MNIST("MNIST/", train=True, transform=None, target_transform=None, download=True)
mnist_test = torchvision.datasets.MNIST("MNIST/", train=False, transform=None, target_transform=None, download=True)


# Preprocess images
def add_spots(x, m=20, std=5, val=1):  # C
    mask = torch.zeros(x.shape)
    N = int(m + std * np.abs(np.random.randn()))
    ids = np.random.randint(np.prod(x.shape), size=N)
    mask.view(-1)[ids] = val
    return torch.clamp(x + mask, 0, 1)


def prepare_images(xt, maxtrans=6, rot=5, noise=10):  # D
    out = torch.zeros(xt.shape)
    for i in range(xt.shape[0]):
        img = xt[i].unsqueeze(dim=0)
        img = torchvision.transforms.functional.to_pil_image(img)
        rand_rot = np.random.randint(-1 * rot, rot, 1) if rot > 0 else 0
        xtrans, ytrans = np.random.randint(-maxtrans, maxtrans, 2)
        img = torchvision.transforms.functional.affine(img, int(rand_rot[0]), (xtrans, ytrans), 1, 0)
        img = torchvision.transforms.functional.to_tensor(img).squeeze()
        if noise > 0:
            img = add_spots(img, m=noise)
        maxval = img.view(-1).max()
        if maxval > 0:
            img = img.float() / maxval
        else:
            img = img.float()
        out[i] = img
    return out


last_conv_n_layers = 30
img_h = 28
img_w = 28
n_nodes = int(16 ** 2)  # This is because for a FeatureExtractor of 28 the output we are getting is 16.
feature_extractor = FeatureExtractorConvolutional(ch_in=1, conv1_ch=16, conv2_ch=20, conv3_ch=24,
                                                  conv4_ch=last_conv_n_layers)
agent = OneHeadAttention(input_h=img_h, input_w=img_w, n_nodes=n_nodes, input_feature_size=last_conv_n_layers, node_size=36,
                         lin_hid=100, out_dim=10, sp_coord_dim=2, feature_extractor=feature_extractor)

epochs = 1000
batch_size = 300
lr = 1e-3
opt = torch.optim.Adam(params=agent.parameters(), lr=lr)
lossfn = nn.NLLLoss()
for i in tqdm(range(epochs)):
    opt.zero_grad()
    batch_ids = np.random.randint(0, 60000, size=batch_size)  # B
    xt = mnist_data.train_data[batch_ids].detach()
    xt = prepare_images(xt, rot=30).unsqueeze(dim=1)  # C
    yt = mnist_data.train_labels[batch_ids].detach()
    pred = agent(xt)
    pred_labels = torch.argmax(pred, dim=1)  # D
    acc_ = 100.0 * (pred_labels == yt).sum() / batch_size  # E
    correct = torch.zeros(batch_size, 10)
    rows = torch.arange(batch_size).long()
    correct[[rows, yt.detach().long()]] = 1.
    loss = lossfn(pred, yt)
    loss.backward()
    opt.step()


def test_acc(model, batch_size=500):
    acc = 0.
    batch_ids = np.random.randint(0, 10000, size=batch_size)
    xt = mnist_test.test_data[batch_ids].detach()
    xt = prepare_images(xt, maxtrans=6, rot=30, noise=10).unsqueeze(dim=1)
    yt = mnist_test.test_labels[batch_ids].detach()
    preds = model(xt)
    pred_ind = torch.argmax(preds.detach(), dim=1)
    acc = (pred_ind == yt).sum().float() / batch_size
    return acc, xt, yt


acc2, xt2, yt2 = test_acc(agent)
print(acc2)

plt.imshow(agent.att_map[0].max(dim=0)[0].view(16, 16))
