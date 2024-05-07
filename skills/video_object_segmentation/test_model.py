import numpy as np
np.random.seed(22)
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from model import VideoObjectSegmentationModel
from dataset import Dataset

model_path = ".././models/breakout-vid-obj-seg.pt"
env = "BreakoutNoFrameskip-v4"
batch_size = 64
H = W = 84
num_frames = 2
model = VideoObjectSegmentationModel("cpu")
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

data_path = f"../../data_expert/{env}/*"
dataset = Dataset(batch_size, num_frames, env, data_path)
inp = dataset.get_batch("train")
inp = torch.cat([torch.unsqueeze(inp[:, 0, :, :], 1), torch.unsqueeze(inp[:, 0, :, :], 1)], 1)
x0_ = model(inp)
x0 = torch.unsqueeze(inp[:, 0, :, :], 1)
x1 = torch.unsqueeze(inp[:, 1, :, :], 1)
loss = model.compute_loss(x0, x0_)
# print(loss)
# print(x0)
# print(x0_)

fig, ax = plt.subplots(1, 4, figsize=(12, 4))
idx = 55
ax[0].imshow(x0[idx].permute([1, 2, 0]).detach().numpy(), cmap='gray')
ax[1].imshow(x1[idx].permute([1, 2, 0]).detach().numpy(), cmap='gray')
ax[2].imshow(x0_[idx].permute([1, 2, 0]).detach().numpy(), cmap='gray')
ax[3].imshow((x0[idx]-x1[idx]).permute([1, 2, 0]).detach().numpy(), cmap='gray')

plt.show()

num_f = 5
masks = model.object_masks
fig, ax = plt.subplots(1, num_f, figsize=(25, 15))
for i in range(num_f):
    ax[i].imshow(masks[idx, i, :].detach().numpy(), cmap='gray')

plt.show()

x = torch.zeros(84, 84)
for i in range(model.K):
    x = x + model.object_masks[idx, i, :].detach().numpy()
plt.imshow(x, cmap='gray')
plt.show()