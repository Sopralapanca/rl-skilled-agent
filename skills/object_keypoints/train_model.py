import random
import numpy as np
import wandb
import os
import sys
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from model import Encoder, KeyNet, RefineNet, Transporter
from dataset import Dataset, Sampler

if len(sys.argv) < 4:
    print(f"Usage: python {sys.argv[0]} <gpu-device> <seed> <env>")
    exit()
gpu = sys.argv[1]
seed = int(sys.argv[2])
env = sys.argv[3]

project = "attskills"

device = torch.device(gpu if torch.cuda.is_available() else "cpu")

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

IMG_SZ = 84
#data_path = f"../../data/{env}"
data_path = f"../../data2/{env}"

NUM_TR_ENVS = 10
NUM_EPS = 1000
MAX_ITER = 1e6
batch_size = 64
image_channels = 1
K = 4
lr = 1e-3
lr_decay = 0.95
lr_deacy_len = 1e5

wandb.init(project=project, tags=["obj_key"])
wandb.config.update({
        "num_eps": NUM_EPS,
        "image-channels": image_channels,
        "image-size": IMG_SZ,
        "steps": MAX_ITER,
        "batch-size": batch_size,
        "K": K,
        "lr": lr,
        "lr-decay": lr_decay,
        "lr-decay-len": lr_deacy_len
    })

encoder = Encoder(image_channels)
key_net = KeyNet(image_channels, K)
refine_net = RefineNet(image_channels)
transporter = Transporter(encoder, key_net, refine_net)
transporter.to(device)
transporter.train()

t_dataset = Dataset(data_path, NUM_EPS, transforms.ToTensor())
t_sampler = Sampler(t_dataset)
t_data_loader = DataLoader(t_dataset, batch_size, sampler=t_sampler)

v_dataset = Dataset(data_path, NUM_EPS, transforms.ToTensor())
v_sampler = Sampler(v_dataset)
v_data_loader = DataLoader(v_dataset, 2*batch_size, sampler=v_sampler)

optimizer = torch.optim.Adam(transporter.parameters(), lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_deacy_len, lr_decay)

best_loss = 100
for i, (xt, xtp1) in enumerate(t_data_loader):
    if i > MAX_ITER:
        break
    transporter.train()
    xt = xt.to(device)
    xtp1 = xtp1.to(device)
    optimizer.zero_grad()
    reconstruction = transporter(xt, xtp1)
    loss = F.mse_loss(reconstruction, xtp1)
    loss.backward()
    optimizer.step()
    scheduler.step()

    with torch.no_grad():
        transporter.eval()
        for j, (xv, xv1) in enumerate(v_data_loader):
            xv = xv.to(device)
            xv1 = xv1.to(device)
            r = transporter(xv, xv1)
            eval_loss = F.mse_loss(r, xv1)
            break

    if eval_loss < best_loss:
        best_loss = eval_loss
        torch.save(transporter.state_dict(), 'breakout-obj-key.pt')
        #torch.save(transporter.state_dict(), os.path.join(wandb.run.dir, 'model.pt'))

    last_lr = scheduler.get_last_lr()[0]
    wandb.log({
            "loss": loss,
            "eval_loss": eval_loss,
            "lr": last_lr
        }, step=i)
    
    # wandb.watch(transporter, F.mse_loss)

    # print(f"[{i}] - Loss: {loss} , Eval Loss: {eval_loss}")
#torch.save(transporter.state_dict(), os.path.join(wandb.run.dir, 'model_final.pt'))