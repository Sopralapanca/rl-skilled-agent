import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import Dataset, Sampler
import torch
import os
import numpy as np
from model import FramePredictionModel
import argparse
from torchvision import transforms

parser = argparse.ArgumentParser()
parser.add_argument("--device", help="integer number of a device to use (0, 1, 2, 3), or cpu",
                    type=str, default="cpu", required=False, choices=["cpu", "0", "1", "2", "3"])
parser.add_argument("--env", help="Name of the environment to use i.e. Pong",
                    type=str, required=True)

args = parser.parse_args()

ENV = args.env
save_name = ENV.split("No")[0].lower()
# print(save_name)
# exit()
# save_name = ""
# if "pong" in ENV.lower():
#     save_name = "pong"
# elif "breakout" in ENV.lower():
#     save_name = "breakout"
# else:
#     print("Env name error")
#     exit()

device = f"cuda:{args.device}" if args.device != "cpu" else "cpu"
if not torch.cuda.is_available() and device != "cpu":
    print("CUDA not available, using CPU")
    device = "cpu"

SAVE_MODELS_DIR = ".././models"
if not os.path.exists(SAVE_MODELS_DIR):
    os.makedirs(SAVE_MODELS_DIR)

#data_path = f"../.././data/{ENV}"
data_path = f"../.././data2/{ENV}"
NUM_EPS = len(os.listdir(data_path))
img_sz = 84
batch_size = 128

eps = np.arange(start=1, stop=NUM_EPS + 1)
np.random.shuffle(eps)
split_idx = int(NUM_EPS * 0.8)
train_idxs = eps[:split_idx]
val_idxs = eps[split_idx:NUM_EPS]

dataset_ts = Dataset(path=data_path, idxs=train_idxs)
t_sampler = Sampler(dataset_ts)
train_load = DataLoader(dataset_ts, batch_size, num_workers=8, sampler=t_sampler)

dataset_vs = Dataset(path=data_path, idxs=val_idxs)
v_sampler = Sampler(dataset_vs)
val_load = DataLoader(dataset_ts, batch_size, num_workers=8, sampler=v_sampler)

# Initialize the autoencoder model
model = FramePredictionModel(input_channels=4).to(device)
# pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print(f"Total params: {pytorch_total_params}") #forse un po troppi parametri


# Define loss function and optimizer
criterion = torch.nn.MSELoss().cuda()
optimizer = optim.Adam(model.parameters(), lr=0.001)

best_loss = 1000

# Training loop
steps = 1000 * len(train_load)
for step in range(steps):
    model.train()
    frames, new_frame = next(iter(train_load))

    optimizer.zero_grad()
    frames = frames.to(device)
    new_frame = new_frame.to(device)

    # Forward pass
    outputs = model(frames)

    # Compute loss and optimize
    train_loss = criterion(outputs, new_frame)
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    with torch.no_grad():
        model.eval()
        frames, new_frame = next(iter(val_load))
        new_frame = new_frame.to(device)
        frames = frames.to(device)
        out = model(frames)
        val_loss = criterion(out, new_frame)


    if val_loss < best_loss:
        print(f"Epoch [{step + 1}/{steps}], Train Loss: {train_loss:.7f}, Val Loss: {val_loss:.7f}")
        best_loss = val_loss
        #torch.save(model.state_dict(), os.path.join(SAVE_MODELS_DIR, save_name + '-frame-prediction.pt'))
        torch.save(model.state_dict(), f'./{save_name}-frame-prediction.pt')
