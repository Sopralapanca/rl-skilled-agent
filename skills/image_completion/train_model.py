import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import Dataset
import torch
import os
import numpy as np
from model import ImageCompletionModel
import argparse

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
batch_size = 32

eps = np.arange(start=1, stop=NUM_EPS + 1)
np.random.shuffle(eps)
split_idx = int(NUM_EPS * 0.8)
train_idxs = eps[:split_idx]
val_idxs = eps[split_idx:NUM_EPS]

dataset_ts = Dataset(data_path, train_idxs, img_sz, square_size=20)
train_load = DataLoader(dataset_ts, batch_size, shuffle=True, num_workers=8)

dataset_vs = Dataset(data_path, val_idxs, img_sz, square_size=20)
val_load = DataLoader(dataset_vs, batch_size, num_workers=8)

# Initialize the autoencoder model
model = ImageCompletionModel().to(device)
# pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print(f"Total params: {pytorch_total_params}") #forse un po troppi parametri


# Define loss function and optimizer
criterion = torch.nn.MSELoss().cuda()
optimizer = optim.Adam(model.parameters(), lr=0.001)

best_loss = 1000

# Training loop
num_epochs = 1200 * len(train_load)
for epoch in range(num_epochs):
    model.train()
    train_losses = []

    for i, (occluded_imgs, imgs) in enumerate(train_load):
        optimizer.zero_grad()
        occluded_imgs = occluded_imgs.to(device)
        imgs = imgs.to(device)

        # Forward pass
        outputs = model(occluded_imgs)

        # Compute loss and optimize
        loss = criterion(outputs, imgs)
        train_losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_train_loss = sum(train_losses) / len(train_losses)

    val_losses = []
    with torch.no_grad():
        model.eval()
        for i, (occluded_imgs, imgs) in enumerate(val_load):
            imgs = imgs.to(device)
            occluded_imgs = occluded_imgs.to(device)
            out = model(occluded_imgs)
            loss = criterion(out, imgs)
            val_losses.append(loss.item())

    avg_val_loss = sum(val_losses) / len(val_losses)

    if avg_val_loss < best_loss:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.7f}, Val Loss: {avg_val_loss:.7f}")
        best_loss = avg_val_loss
        #torch.save(model.state_dict(), os.path.join(SAVE_MODELS_DIR, save_name + '-image-completion.pt'))
        torch.save(model.state_dict(), save_name + '-image-completion.pt')
