import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import Dataset
import torch
import os
import numpy as np
from model import FramePredictionModel
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--device", help="integer number of a device to use (0, 1, 2, 3), or cpu",
                    type=str, default="cpu", required=False, choices=["cpu", "0", "1", "2", "3"])
parser.add_argument("--env", help="Name of the environment to use i.e. Pong",
                    type=str, required=True)

args = parser.parse_args()

ENV = args.env
save_name = ENV.split("No")[0].lower()
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

#data_path = f"../.././data/{ENV}"
data_path = f"../.././data_expert/{ENV}"
NUM_EPS = len(os.listdir(data_path))
img_sz = 84
batch_size = 32

eps = np.arange(start=1, stop=NUM_EPS + 1)
np.random.shuffle(eps)
dataset_ts = Dataset(path=data_path, idxs=eps)
loader = DataLoader(dataset_ts, batch_size, num_workers=8)

# Initialize the autoencoder model
model = FramePredictionModel().to(device)
model.load_state_dict(torch.load("../models/breakout-frame-prediction-expert.pt"))

frames, new_frame = next(iter(loader))

frames = frames.to(device)
new_frame = new_frame.to(device)

with torch.no_grad():
    model.eval()
    output = model(frames)
    for r, f, nf in zip(output[:10], frames[:10], new_frame[:10]):
        fig, axes = plt.subplots(nrows=1, ncols=6, figsize=(12,4))

        axes[0].imshow(f[0].cpu())
        axes[0].set_title("Frame 1")

        axes[1].imshow(f[1].cpu())
        axes[1].set_title("Frame 2")

        axes[2].imshow(f[2].cpu())
        axes[2].set_title("Frame 3")

        axes[3].imshow(f[3].cpu())
        axes[3].set_title("Frame 4")

        axes[4].imshow(r[0].cpu())
        axes[4].set_title("Predicted Image")

        axes[5].imshow(nf[0].cpu())
        axes[5].set_title("Groundtruth")

        plt.show()