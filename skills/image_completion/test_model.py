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
save_name = ""
if "pong" in ENV.lower():
    save_name = "pong"
elif "breakout" in ENV.lower():
    save_name = "breakout"
else:
    print("Env name error")
    exit()

device = f"cuda:{args.device}" if args.device != "cpu" else "cpu"
if not torch.cuda.is_available() and device != "cpu":
    print("CUDA not available, using CPU")
    device = "cpu"

#data_path = f"../.././data/{ENV}"
data_path = f"../.././data2/{ENV}"
NUM_EPS = len(os.listdir(data_path))
img_sz = 84
batch_size = 32

eps = np.arange(start=1, stop=NUM_EPS + 1)
np.random.shuffle(eps)
val_idxs = eps

dataset = Dataset(data_path, val_idxs, img_sz, square_size=20)
val_load = DataLoader(dataset, batch_size, num_workers=8, shuffle=False)

# Initialize the autoencoder model
model = ImageCompletionModel().to(device)
model.load_state_dict(torch.load("../models/" + save_name + "-image-completion.pt"))

occluded_imgs, imgs = next(iter(val_load))
occluded_imgs = occluded_imgs.to(device)
imgs = imgs.to(device)

with torch.no_grad():
    model.eval()
    output = model(occluded_imgs)
    for out, occ, img in zip(output[:10], occluded_imgs[:10], imgs[:10]):
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12,4))
        axes[0].imshow(occ[0].cpu())
        axes[0].set_title("Occluded Image")
        axes[0].axis('off')

        axes[1].imshow(out[0].cpu())
        axes[1].set_title("Reconstructed Image")
        axes[1].axis('off')

        axes[2].imshow(img[0].cpu())
        axes[2].set_title("Ground Truth Image")
        axes[2].axis('off')

        plt.show()