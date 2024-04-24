import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import os
import matplotlib.pyplot as plt


class Dataset(Dataset):
    def __init__(self, path, idxs):
        super().__init__()
        self.path = path
        self.idxs = idxs

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        num_images = 0
        while num_images < 10:
            episode_index = np.random.choice(self.idxs)
            num_images = len(os.listdir(self.path + f"/{episode_index}"))

        start_index = np.random.randint(4, num_images - 5)
        new_frame = start_index + 5

        frames = []
        for i in range(3, -1, -1):
            img = np.array(Image.open(f"{self.path}/{episode_index}/{start_index - i}.png"))
            img = img / 255.0
            img = torch.from_numpy(img).contiguous().float()
            frames.append(img)

        frames = torch.stack(frames, dim=0)
        new_frame = np.array(Image.open(f"{self.path}/{episode_index}/{new_frame}.png"))
        new_frame = new_frame / 255.0
        new_frame = torch.from_numpy(new_frame).contiguous().float()
        new_frame = new_frame.unsqueeze(0)
        return frames, new_frame
