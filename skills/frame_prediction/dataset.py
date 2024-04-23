import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler
import os
import copy
import matplotlib.pyplot as plt

class Dataset(Dataset):
    def __init__(self, path, idxs):
        super().__init__()
        self.path = path
        self.idxs = idxs

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        episode_index, start_index, new_frame = idx
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


class Sampler(Sampler):
    def __init__(self, dataset):
        self.dataset = dataset

    def __iter__(self):
        while True:
            episode_index = np.random.choice(self.dataset.idxs)
            num_images = len(os.listdir(self.dataset.path + f"/{episode_index}"))

            if num_images < 10:
                pass

            # try to reconstruct the frame 5 steps ahead
            start_index = np.random.randint(4, num_images - 5)
            new_frame = start_index + 5

            yield episode_index, start_index, new_frame

    def __len__(self):
        return len(self.dataset.idxs)