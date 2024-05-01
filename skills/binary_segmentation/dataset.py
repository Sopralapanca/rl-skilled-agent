import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import os


class Dataset(Dataset):
    def __init__(self, path, segmented_path, idxs):
        super().__init__()
        self.path = path
        self.segmented_path = segmented_path

        self.idxs = idxs

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        episode_index = np.random.choice(self.idxs)

        num_images = len(os.listdir(self.path + f"/{episode_index}"))
        t = np.random.randint(0, num_images)

        img = np.array(Image.open(f"{self.path}/{episode_index}/{t}.png"))
        img = img / 255.0
        #img = np.mean(img, axis=2)  # make the image grayscale
        img = torch.from_numpy(img).contiguous().float()
        # expand the dimensions of the image
        img = img.unsqueeze(0)

        seg_img = np.array(Image.open(f"{self.segmented_path}/{episode_index}/{t}.png"))
        seg_img = seg_img / 255.0
        # img = np.mean(img, axis=2)  # make the image grayscale
        seg_img = torch.from_numpy(seg_img).contiguous().float()
        # expand the dimensions of the image
        seg_img = seg_img.unsqueeze(0)

        return img, seg_img