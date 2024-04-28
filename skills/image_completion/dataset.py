import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import os
import copy

class Dataset(Dataset):
    def __init__(self, path, idxs, frame_size, square_size=20):
        super().__init__()

        self.path = path
        self.idxs = idxs
        self.frame_size = frame_size
        self.square_size = square_size

    def __len__(self):
        return len(self.idxs)

    def add_black_square(self, img):
        h, w = img.shape
        top = np.random.randint(0, h - self.square_size)
        left = np.random.randint(0, w - self.square_size)
        img[top:top + self.square_size, left:left + self.square_size] = 0.0
        return img

    def __getitem__(self, idx):
        episode_index = np.random.choice(self.idxs)
        num_images = len(os.listdir(self.path + f"/{episode_index}"))

        t = np.random.randint(0, num_images)

        img = np.array(Image.open(f"{self.path}/{episode_index}/{t}.png").resize((self.frame_size, self.frame_size)))
        img = img / 255.0

        occluded_img = self.add_black_square(copy.deepcopy(img))

        img = torch.from_numpy(img).contiguous().float()
        img = img.unsqueeze(0)

        occluded_img = torch.from_numpy(occluded_img).contiguous().float()
        occluded_img = occluded_img.unsqueeze(0)

        return occluded_img, img