import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import os


class Dataset(Dataset):
    def __init__(self, path, idxs, frame_size):
        super().__init__()

        self.path = path
        self.idxs = idxs
        self.frame_size = frame_size

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        n = idx + 1  # idx starts from 0, but episode folders start from 1

        ep_len = len([name for name in os.listdir(f"{self.path}/{n}/") if
                      os.path.isfile(os.path.join(f"{self.path}/{n}/", name))])

        t = np.random.randint(0, ep_len)

        img = np.array(Image.open(f"{self.path}/{n}/{t}.png").resize((self.frame_size, self.frame_size)))
        img = img / 255.0
        #img = np.mean(img, axis=2)  # make the image grayscale
        img = torch.from_numpy(img).contiguous().float()
        # expand the dimensions of the image
        img = img.unsqueeze(0)
        return img