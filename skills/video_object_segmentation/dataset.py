import numpy as np
import glob
import os
import random
import torch
from PIL import Image

class Dataset():
    def __init__(self, batch_size, num_frames, env, data_path, H=84, W=84):
        self.batch_size = batch_size
        self.num_frames = num_frames
        self.H = H
        self.W = W
        self.data_path = data_path
        #print('data path is {}'.format(self.data_path))

        self.episode_paths = sorted(glob.glob(self.data_path))
        print('found {} episode paths'.format(len(self.episode_paths)))

        self.episodes = {}
        for episode_path in self.episode_paths:
            self.episodes[episode_path] = sorted(
                glob.glob(os.path.join(episode_path, '*.png')),
                key=lambda x: int(os.path.basename(x).split('.')[0]),
            )


        all_episodes = sorted(self.episodes.keys(), key=lambda s: int(s.split('/')[-1]))
        # shuffle a list
        #all_episodes = random.sample(all_episodes, len(all_episodes))

        # self.train_data_keys = all_episodes[:-1]
        # self.valid_data_keys = [all_episodes[-1]]
        split = int(0.8 * len(all_episodes))
        self.train_data_keys = all_episodes[:split]
        self.valid_data_keys = all_episodes[split:]
        #print(self.train_data_keys, self.valid_data_keys)

    # def get_batch(self, data_type):
    #     if data_type == "train":
    #         episodes_keys = self.train_data_keys
    #     else: #validation
    #         episodes_keys = self.valid_data_keys
    #     envs = {k : self.envs[k] for k in episodes_keys}
    #     # print(envs)
    #     valid_keys = [k for k in envs.keys() if len(envs[k]) >= 2]
    #     # print(valid_keys)
    #     # probs = np.array([len(envs[k]) for k in valid_keys], dtype=np.float32)
    #     # probs /= np.sum(probs)

    #     frames = np.zeros([self.batch_size if data_type == "train" else 2*self.batch_size, self.num_frames, self.H, self.W])
    #     for bs in range(self.batch_size if data_type == "train" else 2*self.batch_size):
    #         k = np.random.choice(valid_keys)
    #         f = envs[k]
    #         f = sorted(f, key=lambda x: (int(x.split("/")[2]), int(x.split("/")[-1].split(".")[0])))
    #         # print(f)
    #         idx = random.randint(0, len(f) - self.num_frames*4)
    #         # print(idx)
    #         for j in range(self.num_frames):
    #             # print(bs, j)
    #             path = f[idx + j*4]
    #             # print(j, path)
    #             x = np.array(Image.open(path)) / 255.
    #             x = np.expand_dims(x, axis=0)
    #             # print(x.shape)
    #             frames[bs, j:j+1, :, :] = x
        
    #     return torch.from_numpy(frames).float()

    def get_batch(self, data_type):
        if data_type == "train":
            episodes_keys = self.train_data_keys
        else: #validation
            episodes_keys = self.valid_data_keys

        episodes = {k : self.episodes[k] for k in episodes_keys}
        valid_keys = [k for k in episodes.keys() if len(episodes[k]) >= 2]
        # probs = np.array([len(episodes[k]) for k in valid_keys], dtype=np.float32)
        # probs /= np.sum(probs)

        frames = np.zeros([self.batch_size, self.num_frames, self.H, self.W])
        for bs in range(self.batch_size):
            k = np.random.choice(valid_keys)
            f = episodes[k]
            idx = random.randint(0, len(f) - self.num_frames)
            for j in range(self.num_frames):
                # print(bs, j)
                path = f[idx + j]
                x = np.array(Image.open(path)) / 255.
                x = np.expand_dims(x, axis=0)
                # print(x.shape)
                frames[bs, j:j+1, :, :] = x
        
        return torch.from_numpy(frames).float()