import numpy as np
import os
from tqdm import tqdm
from PIL import Image
import argparse

# Script to segment expert data into black and white frames for background and foreground

parser = argparse.ArgumentParser()
parser.add_argument("--env", help="Name of the environment to use i.e. Pong, Breakoout, etc.",
                    type=str, required=True, choices=["Breakout", "Pong", "CartPole-v1",
                                                      'Ms_Pacman', 'Seaquest', 'Qbert', 'Asteroids',
                                                      'Enduro', 'Space_Invaders', 'Road_Runner', 'Beam_Rider'])

args = parser.parse_args()

ENV_NAME = args.env
ENV_NAME = ENV_NAME + "NoFrameskip-v4"
LOAD_DIR = "../data_expert/" + ENV_NAME

if not os.path.exists(LOAD_DIR):
    raise FileNotFoundError(LOAD_DIR)

episodes = os.listdir(LOAD_DIR)
for i in tqdm(range(1, len(episodes) + 1)):
    ep = str(i)
    SAVE_DIR = "../data_segmented_expert/" + ENV_NAME + "/" + ep + "/"

    # Create a directory data with subdirectory "breakout" using os to store the frames
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    frames = os.listdir(os.path.join(LOAD_DIR, ep))
    for frame in frames:
        im = np.asarray(Image.open(os.path.join(LOAD_DIR, ep, frame)))
        mask = im < 3
        res = np.zeros_like(im)
        res[~mask] = 255

        res = Image.fromarray(res.astype(np.uint8))
        res.save(SAVE_DIR+frame)
