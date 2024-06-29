import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

# load saved attention weights in a pickle file and visualize their evolution over TRAINING time

env = "pong" #change with the name of the environment you want to visualize
weights = pickle.load(open(f"../{env}_attention_weights.pkl", "rb"))
weights_8 = []
weights_256 = []
for el in weights:
    el = el.squeeze()
    if el.shape[0] == 8:
        weights_8.append(el)
    else:
        weights_256.append(el)


weights_8 = np.stack(weights_8, axis=0)
weights_256 = np.stack(weights_256, axis=0)

num_timestep = weights_8.shape[0]
num_skills = weights_8.shape[2]


num_envs = weights_8.shape[1]
for i in range(num_envs):
    fig, ax = plt.subplots(1, 4, figsize=(20, 8))
    for j in range(num_skills):
        ax[j].plot(range(num_timestep), weights_8[:, i, j])
        ax[j].set_title(f"Skill {j}")

    plt.savefig(f".././attention_data/pong_env_{i}.png")
    plt.show()
