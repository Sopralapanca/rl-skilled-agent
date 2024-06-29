# Skilled Reinforcement Learning Agents

This repository contains the code for the Master's thesis "Adaptively Combining Skill Embeddings for Reinforcement Learning Agents" by [Giacomo Carfi'](https://github.com/Sopralapanca).

## Abstract
Reinforcement Learning (RL) aims to learn agent behavioral policies by maximizing the cumulative reward obtained by interacting with the environment.
Typical RL approaches learn an end-to-end mapping from observations to action spaces which define the agent's behavior.
On the other hand, Foundational Models learn rich representations of the world which can be used by agents to accelerate the learning process.
In this thesis, we study how to combine these representations to create an enhanced state representation.
Specifically, we propose a technique called Weight Sharing Attention (WSA) which combines embeddings of different Foundational Models, and we empirically assess its performance against alternative combination approaches.
We tested WSA on different Atari games, and we analyzed the issue of out-of-distribution data and how to mitigate it.
We showed that, without fine-tuning of hyperparameters, WSA obtains comparable performance with state-of-the-art methods achieving faster state representation learning.
This method is effective and could allow life-long learning agents to adapt to different scenarios over time.

![alt text](https://i.ibb.co/RSXwS6S/wsa-main-architecture.png)

[//]: # (TODO: how to run the code)
[//]: # (## Running the code)

[//]: # (The code is written in Python and uses PyTorch as the main library for deep learning.)

[//]: # (To run the code, you need to install the required dependencies by running:)

[//]: # (```bash)

[//]: # (pip install -r requirements.txt)

[//]: # (```)