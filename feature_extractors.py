import matplotlib.pyplot as plt
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from skill_models import Skill
import numpy as np


class FeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box,
                 features_dim: int = 256,
                 skills: List[Skill] = None,
                 device="cpu"):
        super().__init__(observation_space, features_dim)

        self.skills = skills
        # [hardcoded] adapters using 1x1 conv
        self.__vobj_seg_adapter = nn.Sequential(
            nn.Conv2d(20, 16, 1),
            nn.Conv2d(16, 16, 5, 5),
            nn.ReLU()
        )
        self.__kpt_enc_adapter = nn.Sequential(
            nn.Conv2d(128, 32, 1),
            nn.Conv2d(32, 32, 6),
            nn.ReLU()
        )
        self.__kpt_key_adapter = nn.Sequential(
            nn.Conv2d(4, 16, 1),
            nn.Conv2d(16, 16, 6),
            nn.ReLU()
        )
        self.adapters = {
            "obj_key_enc": self.__kpt_enc_adapter,
            "obj_key_key": self.__kpt_key_adapter,
            "vid_obj_seg": self.__vobj_seg_adapter
        }
        self.__vobj_seg_adapter.to(device)
        self.__kpt_enc_adapter.to(device)
        self.__kpt_key_adapter.to(device)

        sample = observation_space.sample()  # 4x84x84
        sample = np.expand_dims(sample, axis=0)  # 1x4x84x84
        sample = th.from_numpy(sample) / 255
        sample = sample.to(device)

        skill_out = self.preprocess_input(sample)

        self.num_channels = 0
        for el in skill_out:
            if el.ndim == 4:
                self.num_channels += el.shape[1]

    def preprocess_input(self, observations: th.Tensor) -> [th.Tensor]:
        # print("observation shape", observations.shape)

        skill_out = []
        for skill in self.skills:
            with th.no_grad():
                so = skill.input_adapter(observations)
                so = skill.skill_output(skill.skill_model, so)

            if skill.name == "state_rep_uns":
                so = th.reshape(so, (observations.size(0), -1, 16, 16))
            elif skill.name in self.adapters:
                adapter = self.adapters[skill.name]
                so = adapter(so)

            # print(skill.name, so.shape)
            skill_out.append(so)

        return skill_out

    def get_dimension(self, observations: th.Tensor) -> int:
        out = self.forward(observations)
        return out.shape[1]


class LinearConcatExtractor(FeaturesExtractor):
    def __init__(self, observation_space: spaces.Box,
                 features_dim: int = 256,
                 skills: List[Skill] = None,
                 device="cpu"):
        super().__init__(observation_space, features_dim, skills, device)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # print("lin concat observation shape ", observations.shape)
        skill_out = self.preprocess_input(observations)
        for i in range(len(skill_out)):
            skill_out[i] = th.reshape(skill_out[i], (skill_out[i].size(0), -1))  # flatten

        x = th.cat(skill_out, 1)
        return x


class FixedLinearConcatExtractor(FeaturesExtractor):
    def __init__(self, observation_space: spaces.Box,
                 features_dim: int = 256,
                 skills: List[Skill] = None,
                 fixed_dim: int = 256,
                 device="cpu"):
        super().__init__(observation_space, features_dim, skills, device)

        self.fixed_dim = fixed_dim

        sample = observation_space.sample()  # 4x84x84
        sample = np.expand_dims(sample, axis=0)  # 1x4x84x84
        sample = th.from_numpy(sample) / 255
        sample = sample.to(device)

        skill_out = self.preprocess_input(sample)

        for i in range(len(skill_out)):
            if len(skill_out[i].shape) > 2:
                skill_out[i] = th.reshape(skill_out[i],(skill_out[i].size(0), -1))  # flatten skill out to take the dimension

        self.mlp_layers = nn.ModuleList()
        for i in range(len(skill_out)):
            seq_layer = nn.Sequential(nn.Linear(skill_out[i].shape[1], fixed_dim, device=device), nn.ReLU())
            self.mlp_layers.append(seq_layer)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # print("lin concat observation shape ", observations.shape)
        skill_out = self.preprocess_input(observations)
        for i in range(len(skill_out)):
            skill_out[i] = th.reshape(skill_out[i], (skill_out[i].size(0), -1))  # flatten
            # pass through a mlp layer to reduce and fix the dimension
            seq_layer = self.mlp_layers[i]
            skill_out[i] = seq_layer(skill_out[i])

        x = th.cat(skill_out, 1)
        return x


# ----------------------------------------------------------------------------------

class CNNConcatExtractor(FeaturesExtractor):
    def __init__(self, observation_space: spaces.Box,
                 features_dim: int = 256,
                 skills: List[Skill] = None,
                 num_conv_layers=1,
                 device="cpu"):
        super().__init__(observation_space, features_dim, skills, device)

        self.cnn = nn.Sequential(
            nn.Conv2d(self.num_channels, 32, 3, 1, 1),
            nn.ReLU(),
        )
        if num_conv_layers > 1:
            for _ in range(num_conv_layers - 1):
                self.cnn.append(nn.Conv2d(32, 32, 3, 1, 1))
                self.cnn.append(nn.ReLU())

        self.cnn.to(device)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # print("observation shape", observations.shape)
        skill_out = self.preprocess_input(observations)

        x = th.cat(skill_out, 1)
        x = self.cnn(x)

        x = th.reshape(x, (x.size(0), -1))
        # print("x shape", x.shape)
        return x


# ----------------------------------------------------------------------------------

class CombineExtractor(FeaturesExtractor):
    """ Assumption:
        skills contains `num_linear_skills` linear encoding skills
        followed by higher dimensional skills
    """

    def __init__(self, observation_space: spaces.Box,
                 features_dim: int = 256,
                 skills: List[Skill] = None,
                 num_linear_skills=0,
                 device="cpu"):
        super().__init__(observation_space, features_dim, skills, device)

        assert len(skills) > num_linear_skills
        self.skills = skills
        self.num_lin_skills = num_linear_skills

        self.cnn = nn.Sequential(
            nn.Conv2d(self.num_channels, 32, 3, 1, 1),
            nn.ReLU(),
        )

        self.cnn.to(device)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # print("forward shape", observations.shape)
        skill_out = self.preprocess_input(observations)

        # concat the cnn feature maps
        x = th.cat(skill_out[self.num_lin_skills:], 1)
        x = self.cnn(x)
        x = th.reshape(x, (x.size(0), -1))
        # concat the linear features
        return th.cat([*skill_out[:self.num_lin_skills], x], 1)


# ----------------------------------------------------------------------------------

class SkillsAttentionExtractor(FeaturesExtractor):
    def __init__(self, observation_space: spaces.Box,
                 features_dim: int = 256,
                 skills: List[Skill] = None,
                 n_features: int = 512,
                 n_heads: int = 2,
                 device="cpu"):
        super().__init__(observation_space, features_dim, skills, device)

        self.n_heads = n_heads

        sample = observation_space.sample()  # 4x84x84
        sample = np.expand_dims(sample, axis=0)  # 1x4x84x84
        sample = th.from_numpy(sample) / 255
        sample = sample.to(device)

        skill_out = self.preprocess_input(sample)

        for i in range(len(skill_out)):
            if len(skill_out[i].shape) > 2:
                skill_out[i] = th.reshape(skill_out[i],
                                          (skill_out[i].size(0), -1))  # flatten skill out to take the dimension

        self.mlp_layers = nn.ModuleList()
        for i in range(len(skill_out)):
            seq_layer = nn.Sequential(nn.Linear(skill_out[i].shape[1], n_features, device=device), nn.ReLU())
            self.mlp_layers.append(seq_layer)

        self.attention = nn.MultiheadAttention(embed_dim=n_features, num_heads=self.n_heads,
                                               batch_first=False, device=device)  # batch first was True

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # print("forward observation shape", observations.shape)
        skill_out = self.preprocess_input(observations)

        # consider skills as tokens in a sequence
        # so first fix the dimension of each skill output
        for i in range(len(skill_out)):
            seq_layer = self.mlp_layers[i]
            x = skill_out[i]
            if len(x.shape) > 2:
                x = th.reshape(x, (x.size(0), -1))  # flatten the skill out

            skill_out[i] = seq_layer(x)  # pass through a mlp layer to reduce and fix the dimension

        # From the documentation:
        # L: Target sequence length. =>
        #    This refers to the length of the sequences you want to attend to.
        #    In the case of text data it could be the length of a sentence or the number of tokens in a sequence.
        # N: Batch size. => This indicates the number of sequences or samples you're processing simultaneously.
        # E: embedding dimension. =>
        #    This represents the dimensionality of the embeddings. Dimension of a single token in the sequence.

        # In our case:
        # L: length of the sequences => Number of skills
        # N: Batch size => Number of environments
        # E: embedding dimension => Dimension of a single skill embedding (n_features)

        # now stack the skill outputs to obtain a sequence of tokens
        transformed_embeddings = th.stack(skill_out, 0)  # 5x8x512 #prova a far passare tran

        att_out, att_weights = self.attention(transformed_embeddings, transformed_embeddings, transformed_embeddings)
        att_out = att_out.transpose(0, 1)
        # print("att weights shape", att_weights.shape) num_envs (batch size) x num_heads x num_skill

        combined_embeddings = th.flatten(att_out, start_dim=1)
        return combined_embeddings


class ChannelsAttentionExtractor(FeaturesExtractor):
    def __init__(self, observation_space: spaces.Box,
                 features_dim: int = 256,
                 skills: List[Skill] = None,
                 n_heads: int = 2,
                 device="cpu"):
        super().__init__(observation_space, features_dim, skills, device)

        self.n_heads = n_heads
        self.embed_dim = 256  # 16x16
        self.attention = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.n_heads,
                                               batch_first=False, device=device)  # batch first was True

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # print("forward observation shape", observations.shape)
        skill_out = self.preprocess_input(observations)

        # consider channels as tokens in a sequence

        # From the documentation:
        # L: Target sequence length. =>
        #    This refers to the length of the sequences you want to attend to.
        #    In the case of text data it could be the length of a sentence or the number of tokens in a sequence.
        # N: Batch size. => This indicates the number of sequences or samples you're processing simultaneously.
        # E: embedding dimension. =>
        #    This represents the dimensionality of the embeddings. Dimension of a single token in the sequence.

        for i in range(len(skill_out)):
            skill_out[i] = th.flatten(skill_out[i], start_dim=2)

        x = th.cat(skill_out, 1)  # (8x66x256) #batch_size*n_sequences*size_sequences
        # In our case:
        # L: length of the sequences => Number of channels
        # N: Batch size => Number of environments
        # E: embedding dimension => Dimension of a single skill embedding (n_features)

        # transpose as multiheadattention accept #n_sequences*batch_size*size_sequences
        x = x.transpose(0, 1)

        # pass x to mlp to obtain different Q, K, V vectors?

        att_out, att_weights = self.attention(x, x, x)
        att_out = att_out.transpose(0, 1)
        # print("att weights shape", att_weights.shape) num_envs (batch size) x num_heads x num_skill

        combined_embeddings = th.flatten(att_out, start_dim=1)
        return combined_embeddings




from autoencoders.model import Autoencoder
import torch.nn.functional as F
class DotProductAttentionExtractor(FeaturesExtractor):
    def __init__(self, observation_space: spaces.Box,
                 features_dim: int = 256,
                 skills: List[Skill] = None,
                 game: str = "Pong",
                 n_features: int = 1024,
                 device="cpu"):
        super().__init__(observation_space, features_dim, skills, device)
        model_path = "skills/models/" + game.lower() + "-nature-encoder.pt"
        model = Autoencoder().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
        model.eval()
        self.encoder = model.encoder
        self.n_features = n_features
        self.device = device
        sample = observation_space.sample()  # 4x84x84
        sample = np.expand_dims(sample, axis=0)  # 1x4x84x84
        sample = th.from_numpy(sample) / 255
        sample = sample.to(device)

        x = sample[:, -1:, :, :]
        with torch.no_grad():
            z = self.encoder(x)
            z = th.reshape(z, (z.size(0), -1))
            self.input_size = z.shape[-1]

        self.encoder_seq_layer = nn.Sequential(nn.Linear(self.input_size, self.n_features, device=device), nn.ReLU())


        skill_out = self.preprocess_input(sample)
        for i in range(len(skill_out)):
            if len(skill_out[i].shape) > 2:
                skill_out[i] = th.reshape(skill_out[i],
                                          (skill_out[i].size(0), -1))  # flatten skill out to take the dimension

        self.mlp_layers = nn.ModuleList()
        for i in range(len(skill_out)):
            seq_layer = nn.Sequential(nn.Linear(skill_out[i].shape[1], self.n_features, device=device), nn.ReLU())
            self.mlp_layers.append(seq_layer)


    def forward(self, observations: th.Tensor) -> th.Tensor:
        #print("forward observation shape", observations.shape)
        skill_out = self.preprocess_input(observations)

        with torch.no_grad():
            # pick only the last frame and return a tensor of shape batch_size x 1 x 84 x 84
            x = observations[:, -1:, :, :]
            encoded_frame = self.encoder(x)
            encoded_frame = th.reshape(encoded_frame, (x.size(0), -1))  # prima di fare il flatten, provare a farlo passare in qualche livello convoluzionale per ridurre un po i canali?

        encoded_frame = self.encoder_seq_layer(encoded_frame) #query

        combined_embeddings = th.zeros(observations.shape[0], observations.shape[0], self.n_features).to(self.device)

        for i in range(len(skill_out)):
            seq_layer = self.mlp_layers[i]
            x = skill_out[i]
            if len(x.shape) > 2:
                x = th.reshape(x, (x.size(0), -1))  # flatten the skill out
                x = seq_layer(x)  # pass through a mlp layer to reduce and fix the dimension

            relevance = th.matmul(x, encoded_frame.T)
            att_weigths = F.softmax(relevance, dim=1)
            el = att_weigths.unsqueeze(-1) * x
            combined_embeddings = combined_embeddings + el
        combined_embeddings = combined_embeddings.sum(dim=1)

        return combined_embeddings




class MLPAttentionExtractor(FeaturesExtractor):
    def __init__(self, observation_space: spaces.Box,
                 features_dim: int = 256,
                 skills: List[Skill] = None,
                 game: str = "Pong",
                 n_features: int = 1024,
                 device="cpu"):
        super().__init__(observation_space, features_dim, skills, device)
        model_path = "skills/models/" + game.lower() + "-nature-encoder.pt"
        model = Autoencoder().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
        model.eval()
        self.encoder = model.encoder
        self.n_features = n_features
        self.device = device
        sample = observation_space.sample()  # 4x84x84
        sample = np.expand_dims(sample, axis=0)  # 1x4x84x84
        sample = th.from_numpy(sample) / 255
        sample = sample.to(device)

        x = sample[:, -1:, :, :]
        with torch.no_grad():
            z = self.encoder(x)
            z = th.reshape(z, (z.size(0), -1))
            self.input_size = z.shape[-1]

        self.encoder_seq_layer = nn.Sequential(nn.Linear(self.input_size, self.n_features, device=device), nn.ReLU())


        skill_out = self.preprocess_input(sample)
        for i in range(len(skill_out)):
            if len(skill_out[i].shape) > 2:
                skill_out[i] = th.reshape(skill_out[i],
                                          (skill_out[i].size(0), -1))  # flatten skill out to take the dimension

        self.mlp_layers = nn.ModuleList()
        for i in range(len(skill_out)):
            seq_layer = nn.Sequential(nn.Linear(skill_out[i].shape[1], self.n_features, device=device), nn.ReLU())
            self.mlp_layers.append(seq_layer)

        self.relevance_mlp_layers = nn.ModuleList()
        for i in range(len(skill_out)):
            seq_layer = nn.Sequential(nn.Linear(2*n_features, 1, device=device), nn.ReLU())
            self.relevance_mlp_layers.append(seq_layer)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        #print("forward observation shape", observations.shape)
        skill_out = self.preprocess_input(observations)

        with torch.no_grad():
            # pick only the last frame and return a tensor of shape batch_size x 1 x 84 x 84
            x = observations[:, -1:, :, :]
            encoded_frame = self.encoder(x)
            encoded_frame = th.reshape(encoded_frame, (x.size(0), -1))  # prima di fare il flatten, provare a farlo passare in qualche livello convoluzionale per ridurre un po i canali?

        encoded_frame = self.encoder_seq_layer(encoded_frame) #query
        relevance_t = []

        for i in range(len(skill_out)):
            seq_layer = self.mlp_layers[i]
            x = skill_out[i]
            if len(x.shape) > 2:
                x = th.reshape(x, (x.size(0), -1))  # flatten the skill out

            x = seq_layer(x)  # pass through a mlp layer to reduce and fix the dimension
            skill_out[i] = x

            c = th.cat((x, encoded_frame), dim=1)

            relevance_mlp_layer = self.relevance_mlp_layers[i]
            relevance = relevance_mlp_layer(c)

            relevance_t.append(relevance)

        relevance_t = th.cat(relevance_t, dim=1)
        att_weigths = F.softmax(relevance_t, dim=1)

        s = th.stack(skill_out, dim=1)

        # Reshape weights to (8, nskills, 1) to perform broadcasting
        att_weigths = att_weigths.unsqueeze(2)
        # Perform element-wise multiplication of weights with skills
        weighted_skills = att_weigths * s

        # Sum along the second dimension to get the weighted summation
        weighted_sum = torch.sum(weighted_skills, dim=1)

        return weighted_sum


# ----------------------------------------------------------------------------------
class Reservoir(nn.Module):
    def __init__(self, input_size, reservoir_size, spectral_radius=0.9, max_batch_size=256, device='cpu'):
        super(Reservoir, self).__init__()

        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.spectral_radius = spectral_radius

        # Initialize reservoir weights
        self.W_in = th.randn(input_size, reservoir_size)
        self.W_res = th.randn(reservoir_size, reservoir_size)

        # Scale the spectral radius of W_res
        eigenvalues, eigenvectors = th.linalg.eig(self.W_res)
        max_eigenvalue = th.max(th.abs(eigenvalues))
        self.W_res = self.W_res / max_eigenvalue * spectral_radius

        # Initialize reservoir state
        self.reservoir_state = th.zeros(max_batch_size, reservoir_size)  # first was (1, reservoir_size)
        self.reservoir_state = self.reservoir_state.to(device)
        self.W_in = self.W_in.to(device)
        self.W_res = self.W_res.to(device)

    def forward(self, input_data):
        # Input transformation
        input_projection = th.mm(input_data, self.W_in)
        dim = input_projection.shape[0]
        state_projection = th.mm(self.reservoir_state[:dim, :], self.W_res)

        # Reservoir dynamics
        self.reservoir_state[:dim, :] = th.tanh(input_projection + state_projection[:dim, :])

        return self.reservoir_state[:dim, :]


class ReservoirConcatExtractor(FeaturesExtractor):
    def __init__(self, observation_space: spaces.Box,
                 features_dim: int = 256,
                 skills: List[Skill] = None,
                 input_features_dim: int = 512,
                 max_batch_size: int = 256,
                 device="cpu"):
        super().__init__(observation_space, features_dim, skills, device)

        self.reservoir = Reservoir(input_size=input_features_dim, reservoir_size=features_dim, device=device,
                                   max_batch_size=max_batch_size)
        self.reservoir.to(device)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        skill_out = self.preprocess_input(observations)
        for i in range(len(skill_out)):
            # flatten
            skill_out[i] = th.reshape(skill_out[i], (skill_out[i].size(0), -1))

        x = th.cat(skill_out, 1)
        with torch.no_grad():
            x = self.reservoir(x)

        return x
