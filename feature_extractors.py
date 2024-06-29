import sys

sys.path.append('skills')

import torch
import torch as th
import torch.nn as nn
from typing import List
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from autoencoders.model import Autoencoder
import torch.nn.functional as F
from skill_models import Skill
import numpy as np
import math


class FeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box,
                 features_dim: int = 256,
                 skills: List[Skill] = None,
                 device="cpu",
                 mode=0):
        super().__init__(observation_space, features_dim)

        self.skills = skills

        #TODO: change spatial adapters

        # [hardcoded] adapters using 1x1 conv
        self.__vobj_seg_adapter = nn.Sequential(
            nn.Conv2d(20, 16, 1),
            nn.Conv2d(16, 16, 5, 5),
            nn.ReLU(),
            #nn.Sigmoid()
        )
        self.__kpt_enc_adapter = nn.Sequential(
            nn.Conv2d(128, 32, 1),
            nn.Conv2d(32, 32, 6),
            nn.ReLU(),
            #nn.Sigmoid()
        )
        self.__kpt_key_adapter = nn.Sequential(
            nn.Conv2d(4, 16, 1),
            nn.Conv2d(16, 16, 6),
            nn.ReLU()
            #nn.Sigmoid()
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

        self.skills_embeddings = []
        self.skills_name = []

        skill_out = self.preprocess_input(sample, mode=mode)

        self.num_channels = 0
        for el in skill_out:
            if el.ndim == 4:
                self.num_channels += el.shape[1]

    def preprocess_input(self, observations: th.Tensor, mode: int = 0) -> [th.Tensor]:
        """
        :param observations: torch tensor of shape (n_envs, n_stacked_frames, height, width)
        :param mode:    if 0 reshape linear features into spatial
                        if 1 keep the dimension of linear skills
        """
        # print("observation shape", observations.shape)

        skill_out = []
        for skill in self.skills:
            with th.no_grad():
                so = skill.input_adapter(observations)
                so = skill.skill_output(skill.skill_model, so)

            self.skills_embeddings.append(so)

            if mode == 0:
                if skill.name == "state_rep_uns":
                    so = th.reshape(so, (observations.size(0), -1, 16, 16))
                elif skill.name in self.adapters:
                    adapter = self.adapters[skill.name]
                    so = adapter(so)

            elif mode == 1:
                if skill.name in self.adapters:
                    adapter = self.adapters[skill.name]
                    so = adapter(so)

            self.skills_name.append(skill.name)
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
        """
        :param observation_space: Gymnasium observation space
        :param features_dim: Number of features extracted from the observations. This corresponds to the number of units for the last layer.
        :param skills: List of skill objects.
        :param device: Device used for computation.
        """

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
        """
        :param observation_space: Gymnasium observation space
        :param features_dim: Number of features extracted from the observations. This corresponds to the number of units for the last layer.
        :param skills: List of skill objects.
        :param device: Device used for computation.
        :param fixed_dim: Fixed size for the skills embedding vector
        """

        super().__init__(observation_space, features_dim, skills, device)

        self.fixed_dim = fixed_dim

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
        """
        :param observation_space: Gymnasium observation space
        :param features_dim: Number of features extracted from the observations. This corresponds to the number of units for the last layer.
        :param skills: List of skill objects.
        :param device: Device used for computation.
        :param num_conv_layers: Number of convolutional layers to concatenate skills.
        """

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
                 device="cpu",
                 mode=1):
        super().__init__(observation_space, features_dim, skills, device, mode)

        """
        :param observation_space: Gymnasium observation space
        :param features_dim: Number of features extracted from the observations. This corresponds to the number of units for the last layer.
        :param skills: List of skill objects.
        :param device: Device used for computation.
        :param num_linear_skills: Number of linear skills.
        """
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
        skill_out = self.preprocess_input(observations, mode=1)

        # concat the cnn feature maps and pass to a convolutional layer
        spatial_x = th.cat(skill_out[self.num_lin_skills:], 1)
        spatial_x = self.cnn(spatial_x)
        linear_x = th.reshape(spatial_x, (spatial_x.size(0), -1))

        # concat the linear features all together
        return th.cat([*skill_out[:self.num_lin_skills], linear_x], 1)


# ----------------------------------------------------------------------------------

class DotProductAttentionExtractor(FeaturesExtractor):
    def __init__(self, observation_space: spaces.Box,
                 features_dim: int = 256,
                 skills: List[Skill] = None,
                 game: str = "pong",
                 expert: bool = False,
                 device="cpu"):
        """
        :param observation_space: Gymnasium observation space
        :param features_dim: Number of features extracted from the observations. This corresponds to the number of units for the last layer.
        :param skills: List of skill objects.
        :param device: Device used for computation.
        :param game: Name of the game to load the Autoencoder model
        """
        super().__init__(observation_space, features_dim, skills, device)

        self.device = device
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
            seq_layer = nn.Sequential(nn.Linear(skill_out[i].shape[1], features_dim, device=device), nn.ReLU())
            self.mlp_layers.append(seq_layer)

        if expert:
            model_path = "skills/models/" + game.lower() + "-nature-encoder-expert.pt"
        else:
            model_path = "skills/models/" + game.lower() + "-nature-encoder.pt"

        model = Autoencoder().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
        model.eval()
        self.encoder = model.encoder
        x = sample[:, -1:, :, :]
        with torch.no_grad():
            z = self.encoder(x)
            z = th.reshape(z, (z.size(0), -1))
            self.input_size = z.shape[-1]
        self.encoder_seq_layer = nn.Sequential(nn.Linear(self.input_size, features_dim, device=device), nn.ReLU())

    # taken from https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
    def scaled_dot_product_attention(self, query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False,
                                     scale=None, device='cpu') -> torch.Tensor:
        # Efficient implementation equivalent to the following:
        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
        attn_bias = torch.zeros(L, S, dtype=query.dtype, device=device)
        if is_causal:
            assert attn_mask is None
            temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias.to(query.dtype)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias += attn_mask
        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
        return attn_weight @ value

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

        with torch.no_grad():
            # pick only the last frame and return a tensor of shape batch_size x 1 x 84 x 84
            x = observations[:, -1:, :, :]
            encoded_frame = self.encoder(x)
            encoded_frame = th.reshape(encoded_frame, (x.size(0), -1))
        encoded_frame = self.encoder_seq_layer(encoded_frame)  # query
        encoded_frame = encoded_frame.unsqueeze(1)

        # now stack the skill outputs to obtain a sequence of tokens
        stacked_skills = th.stack(skill_out, 0).permute(1, 0, 2)

        att_out = self.scaled_dot_product_attention(query=encoded_frame, key=stacked_skills, value=stacked_skills,
                                                    device=self.device)
        att_out = att_out.squeeze(1)
        return att_out


class WeightSharingAttentionExtractor(FeaturesExtractor):
    def __init__(self, observation_space: spaces.Box,
                 features_dim: int = 256,
                 skills: List[Skill] = None,
                 game: str = "pong",
                 expert: bool = False,
                 device="cpu"):
        """
        :param observation_space: Gymnasium observation space
        :param features_dim: Number of features extracted from the observations. This corresponds to the number of units for the last layer.
        :param skills: List of skill objects.
        :param device: Device used for computation.
        :param game: Name of the game to load the Autoencoder model
        """
        super().__init__(observation_space, features_dim, skills, device)

        self.device = device
        sample = observation_space.sample()  # 4x84x84
        sample = np.expand_dims(sample, axis=0)  # 1x4x84x84
        sample = th.from_numpy(sample) / 255
        sample = sample.to(device)

        #dropout_p = 0.1

        skill_out = self.preprocess_input(sample)

        for i in range(len(skill_out)):
            if len(skill_out[i].shape) > 2:
                skill_out[i] = th.reshape(skill_out[i],
                                          (skill_out[i].size(0), -1))  # flatten skill out to take the dimension

        # ---------- for the skill ---------- #
        self.mlp_layers = nn.ModuleList()
        for i in range(len(skill_out)):
            seq_layer = nn.Sequential(nn.Linear(skill_out[i].shape[1], features_dim, device=device),
                                      nn.ReLU(),
                                      #nn.Sigmoid(),
                                      #nn.Dropout(p=dropout_p)
                                      #nn.BatchNorm1d(features_dim, device=device),
                                      )
            self.mlp_layers.append(seq_layer)

        # ----------for the context ---------- #
        if expert:
            model_path = "skills/models/" + game.lower() + "-nature-encoder-expert.pt"
        else:
            model_path = "skills/models/" + game.lower() + "-nature-encoder.pt"

        model = Autoencoder().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
        model.eval()

        self.encoder = model.encoder
        x = sample[:, -1:, :, :]
        with torch.no_grad():
            z = self.encoder(x)
            z = th.reshape(z, (z.size(0), -1))
            self.input_size = z.shape[-1]

        self.encoder_seq_layer = nn.Sequential(
            nn.Linear(self.input_size, features_dim, device=device),
            nn.ReLU(),
            #nn.Sigmoid(),
            #nn.Dropout(p=dropout_p)
            #nn.BatchNorm1d(features_dim, device=device),
        )

        # ---------- for WSA ---------- #
        self.weights = nn.Sequential(nn.Linear((2 * features_dim), 1, device=device),
                                     nn.ReLU()
                                     )
        #self.dropout = nn.Dropout(p=dropout_p)

        # ---------- saving info ---------- #

        self.att_weights = {}
        self.spatial_adapters = []
        self.linear_adapters = []
        self.training_weights = []


    def forward(self, observations: th.Tensor) -> th.Tensor:
        #print("forward observation shape", observations.shape)
        # -------------- saving stats -------------- #
        self.skills_embeddings = []
        weights = []

        skill_out = self.preprocess_input(observations)

        with torch.no_grad():
            # pick only the last frame and return a tensor of shape batch_size x 1 x 84 x 84
            x = observations[:, -1:, :, :]
            encoded_frame = self.encoder(x)
            encoded_frame = th.reshape(encoded_frame, (x.size(0), -1))
        encoded_frame = self.encoder_seq_layer(encoded_frame)  # query

        self.spatial_adapters = []
        self.linear_adapters = []

        for i in range(len(skill_out)):
            seq_layer = self.mlp_layers[i]
            x = skill_out[i]
            if len(x.shape) > 2:
                x = th.reshape(x, (x.size(0), -1))  # flatten the skill out

            self.spatial_adapters.append(x)
            skill_out[i] = seq_layer(x)  # pass through a mlp layer to reduce and fix the dimension
            self.linear_adapters.append(skill_out[i])

            concatenated = th.cat([encoded_frame, skill_out[i]], 1)

            weight = self.weights(concatenated)
            weights.append(weight)

        weights = th.stack(weights, 1)
        weights = th.softmax(weights, 1)

        self.training_weights.append(weights.detach().cpu().numpy())
        #weights = self.dropout(weights)

        # save attention weights to plot them in evaluation
        for i, s in enumerate(self.skills):
            self.att_weights[s.name] = [w[i] for w in weights]

        # now stack the skill outputs to obtain a sequence of tokens
        stacked_skills = th.stack(skill_out, 0).permute(1, 0, 2)

        # sum product of weights and skills
        att_out = weights * stacked_skills
        att_out = th.sum(att_out, 1)

        return att_out


# ----------------------------------------------------------------------------------
class Reservoir(nn.Module):
    def __init__(self, input_size, reservoir_size, spectral_radius=0.9, max_batch_size=256, device='cpu'):
        """
        :param input_size: Length of the input vector
        :param reservoir_size: Size of the reservoir matrix
        :param spectral_radius: Desired spectral radius for the reservoir matrix.
        :param device: Device used for computation.
        :param max_batch_size: Maximum batch size for the reservoir matrix.
        """

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
        """
        :param observation_space: Gymnasium observation space
        :param features_dim: Number of features extracted from the observations. This corresponds to the number of units for the last layer.
        :param skills: List of skill objects.
        :param device: Device used for computation.
        :param max_batch_size: NMaximum batch size for the reservoir matrix.
        """
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
