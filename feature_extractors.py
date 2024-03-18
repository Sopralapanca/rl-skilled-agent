import torch as th
import torch.nn as nn

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
        # provare a modificarli per matchare le dimensioni dell'autoencoder
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

    def preprocess_input(self, observations: th.Tensor) -> [th.Tensor]:
        #print("observation shape", observations.shape)

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

            skill_out.append(so)

        return skill_out
    def get_dimension(self, observations: th.Tensor) -> int:
        out = self.forward(observations)
        return out.shape[1]

# feature size = 16896
class LinearConcatExtractor(FeaturesExtractor):
    def __init__(self, observation_space: spaces.Box,
                 features_dim: int = 256,
                 skills: List[Skill] = None,
                 device="cpu"):
        super().__init__(observation_space, features_dim, skills, device)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # print(observations.shape)
        skill_out = self.preprocess_input(observations)
        for i in range(len(skill_out)):
            # flatten
            skill_out[i] = th.reshape(skill_out[i], (skill_out[i].size(0), -1))

        x = th.cat(skill_out, 1)
        return x

# ----------------------------------------------------------------------------------

# feature size = 8192
class CNNConcatExtractor(FeaturesExtractor):
    def __init__(self, observation_space: spaces.Box,
                 features_dim: int = 256,
                 skills: List[Skill] = None,
                 num_conv_layers=1,
                 device="cpu"):
        super().__init__(observation_space, features_dim, skills, device)

        # 2 for state_rep_uns, 16 for obj_key_enc, 32 for vid_obj_seg, 16 for obj_key_key
        num_channels = 2 + 16 + 32 + 16
        self.cnn = nn.Sequential(
            nn.Conv2d(num_channels, 32, 3, 1, 1),
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

        # se utilizzo l'autoencoder con la nature cnn,
        # ottengo una shape in output di (1x64x7x7)
        # non è possibile concatenarlo con gli altri che hanno shape (1x32x16x16)
        # quindi questo tipo di concatenamento con autoencoder non funziona
        x = th.cat(skill_out, 1)
        x = self.cnn(x)

        x = th.reshape(x, (x.size(0), -1))
        # print("x shape", x.shape)
        return x


# ----------------------------------------------------------------------------------

# feature size = 8704
class CombineExtractor(BaseFeaturesExtractor):
    """ Assumption:
        skills contains `num_linear_skills` linear encoding skills
        followed by higher dimensional skills
    """

    def __init__(self, observation_space: spaces.Box,
                 features_dim: int = 256,
                 skills: List[Skill] = None,
                 num_linear_skills=0,
                 device="cpu"):
        super().__init__(observation_space, features_dim)

        assert len(skills) > num_linear_skills
        self.skills = skills
        self.num_lin_skills = num_linear_skills

        num_channels = 16 + 32 + 16
        self.cnn = nn.Sequential(
            nn.Conv2d(num_channels, 32, 3, 1, 1),
            nn.ReLU(),
        )
        self.__vobj_seg_adapter.to(device)
        self.__kpt_enc_adapter.to(device)
        self.__kpt_key_adapter.to(device)
        self.cnn.to(device)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # print(observations.shape)
        skill_out = []
        for skill in self.skills:
            with th.no_grad():
                so = skill.input_adapter(observations)
                so = skill.skill_output(skill.skill_model, so)

            if skill.name in self.adapters:
                adapter = self.adapters[skill.name]
                so = adapter(so)
            # print(skill.name, so.shape)
            skill_out.append(so)

        # concat the cnn feature maps
        x = th.cat(skill_out[self.num_lin_skills:], 1)
        x = self.cnn(x)
        x = th.reshape(x, (x.size(0), -1))
        # concat the linear features
        return th.cat([*skill_out[:self.num_lin_skills], x], 1)


# ----------------------------------------------------------------------------------

class SelfAttentionExtractor(FeaturesExtractor):
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
            seq_layer = nn.Sequential(nn.Linear(skill_out[i].shape[1], n_features), nn.ReLU())
            self.mlp_layers.append(seq_layer)

        self.attention = nn.MultiheadAttention(embed_dim=n_features, num_heads=self.n_heads,
                                               batch_first=False)  # batch first was True

    def forward(self, observations: th.Tensor) -> th.Tensor:
        #print("observation shape", observations.shape)
        skill_out = self.preprocess_input(observations)

        for i in range(len(skill_out)):
            seq_layer = self.mlp_layers[i]
            x = skill_out[i]
            if len(x.shape) > 2:
                x = th.reshape(x, (x.size(0), -1))  # flatten the skill out
            skill_out[i] = seq_layer(x)  # pass through a mlp layer to reduce and fix the dimension

        transformed_embeddings = th.stack(skill_out,0)  # shape num_skills x batch_size (num envs) x n_features (length of the embeddings)
        #transposed_embeddings = transformed_embeddings.permute(1, 0, 2)  # shape batch_size x num_skills x n_features

        att_out, att_weights = self.attention(transformed_embeddings, transformed_embeddings,transformed_embeddings)

        att_out = att_out.transpose(0, 1)
        # flatten the attention output to obtain (8, 1024)
        combined_embeddings = th.flatten(att_out, start_dim=1)

        return combined_embeddings
