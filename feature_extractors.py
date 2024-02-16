import torch as th
import torch.nn as nn

from typing import List
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from skill_models import Skill

# feature size = 16896
class LinearConcatExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box,
                 features_dim: int = 256,
                 skills: List[Skill] = None,
                 device = "cpu"):
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
            "obj_key_enc" : self.__kpt_enc_adapter,
            "obj_key_key" : self.__kpt_key_adapter,
            "vid_obj_seg" : self.__vobj_seg_adapter
        }
        self.__vobj_seg_adapter.to(device)
        self.__kpt_enc_adapter.to(device)
        self.__kpt_key_adapter.to(device)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # print(observations.shape)
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

        x = th.cat(skill_out, 1)
        return th.reshape(x, (x.size(0), -1))
    
# ----------------------------------------------------------------------------------

# feature size = 8192
class CNNConcatExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box,
                 features_dim: int = 256,
                 skills: List[Skill] = None,
                 num_conv_layers = 1,
                 device = "cpu"):
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
            "obj_key_enc" : self.__kpt_enc_adapter,
            "obj_key_key" : self.__kpt_key_adapter,
            "vid_obj_seg" : self.__vobj_seg_adapter
        }
        num_channels = 2 + 16 + 32 + 16
        self.cnn = nn.Sequential(
            nn.Conv2d(num_channels, 32, 3, 1, 1),
            nn.ReLU(),
        )
        if num_conv_layers > 1:
            for _ in range(num_conv_layers-1):
                self.cnn.append(nn.Conv2d(32, 32, 3, 1, 1))
                self.cnn.append(nn.ReLU())
        
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
            
            if skill.name == "state_rep_uns":
                so = th.reshape(so, (observations.size(0), -1, 16, 16))
            elif skill.name in self.adapters:
                adapter = self.adapters[skill.name]
                so = adapter(so)
            # print(skill.name, so.shape)
            skill_out.append(so)

        x = th.cat(skill_out, 1)
        x = self.cnn(x)
        return th.reshape(x, (x.size(0), -1))

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
                 num_linear_skills = 0,
                 device = "cpu"):
        super().__init__(observation_space, features_dim)

        assert len(skills) > num_linear_skills
        self.skills = skills
        self.num_lin_skills = num_linear_skills
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
            "obj_key_enc" : self.__kpt_enc_adapter,
            "obj_key_key" : self.__kpt_key_adapter,
            "vid_obj_seg" : self.__vobj_seg_adapter
        }
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

        x = th.cat(skill_out[self.num_lin_skills:], 1)
        x = self.cnn(x)
        x = th.reshape(x, (x.size(0), -1))
        return th.cat([*skill_out[:self.num_lin_skills], x], 1)

# ----------------------------------------------------------------------------------

class AttentionExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super.__init__(observation_space, features_dim)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        pass