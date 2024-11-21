# Copyright 2024 the authors of NeuRAD and contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch import nn


class ResidualBlock(nn.Module):
    """Abstract Residual Block class."""

    def __init__(self, in_dim: int, dim: int) -> None:
        super().__init__()
        if in_dim != dim:
            self.res_branch = nn.Conv2d(in_dim, dim, kernel_size=1)
        else:
            self.res_branch = nn.Identity()
        self.main_branch = nn.Identity()
        self.final_activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.final_activation(self.res_branch(x) + self.main_branch(x))


class BasicBlock(ResidualBlock):
    """Basic residual block."""

    def __init__(self, in_dim: int, dim: int, kernel_size: int, padding: int, ins_norm: bool = False):
        super().__init__(in_dim, dim)
        self.main_branch = nn.Sequential(
            nn.Conv2d(in_dim, dim, kernel_size=kernel_size, padding=padding),
            nn.InstanceNorm2d(dim) if ins_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=padding),
            nn.InstanceNorm2d(dim) if ins_norm else nn.Identity(),
        )


class BottleneckBlock(ResidualBlock):
    """Simple residual bottleneck block."""

    def __init__(
        self, in_dim: int, dim: int, kernel_size: int, padding: int, channel_multiplier: int = 1, use_bn: bool = False
    ):
        super().__init__(in_dim, dim)
        mid_dim = channel_multiplier * dim
        self.main_branch = nn.Sequential(
            nn.Conv2d(in_dim, mid_dim, kernel_size=1),
            nn.InstanceNorm2d(mid_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_dim, mid_dim, kernel_size=kernel_size, padding=padding),
            nn.InstanceNorm2d(mid_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_dim, dim, kernel_size=1),
            nn.InstanceNorm2d(mid_dim),
        )

class RgbDecoder(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        in_dim = 10
        hidden_dim = 16
        self.class_prefix = "RgbDecoder"+ "#"
        self.decoder =  torch.nn.Sequential(
            torch.nn.Conv2d(in_dim, hidden_dim, kernel_size=1, padding=0),
            torch.nn.ReLU(inplace=True),
            BasicBlock(hidden_dim, hidden_dim, kernel_size=7, padding=3, ins_norm=True),
            BasicBlock(hidden_dim, hidden_dim, kernel_size=7, padding=3, ins_norm=True),
            torch.nn.ConvTranspose2d(
                hidden_dim,
                hidden_dim,
                kernel_size=1,
                stride=1,
            ),
            BasicBlock(hidden_dim, hidden_dim, kernel_size=7, padding=3, ins_norm=True),
            BasicBlock(hidden_dim, hidden_dim, kernel_size=7, padding=3, ins_norm=True),
            torch.nn.Conv2d(hidden_dim, 3, kernel_size=1, padding=0),
            torch.nn.Sigmoid(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # breakpoint()
        return self.decoder(x)
    
    def get_param_groups(self):
        return {
            f"{self.class_prefix}"+"all": self.parameters(),
        }