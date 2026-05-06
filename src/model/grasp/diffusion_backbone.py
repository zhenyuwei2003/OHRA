import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class SinusoidalPosEmb(nn.Module):
    # compute sinusoidal positional embeddings
    def __init__(self, dim: int, theta: int = 10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x: torch.Tensor):
        """
            x: torch.Tensor, shape (B,)
            return emb: torch.Tensor, shape (B, dim)
        """
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :] * self.theta
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class MLP(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_layers_dim,
        output_dim,
        act='mish',
        use_layer_norm=False,
    ):
        super().__init__()
        act = 'leaky_relu' if act is None else act
        act_fn = dict(
            relu=nn.ReLU,
            leaky_relu=nn.LeakyReLU,
            mish=Mish,
            elu=nn.ELU,
            tanh=nn.Tanh,
        )[act]
        # build mlp
        hidden_layers_dim = deepcopy(hidden_layers_dim)
        hidden_layers_dim.insert(0, input_dim)
        self.mlp = nn.Sequential()
        for i in range(1, len(hidden_layers_dim)):
            self.mlp.add_module(f'linear{i - 1}', nn.Linear(hidden_layers_dim[i - 1], hidden_layers_dim[i]))
            # self.mlp.add_module(f'bn{i - 1}', nn.BatchNorm1d(hidden_layers_dim[i]))
            if use_layer_norm:
                self.mlp.add_module(f'ln{i - 1}', nn.LayerNorm(hidden_layers_dim[i]))
            self.mlp.add_module(f'act{i - 1}', act_fn())
        self.mlp.add_module(f'linear{len(hidden_layers_dim) - 1}', nn.Linear(hidden_layers_dim[-1], output_dim))
        # initialize weights
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)

    def forward(self, x):
        return self.mlp(x)


class MLPWrapper(MLP):
    def __init__(self, channels, feature_dim, *args, **kwargs):
        input_dim = channels + feature_dim
        super().__init__(input_dim=input_dim, *args, **kwargs)
        self.embedding = SinusoidalPosEmb(feature_dim)

    def forward(self, x, t, cond):
        t = self.embedding(t)
        return super().forward(torch.cat([x, cond + t], dim=-1))
