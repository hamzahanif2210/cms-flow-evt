import torch
import torch.nn as nn
from modules import DenseNetwork, TimestepEmbedder
from typing import Union


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: Union[float, torch.Tensor] = 1e-5,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class BasicBlock(nn.Module):
    def __init__(self, in_dim, out_dim, stride=1):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
            LayerScale(out_dim),
        )
        self.norm = nn.LayerNorm(out_dim, elementwise_affine=False)

    def forward(self, x):
        x = self.norm(x)
        x = x + self.ffn(x)
        return x


class FlowNumPFNet(nn.Module):
    def __init__(self, config, noisy_dim=1):
        super().__init__()
        self.config = config

        act = config["act"]
        self.hidden_dim = config["hidden_dim"]
        self.act = act
        self.noisy_dim = noisy_dim
        self.n_res_blocks = config["n_res_blocks"]

        self.truth_variables = self.config["truth_variables"]

        self.truth_in_dim = len(self.truth_variables) + 4
        self.global_dim = self.config["global_dim"]

        self.time_embedding = TimestepEmbedder(self.hidden_dim)

        self.truth_init = DenseNetwork(
            inpt_dim=self.truth_in_dim,
            outp_dim=self.hidden_dim,
            hddn_dim=[2 * self.hidden_dim],
            act_h=act,
        )

        self.noisy_init = DenseNetwork(
            inpt_dim=self.noisy_dim,
            outp_dim=self.hidden_dim,
            hddn_dim=[2 * self.hidden_dim],
            act_h=act,
        )

        self.resnet = nn.Sequential(
            *[
                BasicBlock(self.hidden_dim, self.hidden_dim)
                for _ in range(self.n_res_blocks)
            ]
        )

        self.final_layer = DenseNetwork(
            inpt_dim=self.hidden_dim,
            outp_dim=self.noisy_dim,
            hddn_dim=[self.hidden_dim],
            act_h=act,
            ctxt_dim=3 * self.hidden_dim,
        )

        self.global_embedding = DenseNetwork(
            inpt_dim=self.global_dim,
            outp_dim=self.hidden_dim,
            hddn_dim=[2 * self.hidden_dim],
            act_h=act,
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(3 * self.hidden_dim, self.hidden_dim * 2, bias=True)
        )
        self.norm = nn.LayerNorm(self.hidden_dim, elementwise_affine=False)

    def forward(
        self,
        noisy_data,
        truth_data,
        mask,
        global_data,
        timestep,
    ):
        truth_mask = mask

        truth_embd = self.truth_init(truth_data)
        truth_ctxt = torch.sum(
            truth_embd * truth_mask.unsqueeze(-1), dim=1
        ) / torch.sum(truth_mask, dim=1, keepdim=True)
        time_embd = self.time_embedding(timestep)

        ctxt = torch.cat(
            [
                truth_ctxt,
                self.global_embedding(global_data),
                time_embd,
            ],
            -1,
        )

        (
            shift_mlp,
            scale_mlp,
        ) = self.adaLN_modulation(ctxt).chunk(2, dim=-1)

        noisy_embd = self.noisy_init(noisy_data)
        noisy_embd = (1 + scale_mlp) * noisy_embd + shift_mlp
        noisy_embd = self.resnet(noisy_embd)
        noisy_embd = self.norm(noisy_embd)
        noisy_out = self.final_layer(noisy_embd, ctxt)

        return noisy_out
