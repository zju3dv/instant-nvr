import torch
from lib.config import cfg
import torch.nn as nn

class PosEnc(nn.Module):
    def __init__(self, multires, periodic_fns=[torch.sin, torch.cos], retain_input=True):
        super(PosEnc, self).__init__()
        freq_bands = 2.**torch.linspace(0., multires-1, steps=multires)  # (multires)
        freq_bands = freq_bands[..., None, None].expand(multires, len(periodic_fns), 1).clone()  # (multires, 2, 1)
        self.freq_bands = nn.Parameter(freq_bands, requires_grad=False)
        # self.register_buffer('freq_bands', freq_bands)
        self.multires = multires
        self.periodic_fns = periodic_fns
        self.retain_input = retain_input

    def get_dim(self, dim):
        return self.freq_bands.numel() * dim + (dim if self.retain_input else 0)

    # FIXME: LRU_CACHE WILL MAKE YOU UNABLE TO UPDATE INPUT PARAMETER
    def forward(self, inputs):
        # inputs: B, N, 3
        n_b_dim = len(inputs.shape)-1
        dim = inputs.shape[-1]
        ori_inputs = inputs
        inputs = inputs.view(*inputs.shape[:-1], 1, 1, inputs.shape[-1])  # (B, N, 1, 1, 3)
        inputs = inputs * self.freq_bands[(None,)*n_b_dim]  # (B, N, 1, 1, 3) * (1, 1, multires, 2, 3) -> (B, N, multires, 2, 3)
        inputs = torch.cat([self.periodic_fns[i](t) for i, t in enumerate(torch.split(inputs, 1, dim=-2))], dim=-2)
        inputs = inputs.view(*ori_inputs.shape[:-1], self.freq_bands.numel() * dim)  # (B, N, embed_dim - 3?)
        if self.retain_input:
            inputs = torch.cat([ori_inputs, inputs], dim=-1)
        return inputs

def get_embedder(multires, input_dims=3, periodic_fns=[torch.sin, torch.cos], retain_input=True):
    embedder = PosEnc(multires, periodic_fns=periodic_fns, retain_input=retain_input)
    return embedder, embedder.get_dim(input_dims)

class Embedder(nn.Module):
    def __init__(self, res, input_dims, F=2) -> None:
        super().__init__()
        self.embedder, self.out_dim = get_embedder(res, input_dims)
    
    def forward(self, x, batch):
        return self.embedder(x)

xyz_embedder, xyz_dim = get_embedder(cfg.xyz_res)
view_embedder, view_dim = get_embedder(cfg.view_res)
