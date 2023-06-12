import torch
import torch.nn as nn
from lib.utils.blend_utils import pts_sample_uv
from lib.networks.make_network import make_embedder


class Deformer(nn.Module):
    """
    Deform using array
    """

    def __init__(self, deformer_cfg) -> None:
        super().__init__()
        self.embedder = make_embedder(deformer_cfg)
        self.mlp = nn.Sequential(
            nn.Linear(self.embedder.out_dim, 32),
            nn.Softplus(),
            nn.Linear(32, 32),
            nn.Softplus(),
            nn.Linear(32, 3),
        )

    def forward(self, xyz: torch.Tensor, batch, flag: torch.Tensor = None):
        if flag is not None:
            B, NP, _ = xyz.shape
            # flag: B, N
            assert B == 1
            ret = torch.zeros(B, NP, 3, device=xyz.device, dtype=xyz.dtype)
            inds = flag[0].nonzero(as_tuple=True)[0][:, None].expand(-1, 3)
            xyz = xyz[0].gather(dim=0, index=inds)

        uv = pts_sample_uv(xyz, batch['tuv'], batch['tbounds'], mode='bilinear')  # uv: B, 2, N
        uv = uv.permute(0, 2, 1)  # B, N, 2
        uv = uv.view(-1, uv.shape[-1])  # B*N, 2
        t = batch['frame_dim'].expand(uv.shape[0], -1).float()
        uvt = torch.cat([uv, t], dim=-1)  # B*N, 3
        feat = self.embedder(uvt, batch)
        resd = self.mlp(feat)
        resd_tan = 0.05 * torch.tanh(resd)  # B*N, 3

        if flag is not None:
            ret[0, inds[:, 0]] = resd_tan.to(ret.dtype, non_blocking=True)  # ignoring batch dimension
            return ret
        else:
            return resd_tan
