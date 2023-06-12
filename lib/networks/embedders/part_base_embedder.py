from re import I
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sympy import nextprime
from lib.config import cfg
from termcolor import cprint
from lib.train.trainers.trainer import cuda_context


class Embedder(nn.Module):
    def __init__(self,
                 pid=-1,
                 partname='undefined',
                 bbox=np.array([
                     [0, 0, 0],
                     [1, 1, 1]
                 ]),
                 n_levels=16,
                 n_features_per_level=16,
                 b=1.38,
                 log2_hashmap_size=18,
                 base_resolution=2,
                 sum=True,
                 sum_over_features=True,
                 separate_dense=True,
                 use_batch_bounds=cfg.use_batch_bounds,
                 include_input=True,  # this will pass gradient better to input, but if you're using uvt, no need
                 ):
        """
        WIP:
        best iter speed: separate_dense = True
        best performace: separate_dense = False, sum_over_features = True
        """
        super().__init__()
        self.pid = pid
        self.partname = partname
        self.n_levels = n_levels
        self.include_input = include_input
        self.use_batch_bounds = use_batch_bounds
        self.n_entries_per_level = nextprime(2**log2_hashmap_size)

        self.b = b
        self.f = n_features_per_level
        self.base_resolution = base_resolution

        self.bounds = nn.Parameter(torch.tensor(np.array(bbox).reshape((2, 3))).float(), requires_grad=False)

        # every level should have this number of entries per side
        # we'd like the border to be mapped inside 0, 1
        self.entries_num = [int((self.base_resolution * self.b**i)) for i in range(self.n_levels)]
        self.entries_cnt = [self.entries_num[i] ** 3 for i in range(self.n_levels)]
        self.entries_size = [1 / (self.entries_num[i] - 1) for i in range(self.n_levels)]
        self.entries_min = [0 for i in range(self.n_levels)]

        self.entries_size = nn.Parameter(torch.tensor(self.entries_size), requires_grad=False)
        self.entries_num = nn.Parameter(torch.tensor(self.entries_num), requires_grad=False)
        self.entries_min = nn.Parameter(torch.tensor(self.entries_min), requires_grad=False)
        self.entries_cnt = nn.Parameter(torch.tensor(self.entries_cnt), requires_grad=False)
        self.entries_sum = nn.Parameter(self.entries_cnt.cumsum(dim=-1), requires_grad=False)

        self.start_hash = self.n_levels
        for i in range(n_levels):
            if self.entries_cnt[i] > self.n_entries_per_level:
                self.start_hash = i
                break
        self.len_hash = self.n_levels - self.start_hash
        self.separate_dense = separate_dense and self.start_hash  # when everything needs to be hashed for example when body using using small table
        if self.separate_dense:
            data = torch.zeros((self.n_levels, self.n_entries_per_level, self.f))
            nn.init.kaiming_normal_(data)  # NOTE: initialization matters! separate_dense doesn't work well if we initialize the self.dense and self.hash data separately
            dense = torch.cat([data[i, :self.entries_cnt[i], :] for i in range(self.start_hash)], dim=0)
            hash = data[self.start_hash:, :, :]
            self.dense = nn.Parameter(dense)  # sum(non-hash), F
            self.hash = nn.Parameter(hash)  # H, T, F
        else:
            self.hash = nn.Parameter(torch.zeros((self.n_levels, self.n_entries_per_level, self.f)))  # H, T, F
            nn.init.kaiming_normal_(self.hash)

        self.offsets = nn.Parameter(torch.tensor([[0., 0., 0.],
                                                  [0., 0., 1.],
                                                  [0., 1., 0.],
                                                  [0., 1., 1.],
                                                  [1., 0., 0.],
                                                  [1., 0., 1.],
                                                  [1., 1., 0.],
                                                  [1., 1., 1.]]).float(), requires_grad=False)

        self.sum = sum
        self.sum_over_features = sum_over_features

        self.out_dim = 0

        if self.sum:
            if self.sum_over_features:
                self.out_dim += self.n_levels
            else:
                self.out_dim += self.f
        else:
            self.out_dim += self.f * self.n_levels

        if include_input:
            self.out_dim += 3

    def forward(self, xyz: torch.Tensor, batch):
        if self.use_batch_bounds and 'iter_step' in batch and batch['iter_step'] == 1:
            # cprint(f'part: {self.partname}\nori bbox:\n{self.bounds}\nnew bbox:\n{batch["bounds"][0][self.pid]}', color='green')
            self.bounds = nn.Parameter(batch['bounds'][0][self.pid], requires_grad=False)

        N, _ = xyz.shape  # N, 3
        xyz = (xyz - self.bounds[0]) / (self.bounds[1] - self.bounds[0])  # normalized, N, 3

        ind_xyz = xyz[None].expand(self.n_levels, -1, -1)  # L, N, 3
        flt_xyz = ind_xyz / self.entries_size[:, None, None]  # L, N, 3
        int_xyz = (flt_xyz[:, :, None] + self.offsets[None, None]).long()  # will round to zero, L, N, 8, 3
        int_xyz = int_xyz.clip(self.entries_min[:, None, None, None], self.entries_num[:, None, None, None]-1)
        off_xyz = flt_xyz - int_xyz[:, :, 0]  # L, N, 3

        sh = self.start_hash
        nl = self.n_levels

        # x as first digit, y as second digit, z as last digit -> S, N, 8
        ind_dense: torch.Tensor = \
            int_xyz[:sh, ..., 0] * (self.entries_num[:sh]**2)[:, None, None] + \
            int_xyz[:sh, ..., 1] * (self.entries_num[:sh])[:, None, None] + \
            int_xyz[:sh, ..., 2]
        if self.separate_dense:
            ind_dense[1:] = ind_dense[1:] + self.entries_sum[:self.start_hash-1][:, None, None]  # S, N, 8

        # hashing -> H, N, 8
        ind_hash: torch.Tensor = (
            int_xyz[sh:, ..., 0]*cfg.ps[0] ^
            int_xyz[sh:, ..., 1]*cfg.ps[1] ^
            int_xyz[sh:, ..., 2]*cfg.ps[2]
        ) % self.n_entries_per_level
        if not self.separate_dense:
            ind = torch.cat([ind_dense, ind_hash], dim=0)

        # data: L, T, F, ind: L, N, 8 -> L, N, 8, F feature
        # NOTE: gather backward is much faster than index_select
        # val = self.data[torch.arange(nl, dtype=torch.long, device=ind.device)[..., None, None], ind, :]  # -> L, N, 8, F
        L, T, F = self.n_levels, self.n_entries_per_level, self.f
        S, H = self.start_hash, self.n_levels - self.start_hash

        # MARK: this is the first optimizable step, should wait for previous stream to finish updating here
        if 'prev_stream' in cuda_context:
            cuda_context.curr_stream.wait_stream(cuda_context.prev_stream)  # wait for previous stream update to finish
        if self.separate_dense:
            val_dense = self.dense.gather(dim=0, index=ind_dense.view(S * N * 8)[..., None].expand(-1, F)).view(S, N, 8, F)
            val_hash = self.hash.gather(dim=1, index=ind_hash.view(H, N * 8)[..., None].expand(-1, -1, F)).view(H, N, 8, F)
            val = torch.cat([val_dense, val_hash], dim=0)
        else:
            val = self.hash.gather(dim=1, index=ind.view(L, N * 8)[..., None].expand(-1, -1, F)).view(L, N, 8, F)

        # off: L, N, 3, sets: 8, 3 -> L, N, :, 3 and :, :, 8, 3, compute xyz distance to the other corner, mul: multiplier
        mul_xyz = (1 - self.offsets[None, None]) + (2 * self.offsets[None, None] - 1.) * off_xyz[:, :, None]
        mul_xyz = mul_xyz[..., 0] * mul_xyz[..., 1] * mul_xyz[..., 2]  # L, N, 8
        val = (mul_xyz[..., None] * val).sum(dim=-2)  # trilinear interpolated feature, L, N, F

        # feature aggregation
        val = val.permute(1, 0, 2)  # N, L, F
        if self.sum:
            if self.sum_over_features:
                val = val.sum(dim=-1)  # N, F, NOTE: sum over features seems to be producing better results...
            else:
                val = val.sum(dim=-2)  # N, L, NOTE: sum over features seems to be producing better results...
        else:
            val = val.reshape(-1, L*F)  # N, L*F

        # feature boosting
        if self.include_input:
            val = torch.cat([xyz, val], dim=-1)
        return val


if __name__ == "__main__":
    torch.manual_seed(0)
    xyz = torch.Tensor(
        [
            [-0.2, 0.4, 0.3],
            [0.3, -0.7, -0.3]
        ]
    ).cuda()
    batch = {
        'frame_dim': [0]
    }
    embedder = Embedder()
    print(embedder(xyz, batch))
