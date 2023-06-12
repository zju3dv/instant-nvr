import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.config import cfg
from lib.networks.make_network import make_viewdir_embedder, make_residual, make_part_color_network, make_part_embedder, make_deformer
from lib.networks.embedders.part_base_embedder import Embedder as HashEmbedder
from lib.networks.embedders.freq_embedder import Embedder as FreqEmbedder


class MLP(nn.Module):
    def __init__(self, indim=16, outdim=3, d_hidden=64, n_layers=2):
        super(MLP, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.linears = nn.ModuleList([nn.Linear(indim, d_hidden)] + [nn.Linear(d_hidden, d_hidden) for i in range(n_layers - 1)] + [nn.Linear(d_hidden, outdim)])
        self.actvn = nn.Softplus()

    def forward(self, input):
        net = input
        for i, l in enumerate(self.linears[:-1]):
            net = self.actvn(l(net))
        net = self.linears[-1](net)
        return net


ColorNetwork = MLP


class Network(nn.Module):
    def __init__(self, partname, pid):
        super().__init__()
        self.pid = pid
        self.partname = partname

        self.embedder: HashEmbedder = make_part_embedder(cfg, partname, pid)
        self.embedder_dir: FreqEmbedder = make_viewdir_embedder(cfg)
        self.occ = MLP(self.embedder.out_dim, 1 + cfg.geo_feature_dim, **cfg.network.occ)
        indim_rgb = self.embedder.out_dim + self.embedder_dir.out_dim + cfg.geo_feature_dim + cfg.latent_code_dim
        self.rgb_latent = nn.Parameter(torch.zeros(cfg.num_latent_code, cfg.latent_code_dim))
        nn.init.kaiming_normal_(self.rgb_latent)
        self.rgb = make_part_color_network(cfg, partname, indim=indim_rgb)

    def forward(self, tpts: torch.Tensor, viewdir: torch.Tensor, dists: torch.Tensor, batch):
        # tpts: N, 3
        # viewdir: N, 3
        N, D = tpts.shape
        C, L = self.rgb_latent.shape
        embedded = self.embedder(tpts, batch)  # embedding
        hidden: torch.Tensor = self.occ(embedded)  # networking
        occ = 1 - torch.exp(-self.occ.actvn(hidden[..., :1]))  # activation
        feature = hidden[..., 1:]

        embedded_dir = self.embedder_dir(viewdir, batch)  # embedding
        latent_code = self.rgb_latent.gather(dim=0, index=batch['latent_index'].expand(N, L))  # NOTE: ignoring batch dimension
        input = torch.cat([embedded, embedded_dir, feature, latent_code], dim=-1)
        rgb: torch.Tensor = self.rgb(input)  # networking
        rgb = rgb.sigmoid()  # activation

        raw = torch.cat([rgb, occ], dim=-1)
        ret = {'raw': raw, 'occ': occ}

        return ret
