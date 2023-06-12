import torch
import torch.nn.functional as F
import numpy as np
from lib.config import cfg
from pytorch3d.ops.knn import knn_points
from lib.utils.base_utils import DotDict
from typing import Tuple

NUM_PARTS = 5
part_bw_map = {
    'body': [14, 13, 9, 6, 3, 0],
    'leg': [1, 2, 4, 5, 7, 8, 10, 11],
    'head': [12, 15],
    'larm': [16, 18, 20, 22],
    'rarm': [17, 19, 21, 23],
}
partnames = ['body', 'leg', 'head', 'larm', 'rarm']

if cfg.part3:
    NUM_PARTS = 3
    part_bw_map = {
        'body': [14, 13, 9, 6, 3, 0, 16, 18, 20, 22, 17, 19, 21, 23],
        'head': [12, 15],
        'leg': [1, 2, 4, 5, 7, 8, 10, 11],
    }
    partnames = ['body', 'head', 'leg']

if cfg.part6:
    NUM_PARTS = 6
    part_bw_map = {
        'body': [14, 13, 9, 6, 3, 0],
        'head': [12, 15],
        'lleg': [1, 4, 7, 10],
        'rleg': [2, 5, 8, 11],
        'larm': [16, 18, 20, 22],
        'rarm': [17, 19, 21, 23],
    }
    partnames = ['body', 'lleg', 'rleg', 'head', 'larm', 'rarm']

# if cfg.part4:
#     NUM_PARTS = 4
#     part_bw_map = {
#         'body': [14, 13, 9, 6, 3, 0, 1, 2, 4, 5, 7, 8, 10, 11],
#         'head': [12, 15],
#         'larm': [16, 18, 20, 22],
#         'rarm': [17, 19, 21, 23],
#     }
#     partnames = ['body', 'head', 'larm', 'rarm']


def grid_sample(input: torch.Tensor, grid: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    # https://github.com/pytorch/pytorch/issues/34704
    # RuntimeError: derivative for grid_sampler_2d_backward is not implemented
    # this implementation might be slower than the cuda one
    # if args or kwargs:
    #     # warnings.warn(message=f'unused arguments for grid_sample: {args}, {kwargs}')
    #     return F.grid_sample(input, grid, *args, **kwargs)
    if input.ndim == 4:
        # invoke 2d custom grid_sampling
        assert grid.ndim == 4, '4d input needs a 4d grid'
        return grid_sample_2d(input, grid)
    elif input.ndim == 5:
        # invoke 3d custom grid_sampling
        assert grid.ndim == 5, '5d input needs a 5d grid'
        return grid_sample_3d(input, grid)
    else:
        raise NotImplementedError(f'grid_sample not implemented for {input.ndim}d input')


def grid_sample_2d(image, optical):
    N, C, IH, IW = image.shape
    _, H, W, _ = optical.shape

    ix = optical[..., 0]
    iy = optical[..., 1]

    ix = ((ix + 1) / 2) * (IW-1)
    iy = ((iy + 1) / 2) * (IH-1)
    with torch.no_grad():
        ix_nw = torch.floor(ix)
        iy_nw = torch.floor(iy)
        ix_ne = ix_nw + 1
        iy_ne = iy_nw
        ix_sw = ix_nw
        iy_sw = iy_nw + 1
        ix_se = ix_nw + 1
        iy_se = iy_nw + 1

    nw = (ix_se - ix) * (iy_se - iy)
    ne = (ix - ix_sw) * (iy_sw - iy)
    sw = (ix_ne - ix) * (iy - iy_ne)
    se = (ix - ix_nw) * (iy - iy_nw)

    with torch.no_grad():
        torch.clamp(ix_nw, 0, IW-1, out=ix_nw)
        torch.clamp(iy_nw, 0, IH-1, out=iy_nw)

        torch.clamp(ix_ne, 0, IW-1, out=ix_ne)
        torch.clamp(iy_ne, 0, IH-1, out=iy_ne)

        torch.clamp(ix_sw, 0, IW-1, out=ix_sw)
        torch.clamp(iy_sw, 0, IH-1, out=iy_sw)

        torch.clamp(ix_se, 0, IW-1, out=ix_se)
        torch.clamp(iy_se, 0, IH-1, out=iy_se)

    image = image.view(N, C, IH * IW)

    nw_val = torch.gather(image, 2, (iy_nw * IW + ix_nw).long().view(N, 1, H * W).repeat(1, C, 1))
    ne_val = torch.gather(image, 2, (iy_ne * IW + ix_ne).long().view(N, 1, H * W).repeat(1, C, 1))
    sw_val = torch.gather(image, 2, (iy_sw * IW + ix_sw).long().view(N, 1, H * W).repeat(1, C, 1))
    se_val = torch.gather(image, 2, (iy_se * IW + ix_se).long().view(N, 1, H * W).repeat(1, C, 1))

    out_val = (nw_val.view(N, C, H, W) * nw.view(N, 1, H, W) +
               ne_val.view(N, C, H, W) * ne.view(N, 1, H, W) +
               sw_val.view(N, C, H, W) * sw.view(N, 1, H, W) +
               se_val.view(N, C, H, W) * se.view(N, 1, H, W))

    return out_val


def grid_sample_3d(image, optical):
    N, C, ID, IH, IW = image.shape
    _, D, H, W, _ = optical.shape

    ix = optical[..., 0]
    iy = optical[..., 1]
    iz = optical[..., 2]

    ix = ((ix + 1) / 2) * (IW - 1)
    iy = ((iy + 1) / 2) * (IH - 1)
    iz = ((iz + 1) / 2) * (ID - 1)
    with torch.no_grad():

        ix_tnw = torch.floor(ix)
        iy_tnw = torch.floor(iy)
        iz_tnw = torch.floor(iz)

        ix_tne = ix_tnw + 1
        iy_tne = iy_tnw
        iz_tne = iz_tnw

        ix_tsw = ix_tnw
        iy_tsw = iy_tnw + 1
        iz_tsw = iz_tnw

        ix_tse = ix_tnw + 1
        iy_tse = iy_tnw + 1
        iz_tse = iz_tnw

        ix_bnw = ix_tnw
        iy_bnw = iy_tnw
        iz_bnw = iz_tnw + 1

        ix_bne = ix_tnw + 1
        iy_bne = iy_tnw
        iz_bne = iz_tnw + 1

        ix_bsw = ix_tnw
        iy_bsw = iy_tnw + 1
        iz_bsw = iz_tnw + 1

        ix_bse = ix_tnw + 1
        iy_bse = iy_tnw + 1
        iz_bse = iz_tnw + 1

    tnw = (ix_bse - ix) * (iy_bse - iy) * (iz_bse - iz)
    tne = (ix - ix_bsw) * (iy_bsw - iy) * (iz_bsw - iz)
    tsw = (ix_bne - ix) * (iy - iy_bne) * (iz_bne - iz)
    tse = (ix - ix_bnw) * (iy - iy_bnw) * (iz_bnw - iz)
    bnw = (ix_tse - ix) * (iy_tse - iy) * (iz - iz_tse)
    bne = (ix - ix_tsw) * (iy_tsw - iy) * (iz - iz_tsw)
    bsw = (ix_tne - ix) * (iy - iy_tne) * (iz - iz_tne)
    bse = (ix - ix_tnw) * (iy - iy_tnw) * (iz - iz_tnw)

    with torch.no_grad():

        torch.clamp(ix_tnw, 0, IW - 1, out=ix_tnw)
        torch.clamp(iy_tnw, 0, IH - 1, out=iy_tnw)
        torch.clamp(iz_tnw, 0, ID - 1, out=iz_tnw)

        torch.clamp(ix_tne, 0, IW - 1, out=ix_tne)
        torch.clamp(iy_tne, 0, IH - 1, out=iy_tne)
        torch.clamp(iz_tne, 0, ID - 1, out=iz_tne)

        torch.clamp(ix_tsw, 0, IW - 1, out=ix_tsw)
        torch.clamp(iy_tsw, 0, IH - 1, out=iy_tsw)
        torch.clamp(iz_tsw, 0, ID - 1, out=iz_tsw)

        torch.clamp(ix_tse, 0, IW - 1, out=ix_tse)
        torch.clamp(iy_tse, 0, IH - 1, out=iy_tse)
        torch.clamp(iz_tse, 0, ID - 1, out=iz_tse)

        torch.clamp(ix_bnw, 0, IW - 1, out=ix_bnw)
        torch.clamp(iy_bnw, 0, IH - 1, out=iy_bnw)
        torch.clamp(iz_bnw, 0, ID - 1, out=iz_bnw)

        torch.clamp(ix_bne, 0, IW - 1, out=ix_bne)
        torch.clamp(iy_bne, 0, IH - 1, out=iy_bne)
        torch.clamp(iz_bne, 0, ID - 1, out=iz_bne)

        torch.clamp(ix_bsw, 0, IW - 1, out=ix_bsw)
        torch.clamp(iy_bsw, 0, IH - 1, out=iy_bsw)
        torch.clamp(iz_bsw, 0, ID - 1, out=iz_bsw)

        torch.clamp(ix_bse, 0, IW - 1, out=ix_bse)
        torch.clamp(iy_bse, 0, IH - 1, out=iy_bse)
        torch.clamp(iz_bse, 0, ID - 1, out=iz_bse)

    image = image.view(N, C, ID * IH * IW)

    tnw_val = torch.gather(image, 2, (iz_tnw * IW * IH + iy_tnw * IW + ix_tnw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    tne_val = torch.gather(image, 2, (iz_tne * IW * IH + iy_tne * IW + ix_tne).long().view(N, 1, D * H * W).repeat(1, C, 1))
    tsw_val = torch.gather(image, 2, (iz_tsw * IW * IH + iy_tsw * IW + ix_tsw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    tse_val = torch.gather(image, 2, (iz_tse * IW * IH + iy_tse * IW + ix_tse).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bnw_val = torch.gather(image, 2, (iz_bnw * IW * IH + iy_bnw * IW + ix_bnw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bne_val = torch.gather(image, 2, (iz_bne * IW * IH + iy_bne * IW + ix_bne).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bsw_val = torch.gather(image, 2, (iz_bsw * IW * IH + iy_bsw * IW + ix_bsw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bse_val = torch.gather(image, 2, (iz_bse * IW * IH + iy_bse * IW + ix_bse).long().view(N, 1, D * H * W).repeat(1, C, 1))

    out_val = (tnw_val.view(N, C, D, H, W) * tnw.view(N, 1, D, H, W) +
               tne_val.view(N, C, D, H, W) * tne.view(N, 1, D, H, W) +
               tsw_val.view(N, C, D, H, W) * tsw.view(N, 1, D, H, W) +
               tse_val.view(N, C, D, H, W) * tse.view(N, 1, D, H, W) +
               bnw_val.view(N, C, D, H, W) * bnw.view(N, 1, D, H, W) +
               bne_val.view(N, C, D, H, W) * bne.view(N, 1, D, H, W) +
               bsw_val.view(N, C, D, H, W) * bsw.view(N, 1, D, H, W) +
               bse_val.view(N, C, D, H, W) * bse.view(N, 1, D, H, W))

    return out_val


# Batched inverse of lower triangular matrices
def torch_inverse_decomp(L: torch.Tensor, eps=1e-10):
    n = L.shape[-1]
    invL = torch.zeros_like(L)
    for j in range(0, n):
        invL[..., j, j] = 1.0 / (L[..., j, j] + eps)
        for i in range(j+1, n):
            S = 0.0
            for k in range(i+1):
                S = S - L[..., i, k] * invL[..., k, j].clone()
            invL[..., i, j] = S / (L[..., i, i] + eps)

    return invL


g_idx_i = torch.tensor(
    [
        [
            [[1, 1], [2, 2]],
            [[1, 1], [2, 2]],
            [[1, 1], [2, 2]],
        ],
        [
            [[0, 0], [2, 2]],
            [[0, 0], [2, 2]],
            [[0, 0], [2, 2]],
        ],
        [
            [[0, 0], [1, 1]],
            [[0, 0], [1, 1]],
            [[0, 0], [1, 1]],
        ],
    ], device='cuda', dtype=torch.long)

g_idx_j = torch.tensor(
    [
        [
            [[1, 2], [1, 2]],
            [[0, 2], [0, 2]],
            [[0, 1], [0, 1]],
        ],
        [
            [[1, 2], [1, 2]],
            [[0, 2], [0, 2]],
            [[0, 1], [0, 1]],
        ],
        [
            [[1, 2], [1, 2]],
            [[0, 2], [0, 2]],
            [[0, 1], [0, 1]],
        ],
    ], device='cuda', dtype=torch.long)

g_signs = torch.tensor([
    [+1, -1, +1],
    [-1, +1, -1],
    [+1, -1, +1],
], device='cuda', dtype=torch.long)


def torch_inverse_3x3(R: torch.Tensor, eps=torch.finfo(torch.float).eps):
    # B, N, 3, 3
    """
    a, b, c | m00, m01, m02
    d, e, f | m10, m11, m12
    g, h, i | m20, m21, m22
    """
    B, N, _, _ = R.shape

    minors = R.new_zeros(B, N, 3, 3, 2, 2)
    idx_i = g_idx_i.to(R.device)  # almost never need to copy
    idx_j = g_idx_j.to(R.device)  # almost never need to copy
    signs = g_signs.to(R.device)  # almost never need to copy

    for i in range(3):
        for j in range(3):
            minors[:, :, i, j, :, :] = R[:, :, idx_i[i, j], idx_j[i, j]]

    minors = minors[:, :, :, :, 0, 0] * minors[:, :, :, :, 1, 1] - minors[:, :, :, :, 0, 1] * minors[:, :, :, :, 1, 0]
    cofactors = minors * signs[None, None]  # 3,3 -> B,N,3,3
    cofactors_t = cofactors.transpose(-2, -1)  # B, N, 3, 3
    determinant = R[:, :, 0, 0] * minors[:, :, 0, 0] - R[:, :, 0, 1] * minors[:, :, 0, 1] + R[:, :, 0, 2] * minors[:, :, 0, 2]  # B, N
    inverse = cofactors_t / (determinant[:, :, None, None] + eps)

    return inverse


def torch_inverse_3x3_separated(R: torch.Tensor):  # 1e-10 might gives trouble for fp16
    # R = R.to(torch.float, non_blocking=True)
    eps = torch.finfo(R.dtype).eps
    # n_batch, n_bones, 3, 3
    """
    a, b, c | m00, m01, m02
    d, e, f | m10, m11, m12
    g, h, i | m20, m21, m22
    """
    # convenient access
    r00 = R[..., 0, 0]
    r01 = R[..., 0, 1]
    r02 = R[..., 0, 2]
    r10 = R[..., 1, 0]
    r11 = R[..., 1, 1]
    r12 = R[..., 1, 2]
    r20 = R[..., 2, 0]
    r21 = R[..., 2, 1]
    r22 = R[..., 2, 2]

    # determinant of matrix minors
    m00 = + r11 * r22 - r21 * r12
    m01 = - r10 * r22 + r20 * r12
    m02 = + r10 * r21 - r20 * r11
    m10 = - r01 * r22 + r21 * r02
    m11 = + r00 * r22 - r20 * r02
    m12 = - r00 * r21 + r20 * r01
    m20 = + r01 * r12 - r11 * r02
    m21 = - r00 * r12 + r10 * r02
    m22 = + r00 * r11 - r10 * r01

    # transpose of determinant of matrix minors
    col0 = torch.stack([m00, m01, m02], dim=-1)
    col1 = torch.stack([m10, m11, m12], dim=-1)
    col2 = torch.stack([m20, m21, m22], dim=-1)
    m = torch.stack([col0, col1, col2], dim=-1)

    # determinant of matrix
    d = r00 * m00 + r01 * m01 + r02 * m02

    # inverse of 3x3 matrix
    inv = m / (d[..., None, None] + eps)

    return inv


def world_points_to_pose_points(wpts, Rh, Th):
    """
    wpts: n_batch, n_points, 3
    Rh: n_batch, 3, 3
    Th: n_batch, 1, 3
    """
    pts = torch.matmul(wpts - Th, Rh)
    return pts


def world_dirs_to_pose_dirs(wdirs, Rh):
    """
    wdirs: n_batch, n_points, 3
    Rh: n_batch, 3, 3
    """
    pts = torch.matmul(wdirs, Rh)
    return pts


def pose_points_to_world_points(ppts, Rh, Th):
    """
    ppts: n_batch, n_points, 3
    Rh: n_batch, 3, 3
    Th: n_batch, 1, 3
    """
    pts = torch.matmul(ppts, Rh.transpose(1, 2)) + Th
    return pts


def get_blend_params(bw: torch.Tensor, A: torch.Tensor):
    """transform points from the pose space to the T pose
    ppts: n_batch, n_points, 3
    bw: n_batch, 24, n_points
    A: n_batch, 24, 4, 4
    """
    B, K, N = bw.shape
    bw = bw.permute(0, 2, 1)
    A_bw = torch.bmm(bw, A.view(B, K, -1))
    A_bw = A_bw.view(B, -1, 4, 4)
    return A_bw


def get_inverse_blend_params(bw: torch.Tensor, A: torch.Tensor):
    """transform points from the pose space to the T pose
    ppts: n_batch, n_points, 3
    bw: n_batch, 24, n_points
    A: n_batch, 24, 4, 4
    """
    B, K, N = bw.shape
    bw = bw.permute(0, 2, 1)
    A_bw = torch.bmm(bw, A.view(B, K, -1))
    A_bw = A_bw.view(B, -1, 4, 4)
    R_inv = torch_inverse_3x3(A_bw[..., :3, :3])
    return A_bw, R_inv


def pose_points_to_tpose_points(ppts: torch.Tensor, bw=None, A=None, A_bw=None, R_inv=None):
    """transform points from the pose space to the T pose
    ppts: n_batch, n_points, 3
    bw: n_batch, 24, n_points
    A: n_batch, 24, 4, 4
    """
    if A_bw is None:
        sh = ppts.shape
        bw = bw.permute(0, 2, 1)
        A_bw = torch.bmm(bw, A.view(sh[0], 24, -1))
        A_bw = A_bw.view(sh[0], -1, 4, 4)
    pts = ppts - A_bw[..., :3, 3]
    if R_inv is None:
        R_inv = torch_inverse_3x3(A_bw[..., :3, :3])
    pts = torch.sum(R_inv * pts[:, :, None], dim=3)
    return pts


def pose_dirs_to_tpose_dirs(pdirs: torch.Tensor, bw=None, A=None, A_bw=None, R_inv=None):
    """transform directions from the pose space to the T pose
    ddirs: n_batch, n_points, 3
    bw: n_batch, 24, n_points
    A: n_batch, 24, 4, 4
    """
    if A_bw is None:
        sh = pdirs.shape
        bw = bw.permute(0, 2, 1)
        A_bw = torch.bmm(bw, A.view(sh[0], 24, -1))
        A_bw = A_bw.view(sh[0], -1, 4, 4)
    if R_inv is None:
        R_inv = torch_inverse_3x3(A_bw[..., :3, :3])
    pts = torch.sum(R_inv * pdirs[:, :, None], dim=3)
    return pts


def tpose_points_to_pose_points(pts, bw=None, A=None, A_bw=None):
    """transform points from the T pose to the pose space
    ppts: n_batch, n_points, 3
    bw: n_batch, 24, n_points
    A: n_batch, 24, 4, 4
    """
    if A_bw is None:
        sh = pts.shape
        bw = bw.permute(0, 2, 1)
        A = torch.bmm(bw, A.view(sh[0], 24, -1))
        A_bw = A.view(sh[0], -1, 4, 4)
    R = A_bw[..., :3, :3]
    pts = torch.sum(R * pts[:, :, None], dim=3)
    pts = pts + A_bw[..., :3, 3]
    return pts


def tpose_dirs_to_pose_dirs(ddirs, bw=None, A=None, A_bw=None):
    """transform directions from the T pose to the pose space
    ddirs: n_batch, n_points, 3
    bw: n_batch, 24, n_points
    A: n_batch, 24, 4, 4
    """
    if A_bw is None:
        sh = ddirs.shape
        bw = bw.permute(0, 2, 1)
        A = torch.bmm(bw, A.view(sh[0], 24, -1))
        A_bw = A.view(sh[0], -1, 4, 4)
    R = A_bw[..., :3, :3]
    pts = torch.sum(R * ddirs[:, :, None], dim=3)
    return pts


def grid_sample_blend_weights(grid_coords, bw):
    # the blend weight is indexed by xyz
    grid_coords = grid_coords[:, None, None]
    bw = F.grid_sample(bw,
                       grid_coords,
                       padding_mode='border',
                       align_corners=True)
    bw = bw[:, :, 0, 0]
    return bw


def pts_sample_blend_weights(pts, bw, bounds):
    """sample blend weights for points
    pts: n_batch, n_points, 3
    bw: n_batch, d, h, w, 25
    bounds: n_batch, 2, 3
    """
    # interpolate blend weights
    min_xyz = bounds[:, 0]
    max_xyz = bounds[:, 1]
    bounds = max_xyz[:, None] - min_xyz[:, None]
    grid_coords = (pts - min_xyz[:, None]) / bounds
    grid_coords: torch.Tensor = grid_coords * 2 - 1
    # convert xyz to zyx, since the blend weight is indexed by xyz
    grid_coords = grid_coords.flip(-1)

    # the blend weight is indexed by xyz
    bw = bw.permute(0, 4, 1, 2, 3)
    grid_coords = grid_coords[:, None, None]
    bw = F.grid_sample(bw,
                       grid_coords,
                       padding_mode='border',
                       align_corners=True)
    bw = bw[:, :, 0, 0]

    return bw


def pts_sample_uv(pts, uv, bounds, mode='bilinear'):
    """sample uv coords for points
    pts: n_batch, n_points, 3
    bw: n_batch, d, h, w, 2
    bounds: n_batch, 2, 3
    """
    pts = pts.clone()

    # interpolate blend weights
    min_xyz = bounds[:, 0]
    max_xyz = bounds[:, 1]
    bounds = max_xyz[:, None] - min_xyz[:, None]
    grid_coords = (pts - min_xyz[:, None]) / bounds
    grid_coords = grid_coords * 2 - 1
    # convert xyz to zyx, since the blend weight is indexed by xyz
    grid_coords = grid_coords.flip(-1)

    # the blend weight is indexed by xyz
    uv = uv.permute(0, 4, 1, 2, 3)
    grid_coords = grid_coords[:, None, None]
    uv = F.grid_sample(uv,
                       grid_coords,
                       mode=mode,
                       padding_mode='border',
                       align_corners=True)
    uv = uv[:, :, 0, 0]

    return uv


def pts_sample_xxx(pts, xxx, bounds, mode):
    """sample something for points
    pts: n_batch, n_points, 3
    xxx: n_batch, d, h, w, x
    bounds: n_batch, 2, 3
    """
    pts = pts.clone()
    breakpoint()

    # interpolate blend weights
    min_xyz = bounds[:, 0]
    max_xyz = bounds[:, 1]
    bounds = max_xyz[:, None] - min_xyz[:, None]

    flag_min = torch.all(pts > min_xyz[:, None, :], dim=-1)
    flag_max = torch.all(pts < max_xyz[:, None, :], dim=-1)
    flag = flag_min & flag_max
    flag = flag

    grid_coords = (pts - min_xyz[:, None]) / bounds
    grid_coords = grid_coords * 2 - 1
    # convert xyz to zyx, since the blend weight is indexed by xyz
    grid_coords = grid_coords.flip(-1)

    # the blend weight is indexed by xyz
    xxx = xxx.permute(0, 4, 1, 2, 3)
    grid_coords = grid_coords[:, None, None]
    xxx = F.grid_sample(xxx,
                        grid_coords,
                        mode=mode,
                        padding_mode='border',
                        align_corners=True)
    xxx = xxx[:, :, 0, 0]

    return xxx, flag


def grid_sample_A_blend_weights(nf_grid_coords, bw):
    """
    nf_grid_coords: batch_size x N_samples x 24 x 3
    bw: batch_size x 24 x 64 x 64 x 64
    """
    bws = []
    for i in range(24):
        nf_grid_coords_ = nf_grid_coords[:, :, i]
        nf_grid_coords_ = nf_grid_coords_[:, None, None]
        bw_ = F.grid_sample(bw[:, i:i + 1],
                            nf_grid_coords_,
                            padding_mode='border',
                            align_corners=True)
        bw_ = bw_[:, :, 0, 0]
        bws.append(bw_)
    bw = torch.cat(bws, dim=1)
    return bw


def get_sampling_points(bounds, N_samples):
    sh = bounds.shape
    min_xyz = bounds[:, 0]
    max_xyz = bounds[:, 1]
    x_vals = torch.rand([sh[0], N_samples])
    y_vals = torch.rand([sh[0], N_samples])
    z_vals = torch.rand([sh[0], N_samples])
    vals = torch.stack([x_vals, y_vals, z_vals], dim=2)
    vals = vals.to(bounds.device)
    pts = (max_xyz - min_xyz)[:, None] * vals + min_xyz[:, None]
    return pts


def get_smpl_faces(pkl_path):
    import pickle

    def read_pickle(pkl_path):
        with open(pkl_path, 'rb') as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            return u.load()
    smpl = read_pickle(pkl_path)
    faces = smpl['f']
    return faces


def load_obj(path):
    model = {}
    pts = []
    tex = []
    faces = []

    with open(path) as file:
        while True:
            line = file.readline()
            if not line:
                break
            strs = line.split(" ")
            if strs[0] == "v":
                pts.append((float(strs[1]), float(strs[2]), float(strs[3])))
            if strs[0] == "vt":
                tex.append((float(strs[1]), float(strs[2])))

    uv = np.zeros([len(pts), 2], dtype=np.float32)
    with open(path) as file:
        while True:
            line = file.readline()
            if not line:
                break
            strs = line.split(" ")
            if strs[0] == "f":
                face = (int(strs[1].split("/")[0]) - 1,
                        int(strs[2].split("/")[0]) - 1,
                        int(strs[4].split("/")[0]) - 1)
                texcoord = (int(strs[1].split("/")[1]) - 1,
                            int(strs[2].split("/")[1]) - 1,
                            int(strs[4].split("/")[1]) - 1)
                faces.append(face)
                for i in range(3):
                    uv[face[i]] = tex[texcoord[i]]

        model['pts'] = np.array(pts)
        model['faces'] = np.array(faces)
        model['uv'] = uv

    return model


def point_project_to_triangle(points: torch.Tensor, triangle: torch.Tensor) -> torch.Tensor:
    """
    Compute the point projected to triangle
    points: Float tensor of shape (n_points, 3)
    triangle: Float tensor of shape (n_points, 3, 3)
    """
    a, b, c = triangle.unbind(dim=1)
    cross = torch.cross(b - a, c - a)
    normal = F.normalize(cross, dim=-1)
    tt = normal.dot(a) - normal.dot(points)
    p0 = points + tt * normal
    return p0


def point_to_bary(point: torch.Tensor, tri: torch.Tensor) -> torch.Tensor:
    """
    Computes the barycentric coordinates of point wrt triangle (tri)
    Note that point needs to live in the space spanned by tri = (a, b, c),
    i.e. by taking the projection of an arbitrary point on the space spanned by tri
    Args:
        point: FloatTensor of shape (N, 3)
        tri: FloatTensor of shape (N, 3, 3)
    Returns:
        bary: FloatTensor of shape (N, 3)
    """
    assert point.dim() == 2 and point.shape[1] == 3
    assert tri.dim() == 3 and tri.shape[1] == 3 and tri.shape[2] == 3
    assert point.shape[0] == tri.shape[0]

    a, b, c = tri.unbind(1)

    v0 = b - a
    v1 = c - a
    v2 = point - a

    d00 = v0.dot(v0)
    d01 = v0.dot(v1)
    d11 = v1.dot(v1)
    d20 = v2.dot(v0)
    d21 = v2.dot(v1)

    denom = d00 * d11 - d01 * d01 + 1e-6
    s2 = (d11 * d20 - d01 * d21) / denom
    s3 = (d00 * d21 - d01 * d20) / denom
    s1 = 1.0 - s2 - s3

    bary = torch.tensor([s1, s2, s3]).permute(1, 0)
    return bary


def cast_knn_points(src, ref, K=1, **kwargs):
    ret = knn_points(src.float(), ref.float(), K=K, return_nn=False, return_sorted=False, **kwargs)
    dists, idx = ret.dists, ret.idx  # returns l2 distance?
    ret = DotDict()
    ret.dists = dists.sqrt()
    ret.idx = idx
    return ret


def sample_blend_closest_points(src: torch.Tensor, ref: torch.Tensor, values: torch.Tensor, K: int = 4, eps: float = 1e-8, radius=0.075, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    # not so useful to aggregate all K points
    n_batch, n_points, _ = src.shape
    ret = cast_knn_points(src, ref, K=K, **kwargs)
    dists, vert_ids = ret.dists, ret.idx  # (n_batch, n_points, K)
    # sampled = values[vert_ids]  # (n_batch, n_points, K, D)
    weights = (-dists**2 / (2*radius**2)).exp()
    weights /= (weights.sum(dim=-1, keepdim=True) + eps)
    dists = torch.einsum('ijk,ijk->ij', dists, weights)
    if values is None:
        return dists.view(B, N1, 1)
    # sampled *= weights[..., None]  # augment weight in last dim for bones # written separatedly to avoid OOM
    # sampled = sampled.sum(dim=-2)  # sum over second to last for weighted bw
    # values = values.view(-1, values.shape[-1])  # (n, D)

    # vert_ids: B, N1, K
    # values: B, N2, V
    # we want a value to be shaped B, N1, K, V
    B, N1, K = vert_ids.shape
    B, N2, V = values.shape
    values = values.gather(dim=1, index=vert_ids.view(B, N1*K, 1).expand(-1, -1, V)).view(B, N1, K, V)  # B, N1 * K -> B, N1, K, V
    sampled = torch.einsum('ijkl,ijk->ijl', values, weights)
    return sampled.view(B, N1, V), dists.view(B, N1, 1)


def pts_knn_blend_weights(pose_pts, pverts, pbw, K=4):
    B, N, _ = pose_pts.shape
    # assert pverts.shape == (1, 6890, 3)
    # assert pbw.shape[:-1] == (1, 6890)
    sampled_bw, dist = sample_blend_closest_points(pose_pts, pverts, pbw, K=K)
    ret = torch.cat((sampled_bw, dist), dim=-1)
    return ret


def pts_knn_blend_weights_multiassign_batched(pose_pts, pose_verts, pose_bw, pts_part):
    # return (1, N, P, 25)
    B, N, D = pose_bw.shape
    assert pose_verts.shape == (1, 6890, 3)
    assert pose_bw.shape[:-1] == (1, 6890)
    P = NUM_PARTS

    pose = pose_pts.expand(P, -1, -1)
    pts = torch.zeros(P, N, 3, device=pose.device)
    pbw = torch.zeros(P, N, D, device=pose.device)
    lengths2 = torch.zeros(P, dtype=torch.long, device=pose_pts.device)
    for pid in range(P):
        part_flag = (pts_part == pid)
        lengths2[pid] = part_flag.count_nonzero()
        pts[pid, :lengths2[pid]] = pose_verts[part_flag]
        pbw[pid, :lengths2[pid]] = pose_bw[part_flag]

    sampled_bw, dist = sample_blend_closest_points(pose, pts, pbw, lengths2=lengths2)
    multi_bw = torch.cat((sampled_bw, dist), dim=-1)
    multi_bw = multi_bw.permute(1, 0, 2)[None]  # P, N, 25 -> (1, N, P, 25)
    return multi_bw


def pts_knn_blend_weights_multiassign(pose_pts, pverts, pbw, pts_part):
    B, N, _ = pose_pts.shape
    assert pverts.shape == (1, 6890, 3)
    assert pbw.shape[:-1] == (1, 6890)
    P = NUM_PARTS
    # multi_bw = torch.zeros(1, N, P, 25)
    multi_bw = []
    for pid in range(P):
        part_flag = (pts_part == pid)
        part_pts = pverts[part_flag][None]
        part_pbw = pbw[part_flag][None]
        sampled_bw, dist = sample_blend_closest_points(pose_pts, part_pts, part_pbw, K=cfg.knn_k)
        ret = torch.cat((sampled_bw, dist), dim=-1)
        # multi_bw[:, :, pid] = ret
        multi_bw.append(ret)
    multi_bw = torch.stack(multi_bw, dim=2)
    return multi_bw


def pts_knn_blend_weights_multiassign_batch(pose_pts, pose_verts, pose_bw, lengths2):
    # return (1, N, P, 25)
    P = NUM_PARTS
    pose = pose_pts.expand(P, -1, -1)

    sampled_bw, dist = sample_blend_closest_points(pose, pose_verts, pose_bw, lengths2=lengths2)
    multi_bw = torch.cat((sampled_bw, dist), dim=-1)
    multi_bw = multi_bw.permute(1, 0, 2)[None]  # P, N, 25 -> (1, N, P, 25)
    return multi_bw
