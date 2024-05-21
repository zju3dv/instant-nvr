import torch
import torch.nn.functional as F
from lib.config import cfg
from .nerf_net_utils import *
from .. import embedder
from lib.utils.net_utils import volume_rendering
from lib.utils.blend_utils import *
from lib.networks.bw_deform.inb_part_network_multiassign import Network, compute_val_pair_around_range


class Renderer:
    def __init__(self, net: Network):
        self.net = net

    def get_wsampling_points(self, ray_o, ray_d, near, far):
        # calculate the steps for each ray
        t_vals = torch.linspace(0., 1., steps=cfg.N_samples, device=near.device, dtype=near.dtype)
        z_vals = near[..., None] * (1. - t_vals) + far[..., None] * t_vals

        if cfg.perturb > 0. and self.net.training:
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape, device=upper.device, dtype=upper.dtype)
            z_vals = lower + (upper - lower) * t_rand

        pts = ray_o[:, :, None] + ray_d[:, :, None] * z_vals[..., None]

        return pts, z_vals

    def get_density_color(self, wpts, viewdir, z_vals, raw_decoder):
        """
        wpts: n_batch, n_pixel, n_sample, 3
        viewdir: n_batch, n_pixel, 3
        z_vals: n_batch, n_pixel, n_sample
        """
        n_batch, n_pixel, n_sample = wpts.shape[:3]
        wpts = wpts.view(n_batch * n_pixel * n_sample, -1)
        viewdir = viewdir[:, :, None].repeat(1, 1, n_sample, 1).contiguous()
        viewdir = viewdir.view(n_batch * n_pixel * n_sample, -1)

        # calculate dists for the opacity computation
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, dists[..., -1:]], dim=2)
        dists = dists.view(n_batch * n_pixel * n_sample)

        ret = raw_decoder(wpts, viewdir, dists)

        return ret

    def get_pixel_value(self, ray_o, ray_d, near, far, occ, batch):
        n_batch = ray_o.shape[0]

        # sampling points for nerf training
        wpts, z_vals = self.get_wsampling_points(ray_o, ray_d, near, far)
        n_batch, n_pixel, n_sample = wpts.shape[:3]

        # viewing direction, ray_d has been normalized in the dataset
        viewdir = ray_d

        def raw_decoder(wpts_val, viewdir_val, dists_val): return self.net(
            wpts_val, viewdir_val, dists_val, batch)

        # compute the color and density
        ret = self.get_density_color(wpts, viewdir, z_vals, raw_decoder)

        raw = ret['raw'].reshape(n_batch, n_pixel, n_sample, 4)
        rgb = raw[..., :3]
        occ = raw[..., 3]
        weights, rgb_map, acc_map = volume_rendering(rgb, occ, bg_brightness = cfg.white_bkgd, use_random_bg = cfg.random_bg)
        weights = weights.view(-1, *weights.shape[2:])
        rgb_map = rgb_map.view(-1, *rgb_map.shape[2:])
        acc_map = acc_map.view(-1, *acc_map.shape[2:])
        z_vals = z_vals.view(-1, n_sample)

        if cfg.use_pair_reg and self.net.training:
            def reg_decoder(x): return self.net.resd(x, batch)
            tocc: torch.Tensor = ret['tocc'].view(-1)
            reg_inds = (tocc - 0.5).abs() < 0.02  # N,
            reg_inds = reg_inds.nonzero(as_tuple=True)[0][..., None].expand(-1, 3)  # will sync GPU & CPU, slow
            if reg_inds.numel():  
                tpts: torch.Tensor = ret['tpts'].view(-1, 3)  # N, 3
                resd: torch.Tensor = ret['resd'].view(-1, 3)  # N, 3
                reg_tpts = tpts.gather(dim=0, index=reg_inds)[None]  # N, 3 -> B, N, 3
                reg_resd = resd.gather(dim=0, index=reg_inds)[None]  # N, 3 -> B, N, 3
                # selection = torch.randperm(reg_tpts.shape[1])[:4096]
                # reg_tpts = reg_tpts[:, selection]
                raw_value = compute_val_pair_around_range(reg_tpts, reg_decoder, 0.01, reg_resd)  # with precomputation
                oresd = raw_value
                ret['oresd'] = oresd
            else:
                ret['oresd'] = reg_inds.new_zeros(1, 0, 3)

        if cfg.use_reg_distortion and self.net.training:
            weight_i_weight_j = weights.reshape(n_pixel, n_sample, 1) * weights.reshape(n_pixel, 1, n_sample)
            cur_z_vals = z_vals[:, :]
            next_z_vals = torch.cat([z_vals[:, 1:], z_vals[:, -1:]], dim=-1)
            midpoint = (cur_z_vals + next_z_vals) / 2
            diff_z_vals = torch.abs(midpoint.reshape(n_pixel, n_sample, 1) - midpoint.reshape(n_pixel, 1, n_sample))
            reg_distortion_loss = (weight_i_weight_j * diff_z_vals).sum(dim=-1).sum(dim=-1)
            ret.update({'reg_distortion_loss': reg_distortion_loss[None]})

        rgb_map = rgb_map.view(n_batch, n_pixel, -1)
        acc_map = acc_map.view(n_batch, n_pixel)
        # depth_map = depth_map.view(n_batch, n_pixel)

        wpts_raw = ret['raw'].view(n_batch, n_pixel, n_sample, 4)

        ret.update({
            'rgb_map': rgb_map,
            'acc_map': acc_map,
            'raw': raw.view(n_batch, -1, 4)
        })

        if cfg.use_freespace_loss and self.net.training:
            free_wpts_raw = wpts_raw[occ == 0]
            free_occ = free_wpts_raw[..., 3]
            ret.update({"freespace_occupancy": free_occ[None]})

        if cfg.use_occ_loss and self.net.training:
            obj_wpts_raw = wpts_raw[occ == 1]
            obj_occ = obj_wpts_raw[..., 3]
            if(obj_occ.shape[1] > 0 and obj_occ.shape[0] > 0):
                obj_occ = obj_occ.max(dim=1)[0][None]
            else:
                obj_occ = torch.ones((n_batch, n_pixel)).to(obj_occ)
            ret.update({"obj_occupancy": obj_occ})

        if 'sdf' in ret:
            ret.update({'sdf': ret['sdf'].view(n_batch, -1, 1)})

        if 'resd' in ret:
            resd = ret['resd'].view(n_batch, -1, 3)
            ret.update({'resd': resd})

        if 'fw_resd' in ret:
            fw_resd = ret['fw_resd'].view(n_batch, -1, 3)
            bw_resd = ret['bw_resd'].view(n_batch, -1, 3)
            ret.update({'fw_resd': fw_resd, 'bw_resd': bw_resd})

        if 'pose_pts' in ret:
            pose_pts = ret['pose_pts'].view(n_batch, -1, 3)
            pose_pts_pred = ret['pose_pts_pred'].view(n_batch, -1, 3)
            ret.update({'pose_pts': pose_pts, 'pose_pts_pred': pose_pts_pred})

        if 'pred_pbw' in ret:
            pred_pbw = ret['pred_pbw'].view(n_batch, -1, 24)
            smpl_tbw = ret['smpl_tbw'].view(n_batch, -1, 24)
            ret.update({'pred_pbw': pred_pbw, 'smpl_tbw': smpl_tbw})

        if 'pbw' in ret:
            pbw = ret['pbw'].view(n_batch, -1, 24)
            ret.update({'pbw': pbw})

        if 'tbw' in ret:
            tbw = ret['tbw'].view(n_batch, -1, 24)
            ret.update({'tbw': tbw})

        if 'gradients' in ret:
            gradients = ret['gradients'].view(n_batch, -1, 3)
            ret.update({'gradients': gradients})

        if 'observed_gradients' in ret:
            ogradients = ret['observed_gradients'].view(n_batch, -1, 3)
            ret.update({'observed_gradients': ogradients})

        if 'resd_jacobian' in ret:
            jac = ret['resd_jacobian'].view(n_batch, -1, 3, 3)
            ret.update({'resd_jacobian': jac})

        if 'sdf' in ret:
            # get pixels that outside the mask or no ray-geometry intersection
            sdf = ret['sdf'].view(n_batch, n_pixel, n_sample)
            min_sdf = sdf.min(dim=2)[0]
            free_sdf = min_sdf[occ == 0]
            free_label = torch.zeros_like(free_sdf)

            with torch.no_grad():
                intersection_mask, _ = get_intersection_mask(sdf, z_vals)
            ind = (intersection_mask == False) * (occ == 1)
            sdf = min_sdf[ind]
            label = torch.ones_like(sdf)

            sdf = torch.cat([sdf, free_sdf])
            label = torch.cat([label, free_label])
            ret.update({
                'msk_sdf': sdf.view(n_batch, -1),
                'msk_label': label.view(n_batch, -1)
            })

        # get intersected points
        if cfg.train_with_normal and 'iter_step' in batch and batch[
                'iter_step'] > 10000:
            ret = self.get_intersection_point(intersection_mask, occ, wpts,
                                              z_vals, ray_o, ray_d, ret, batch)

        if not rgb_map.requires_grad:
            ret = {k: ret[k].detach().cpu() for k in ret.keys()}

        return ret

    def render(self, batch, test=False, epoch=-1):
        ray_o = batch['ray_o']
        ray_d = batch['ray_d']
        near = batch['near']
        far = batch['far']
        occ = batch['occupancy']
        sh = ray_o.shape

        if epoch != -1:
            batch['epoch'] = epoch

        # volume rendering for each pixel
        n_batch, n_pixel = ray_o.shape[:2]
        chunk = cfg.chunk if self.net.training else cfg.render_chunk
        ret_list = []
        # print(ray_o.shape)
        # print(batch['mask_at_box'].shape)

        if chunk >= n_pixel:
            ret = self.get_pixel_value(ray_o, ray_d, near, far, occ, batch)
        else:
            for i in range(0, n_pixel, chunk):
                ray_o_chunk = ray_o[:, i:i + chunk]
                ray_d_chunk = ray_d[:, i:i + chunk]
                near_chunk = near[:, i:i + chunk]
                far_chunk = far[:, i:i + chunk]
                occ_chunk = occ[:, i:i + chunk]
                pixel_value = self.get_pixel_value(ray_o_chunk, ray_d_chunk,
                                                   near_chunk, far_chunk,
                                                   occ_chunk, batch)
                ret_list.append(pixel_value)

            keys = ret_list[0].keys()
            ret = {k: torch.cat([r[k] for r in ret_list], dim=1) for k in keys}

        return ret
