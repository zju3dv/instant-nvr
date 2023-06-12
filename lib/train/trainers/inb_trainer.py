import torch.nn as nn
from lib.config import cfg
import torch
import torchvision.models.vgg as vgg
from collections import namedtuple
from lib.networks.renderer import inb_renderer
from lib.train import make_optimizer
from . import crit
from lib.utils.if_nerf import if_nerf_net_utils
import numpy as np
import torch.nn.functional as F
from lib.utils.loss_utils import SSIM
from lib.train.trainers.loss.fourier_loss import FourierLoss
from lib.train.trainers.loss.perceptual_loss import PerceptualLoss
from lib.train.trainers.loss.tv_image_loss import TVImageLoss
from termcolor import cprint


class NetworkWrapper(nn.Module):
    def __init__(self, net):
        super(NetworkWrapper, self).__init__()
        self.net = net
        self.renderer = inb_renderer.Renderer(self.net)

        self.bw_crit = torch.nn.functional.smooth_l1_loss
        self.img2mse = lambda x, y: torch.mean((x - y)**2)

        if cfg.use_lpips:
            self.perceptual_loss = PerceptualLoss()

        if cfg.use_ssim:
            self.ssim_loss = SSIM(window_size=11)

        if cfg.use_fourier:
            self.fourier_loss = FourierLoss()

        if cfg.use_tv_image:
            self.tv_image_loss = TVImageLoss()

    def forward(self, batch, epoch=-1, split='train'):
        ret = self.renderer.render(batch, test=False, epoch=epoch)
        scalar_stats = {}
        loss = torch.tensor(0.0, device=batch['latent_index'].device)

        if 'oresd' in ret and ret['oresd'].numel():
            oresd = crit.reg_raw_crit(ret['oresd'])  # svd of jacobian elastic loss
            scalar_stats.update({'pair_loss': oresd})
            loss += cfg.pair_loss_weight * oresd

        if 'ovelo' in ret and ret['ovelo'].numel():
            ovelo = crit.reg_raw_crit(ret['ovelo'])  # length of difference in value of neighbor points
            scalar_stats.update({'ovelo': ovelo})
            loss += cfg.pair_loss_weight * ovelo

        if 'reg' in ret:
            reg = ret['reg']
            reg_loss = torch.mean(reg)
            loss += cfg.reg_loss_weight * reg_loss
            scalar_stats.update({"reg": reg_loss})

        if "tv" in ret:
            tv = ret['tv']
            tv_loss = torch.mean(tv)
            loss += tv_loss
            scalar_stats.update({'tv': tv_loss})

        if "opl" in ret:
            opl_loss = torch.mean(ret['opl'])
            loss += opl_loss
            scalar_stats.update({'opl': opl_loss})

        if "freespace_occupancy" in ret:
            freespace_occupancy = ret['freespace_occupancy']
            label = torch.zeros_like(freespace_occupancy)
            free_loss = F.binary_cross_entropy(freespace_occupancy, label)
            scalar_stats.update({"free_loss": free_loss})
            loss += cfg.free_loss_weight * free_loss

        if "obj_occupancy" in ret:
            obj_occupancy = ret['obj_occupancy']
            n_batch, n_pixel = obj_occupancy.shape[:2]
            obj_occupancy = obj_occupancy[obj_occupancy < 0.5]
            if len(obj_occupancy) > 0:
                label = torch.ones_like(obj_occupancy)
                occ_loss = F.binary_cross_entropy(obj_occupancy, label)
                occ_loss = occ_loss / (n_batch * n_pixel)
                scalar_stats.update({"occ_loss": occ_loss})
                loss += cfg.occ_loss_weight * occ_loss

        if 'reg_distortion_loss' in ret:
            reg_distortion_loss = ret['reg_distortion_loss'].mean()
            scalar_stats.update({"reg_dist": reg_distortion_loss})
            loss += cfg.reg_dist_weight * reg_distortion_loss

        if 'resd' in ret:
            offset_loss = torch.norm(ret['resd'], dim=2).mean()
            scalar_stats.update({'offset_loss': offset_loss})
            loss += cfg.resd_loss_weight * offset_loss

        if 'rgb_res' in ret:
            rgb_resd_loss = torch.norm(ret['rgb_res'], dim=2).mean()
            scalar_stats.update({'rgb_resd_loss': rgb_resd_loss})
            loss += cfg.rgb_resd_loss_coe * rgb_resd_loss

        if 'fw_resd' in ret:
            resd = ret['fw_resd'] + ret['bw_resd']
            fwresd_loss = torch.norm(resd, dim=2).mean()
            scalar_stats.update({'fwresd_loss': fwresd_loss})
            loss += fwresd_loss

        if 'pred_pbw' in ret:
            bw_loss = (ret['pred_pbw'] - ret['smpl_tbw']).pow(2).mean()
            scalar_stats.update({'tbw_loss': bw_loss})
            loss += bw_loss

        if batch['latent_index'].item() < cfg.num_trained_mask and 'msk_sdf' in ret:
            mask_loss = crit.sdf_mask_crit(ret, batch)
            scalar_stats.update({'mask_loss': mask_loss})
            loss += mask_loss

        if 'surf_normal' in ret:
            normal_loss = crit.normal_crit(ret, batch)
            scalar_stats.update({'normal_loss': normal_loss})
            loss += 0.01 * normal_loss

        if 'gradients' in ret:
            gradients = ret['gradients']
            grad_loss = (torch.norm(gradients, dim=2) - 1.0)**2
            grad_loss = grad_loss.mean()
            scalar_stats.update({'grad_loss': grad_loss})
            loss += 0.1 * grad_loss

        if 'observed_gradients' in ret:
            ogradients = ret['observed_gradients']
            ograd_loss = (torch.norm(ogradients, dim=2) - 1.0)**2
            ograd_loss = ograd_loss.mean()
            scalar_stats.update({'ograd_loss': ograd_loss})
            loss += 0.1 * ograd_loss

        if 'resd_jacobian' in ret:
            elas_loss = crit.elastic_crit(ret, batch)
            scalar_stats.update({'elas_loss': elas_loss})
            loss += 0.1 * elas_loss

        if 'pbw' in ret:
            bw_loss = self.bw_crit(ret['pbw'], ret['tbw'])
            scalar_stats.update({'bw_loss': bw_loss})
            loss += bw_loss

        # 这里其实不应该mask?
        # mask = batch['mask_at_box']
        # img_loss = self.img2mse(ret['rgb_map'][mask], batch['rgb'][mask])

        if split == 'val':
            rgb_pred = ret['rgb_map'][0].detach().cpu()
            rgb_gt = batch['rgb'][0].detach().cpu()

            # img_loss = torch.mean((rgb_pred - rgb_gt)**2)
            # psnr = -10 * np.log(img_loss.item()) / np.log(10)
            # scalar_stats.update({'img_loss': img_loss, 'psnr': torch.Tensor([psnr])})
            # loss += img_loss

            mask_at_box = batch['mask_at_box'][0].detach().cpu()
            H, W = batch['H'].item(), batch['W'].item()
            mask_at_box = mask_at_box.reshape(H, W)

            img_pred = torch.zeros((H, W, 3))
            img_pred[mask_at_box] = rgb_pred

            img_gt = torch.zeros((H, W, 3))
            img_gt[mask_at_box] = rgb_gt

            error_map = torch.abs(img_pred - img_gt).sum(dim=-1)

            scalar_stats.update({'loss': loss})
            image_stats = {
                "img_gt": img_gt,
                "img_pred": img_pred,
                "error_map": error_map
            }
        elif split == 'train':
            img_loss = self.img2mse(ret['rgb_map'], batch['rgb'])
            err = (torch.abs(ret['rgb_map'] - batch['rgb']).sum(dim=-1)).detach().cpu()
            psnr = -10 * np.log(img_loss.item()) / np.log(10)
            scalar_stats.update({'img_loss': img_loss, 'psnr': torch.Tensor([psnr])})
            breakpoint()

            if cfg.use_lpips or cfg.use_ssim or cfg.use_fourier or cfg.use_tv_image:
                H, W = batch['H'].item(), batch['W'].item()
                rgb_pred = ret['rgb_map']
                rgb_gt = batch['rgb']
                occ_gt = batch['occupancy']

                breakpoint()

                mask_at_box = batch['mask_at_box'][0].detach().cpu()
                H, W = batch['H'].item(), batch['W'].item()
                mask_at_box = mask_at_box.reshape(H, W)

                img_pred = torch.zeros((H, W, 3), device=rgb_pred.device)
                img_pred[mask_at_box] = rgb_pred

                img_gt = torch.zeros((H, W, 3), device=rgb_pred.device)
                img_gt[mask_at_box] = rgb_gt

                mask_gt = torch.zeros((H, W), device=rgb_pred.device, dtype=torch.bool)
                mask_gt[mask_at_box] = occ_gt.bool()

                breakpoint()

                if cfg.use_lpips:
                    img_lpips_loss = self.perceptual_loss(img_pred.permute(2, 0, 1)[None], img_gt.permute(2, 0, 1)[None])
                    scalar_stats.update({'lpips_loss': img_lpips_loss})
                    loss += img_lpips_loss
                elif cfg.use_ssim:
                    img_ssim_loss = 1 - self.ssim_loss(img_pred.permute(2, 0, 1)[None], img_gt.permute(2, 0, 1)[None])
                    scalar_stats.update({'ssim_loss': img_ssim_loss})
                    loss += 0.1 * img_ssim_loss + img_loss
                elif cfg.use_fourier:
                    img_fourier_loss = self.fourier_loss(img_pred, img_gt)
                    scalar_stats.update({'fourier_loss': img_fourier_loss})
                    loss += 0.1 * img_fourier_loss + img_loss
                elif cfg.use_tv_image:
                    img_tv_loss = self.tv_image_loss(img_pred, img_gt, mask_gt)
                    scalar_stats.update({'tv_loss': img_tv_loss})
                    loss += 0.01 * img_tv_loss + img_loss
                else:
                    raise NotImplementedError
            else:
                loss += img_loss

            if 'rgb0' in ret:
                img_loss0 = self.img2mse(ret['rgb0'], batch['rgb'])
                scalar_stats.update({'img_loss0': img_loss0})
                loss += img_loss0

            scalar_stats.update({'loss': loss})
            image_stats = {}

            ret.update({"error": err})
        else:
            raise NotImplementedError

        # if any([v.isnan().any() for v in scalar_stats.values()]):
        #     cprint("Warning: ", color='yellow', attrs=['bold', 'blink'], end='')
        #     cprint("forward pass produced nan, pls debug. I'll retry the encoding", color='red')

        return ret, loss, scalar_stats, image_stats
