import torch
import os
from torch import nn
import numpy as np
import torch.nn.functional
from collections import OrderedDict
from termcolor import colored

from functools import lru_cache


def render_weights(alpha: torch.Tensor, epsilon=1e-10):
    # alpha: n_batch, n_rays, n_samples
    weights = alpha * torch.cumprod(torch.cat([alpha.new_ones(*alpha.shape[:2], 1), 1.-alpha + epsilon], dim=-1), dim=-1)[..., :-1]  # (n_batch, n_rays, n_samples)
    return weights


def volume_rendering(rgb, alpha, epsilon=1e-8, bg_brightness=None, bg_image=None, use_random_bg=False):
    # NOTE: here alpha's last dim is not 1, but n_samples
    # rgb: n_batch, n_rays, n_samples, 3
    # alpha: n_batch, n_rays, n_samples
    # bg_image: n_batch, n_rays, 3 or None, if this is given as not None, the last sample on the ray will be replaced by this value (assuming this lies on the background)
    # We need to assume:
    # 1. network will find the True geometry, thus giving the background image its real value
    # 2. background image is rendered in a non-static fasion
    # returns:
    # weights: n_batch, n_rays, n_samples
    # rgb_map: n_batch, n_rays, 3
    # acc_map: n_batch, n_rays,

    if bg_image is not None:
        rgb[:, :, -1] = bg_image

    if use_random_bg:
        bg_brightness = torch.rand_like(rgb_map)

    weights = render_weights(alpha, epsilon)  # (n_batch, n_rays, n_samples)
    rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)  # (n_batch, n_rays, 3)
    acc_map = torch.sum(weights, -1)  # (n_batch, n_rays)

    if bg_brightness is not None:
        rgb_map = rgb_map + (1. - acc_map[..., None]) * bg_brightness

    return weights, rgb_map, acc_map


def get_aspect_bounds(bounds):
    # bounds: B, 2, 3
    half_edge = (bounds[:, 1:] - bounds[:, :1]) / 2  # 1, 1, 3
    half_long_edge = half_edge.max(dim=-1, keepdim=True)[0].expand(-1, -1, 3)
    middle_point = half_edge + bounds[:, :1]  # 1, 1, 3
    return torch.cat([middle_point - half_long_edge, middle_point + half_long_edge], dim=-2)


@lru_cache
def get_ndc_transform(bounds: torch.Tensor, preserve_aspect_ratio: bool = False):
    if preserve_aspect_ratio:
        bounds = get_aspect_bounds(bounds)
    n_batch = bounds.shape[0]

    # move to -1
    # scale to 1
    # scale * 2
    # move - 1

    move0 = torch.eye(4, device=bounds.device)[None].expand(n_batch, -1, -1)
    move0[:, :3, -1] = -bounds[:, :1]

    scale0 = torch.eye(4, device=bounds.device)[None].expand(n_batch, -1, -1)
    scale0[:, torch.arange(3), torch.arange(3)] = 1 / (bounds[:, 1:] - bounds[:, :1])

    scale1 = torch.eye(4, device=bounds.device)[None].expand(n_batch, -1, -1)
    scale1[:, torch.arange(3), torch.arange(3)] = 2

    move1 = torch.eye(4, device=bounds.device)[None].expand(n_batch, -1, -1)
    move1[:, :3, -1] = -1

    M = move1.matmul(scale1.matmul(scale0.matmul(move0)))

    return M  # only scale and translation has value


@lru_cache
def get_inv_ndc_transform(bounds: torch.Tensor, preserve_aspect_ratio: bool = False):
    M = get_ndc_transform(bounds, preserve_aspect_ratio)
    invM = scale_trans_inverse(M)
    return invM


@lru_cache
def get_dir_ndc_transform(bounds: torch.Tensor, preserve_aspect_ratio: bool = False):
    # https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/geometry/transforming-normals
    invM = get_inv_ndc_transform(bounds, preserve_aspect_ratio)
    return invM.mT


def scale_trans_inverse(M: torch.Tensor):
    n_batch = M.shape[0]
    invS = 1 / M[:, torch.arange(3), torch.arange(3)]
    invT = -M[:, :3, 3:] * invS[..., None]
    invM = torch.eye(4, device=M.device)[None].expand(n_batch, -1, -1)
    invM[:, torch.arange(3), torch.arange(3)] = invS
    invM[:, :3, 3:] = invT

    return invM


def affine_inverse(M: torch.Tensor):
    n_batch = M.shape[0]
    invR = M[:, :3, :3].mT
    invT = -invR.matmul(M[:, :3, 3:])
    invM = torch.eye(4, device=M.device)[None].expand(n_batch, -1, -1)
    invM[:, :3, :3] = invR
    invM[:, :3, 3:] = invT

    return invM


def ndc(pts, bounds, preserve_aspect_ratio=False) -> torch.Tensor:
    # both with batch dimension
    # pts has no last dimension
    M = get_ndc_transform(bounds, preserve_aspect_ratio)
    R = M[:, :3, :3]
    T = M[:, :3, 3:]
    pts = pts.matmul(R.mT) + T.mT
    return pts


def inv_ndc(pts, bounds, preserve_aspect_ratio=False) -> torch.Tensor:
    M = get_inv_ndc_transform(bounds, preserve_aspect_ratio)
    R = M[:, :3, :3]
    T = M[:, :3, 3:]
    pts = pts.matmul(R.mT) + T.mT
    return pts


def dir_ndc(dir, bounds, preserve_aspect_ratio=False) -> torch.Tensor:
    M = get_dir_ndc_transform(bounds, preserve_aspect_ratio)
    R = M[:, :3, :3]
    dir = dir.matmul(R.mT)
    return dir


def sigmoid(x):
    y = torch.clamp(x.sigmoid(), min=1e-4, max=1 - 1e-4)
    return y


def _neg_loss(pred, gt):
    ''' Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
        Arguments:
            pred (batch x c x h x w)
            gt_regr (batch x c x h x w)
    '''
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred,
                                               2) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


class FocalLoss(nn.Module):
    '''nn.Module warpper for focal loss'''

    def __init__(self):
        super(FocalLoss, self).__init__()
        self.neg_loss = _neg_loss

    def forward(self, out, target):
        return self.neg_loss(out, target)


def smooth_l1_loss(vertex_pred,
                   vertex_targets,
                   vertex_weights,
                   sigma=1.0,
                   normalize=True,
                   reduce=True):
    """
    :param vertex_pred:     [b, vn*2, h, w]
    :param vertex_targets:  [b, vn*2, h, w]
    :param vertex_weights:  [b, 1, h, w]
    :param sigma:
    :param normalize:
    :param reduce:
    :return:
    """
    b, ver_dim, _, _ = vertex_pred.shape
    sigma_2 = sigma**2
    vertex_diff = vertex_pred - vertex_targets
    diff = vertex_weights * vertex_diff
    abs_diff = torch.abs(diff)
    smoothL1_sign = (abs_diff < 1. / sigma_2).detach().float()
    in_loss = torch.pow(diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
        + (abs_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)

    if normalize:
        in_loss = torch.sum(in_loss.view(b, -1), 1) / (
            ver_dim * torch.sum(vertex_weights.view(b, -1), 1) + 1e-3)

    if reduce:
        in_loss = torch.mean(in_loss)

    return in_loss


class SmoothL1Loss(nn.Module):
    def __init__(self):
        super(SmoothL1Loss, self).__init__()
        self.smooth_l1_loss = smooth_l1_loss

    def forward(self,
                preds,
                targets,
                weights,
                sigma=1.0,
                normalize=True,
                reduce=True):
        return self.smooth_l1_loss(preds, targets, weights, sigma, normalize,
                                   reduce)


class AELoss(nn.Module):
    def __init__(self):
        super(AELoss, self).__init__()

    def forward(self, ae, ind, ind_mask):
        """
        ae: [b, 1, h, w]
        ind: [b, max_objs, max_parts]
        ind_mask: [b, max_objs, max_parts]
        obj_mask: [b, max_objs]
        """
        # first index
        b, _, h, w = ae.shape
        b, max_objs, max_parts = ind.shape
        obj_mask = torch.sum(ind_mask, dim=2) != 0

        ae = ae.view(b, h * w, 1)
        seed_ind = ind.view(b, max_objs * max_parts, 1)
        tag = ae.gather(1, seed_ind).view(b, max_objs, max_parts)

        # compute the mean
        tag_mean = tag * ind_mask
        tag_mean = tag_mean.sum(2) / (ind_mask.sum(2) + 1e-4)

        # pull ae of the same object to their mean
        pull_dist = (tag - tag_mean.unsqueeze(2)).pow(2) * ind_mask
        obj_num = obj_mask.sum(dim=1).float()
        pull = (pull_dist.sum(dim=(1, 2)) / (obj_num + 1e-4)).sum()
        pull /= b

        # push away the mean of different objects
        push_dist = torch.abs(tag_mean.unsqueeze(1) - tag_mean.unsqueeze(2))
        push_dist = 1 - push_dist
        push_dist = nn.functional.relu(push_dist, inplace=True)
        obj_mask = (obj_mask.unsqueeze(1) + obj_mask.unsqueeze(2)) == 2
        push_dist = push_dist * obj_mask.float()
        push = ((push_dist.sum(dim=(1, 2)) - obj_num) /
                (obj_num * (obj_num - 1) + 1e-4)).sum()
        push /= b
        return pull, push


class PolyMatchingLoss(nn.Module):
    def __init__(self, pnum):
        super(PolyMatchingLoss, self).__init__()

        self.pnum = pnum
        batch_size = 1
        pidxall = np.zeros(shape=(batch_size, pnum, pnum), dtype=np.int32)
        for b in range(batch_size):
            for i in range(pnum):
                pidx = (np.arange(pnum) + i) % pnum
                pidxall[b, i] = pidx

        device = torch.device('cuda')
        pidxall = torch.from_numpy(
            np.reshape(pidxall, newshape=(batch_size, -1))).to(device)

        self.feature_id = pidxall.unsqueeze_(2).long().expand(
            pidxall.size(0), pidxall.size(1), 2).detach()

    def forward(self, pred, gt, loss_type="L2"):
        pnum = self.pnum
        batch_size = pred.size()[0]
        feature_id = self.feature_id.expand(batch_size,
                                            self.feature_id.size(1), 2)
        device = torch.device('cuda')

        gt_expand = torch.gather(gt, 1,
                                 feature_id).view(batch_size, pnum, pnum, 2)

        pred_expand = pred.unsqueeze(1)

        dis = pred_expand - gt_expand

        if loss_type == "L2":
            dis = (dis**2).sum(3).sqrt().sum(2)
        elif loss_type == "L1":
            dis = torch.abs(dis).sum(3).sum(2)

        min_dis, min_id = torch.min(dis, dim=1, keepdim=True)
        # print(min_id)

        # min_id = torch.from_numpy(min_id.data.cpu().numpy()).to(device)
        # min_gt_id_to_gather = min_id.unsqueeze_(2).unsqueeze_(3).long().\
        #                         expand(min_id.size(0), min_id.size(1), gt_expand.size(2), gt_expand.size(3))
        # gt_right_order = torch.gather(gt_expand, 1, min_gt_id_to_gather).view(batch_size, pnum, 2)

        return torch.mean(min_dis)


class AttentionLoss(nn.Module):
    def __init__(self, beta=4, gamma=0.5):
        super(AttentionLoss, self).__init__()

        self.beta = beta
        self.gamma = gamma

    def forward(self, pred, gt):
        num_pos = torch.sum(gt)
        num_neg = torch.sum(1 - gt)
        alpha = num_neg / (num_pos + num_neg)
        edge_beta = torch.pow(self.beta, torch.pow(1 - pred, self.gamma))
        bg_beta = torch.pow(self.beta, torch.pow(pred, self.gamma))

        loss = 0
        loss = loss - alpha * edge_beta * torch.log(pred) * gt
        loss = loss - (1 - alpha) * bg_beta * torch.log(1 - pred) * (1 - gt)
        return torch.mean(loss)


def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _tranpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


class Ind2dRegL1Loss(nn.Module):
    def __init__(self, type='l1'):
        super(Ind2dRegL1Loss, self).__init__()
        if type == 'l1':
            self.loss = torch.nn.functional.l1_loss
        elif type == 'smooth_l1':
            self.loss = torch.nn.functional.smooth_l1_loss

    def forward(self, output, target, ind, ind_mask):
        """ind: [b, max_objs, max_parts]"""
        b, max_objs, max_parts = ind.shape
        ind = ind.view(b, max_objs * max_parts)
        pred = _tranpose_and_gather_feat(output,
                                         ind).view(b, max_objs, max_parts,
                                                   output.size(1))
        mask = ind_mask.unsqueeze(3).expand_as(pred)
        loss = self.loss(pred * mask, target * mask, reduction='sum')
        loss = loss / (mask.sum() + 1e-4)
        return loss


class IndL1Loss1d(nn.Module):
    def __init__(self, type='l1'):
        super(IndL1Loss1d, self).__init__()
        if type == 'l1':
            self.loss = torch.nn.functional.l1_loss
        elif type == 'smooth_l1':
            self.loss = torch.nn.functional.smooth_l1_loss

    def forward(self, output, target, ind, weight):
        """ind: [b, n]"""
        output = _tranpose_and_gather_feat(output, ind)
        weight = weight.unsqueeze(2)
        loss = self.loss(output * weight, target * weight, reduction='sum')
        loss = loss / (weight.sum() * output.size(2) + 1e-4)
        return loss


class GeoCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(GeoCrossEntropyLoss, self).__init__()

    def forward(self, output, target, poly):
        output = torch.nn.functional.softmax(output, dim=1)
        output = torch.log(torch.clamp(output, min=1e-4))
        poly = poly.view(poly.size(0), 4, poly.size(1) // 4, 2)
        target = target[..., None, None].expand(poly.size(0), poly.size(1), 1,
                                                poly.size(3))
        target_poly = torch.gather(poly, 2, target)
        sigma = (poly[:, :, 0] - poly[:, :, 1]).pow(2).sum(2, keepdim=True)
        kernel = torch.exp(-(poly - target_poly).pow(2).sum(3) / (sigma / 3))
        loss = -(output * kernel.transpose(2, 1)).sum(1).mean()
        return loss


def load_model(net,
               optim,
               scheduler,
               recorder,
               model_dir,
               resume=True,
               epoch=-1):
    if not resume:
        print(colored('remove contents of directory %s' % model_dir, 'red'))
        os.system('rm -rf {}'.format(model_dir))

    if not os.path.exists(model_dir):
        return 0

    pths = [
        int(pth.split('.')[0]) for pth in os.listdir(model_dir)
        if pth != 'latest.pth'
    ]
    if len(pths) == 0 and 'latest.pth' not in os.listdir(model_dir):
        return 0
    if epoch == -1:
        if 'latest.pth' in os.listdir(model_dir):
            pth = 'latest'
        else:
            pth = max(pths)
    else:
        pth = epoch
    print('load model: {}'.format(os.path.join(model_dir,
                                               '{}.pth'.format(pth))))
    pretrained_model = torch.load(
        os.path.join(model_dir, '{}.pth'.format(pth)), 'cpu')
    net.load_state_dict(pretrained_model['net'])
    optim.load_state_dict(pretrained_model['optim'])
    scheduler.load_state_dict(pretrained_model['scheduler'])
    recorder.load_state_dict(pretrained_model['recorder'])
    return pretrained_model['epoch'] + 1


def save_model(net, optim, scheduler, recorder, model_dir, epoch, last=False):
    os.system('mkdir -p {}'.format(model_dir))
    model = {
        'net': net.state_dict(),
        'optim': optim.state_dict(),
        'scheduler': scheduler.state_dict(),
        'recorder': recorder.state_dict(),
        'epoch': epoch
    }
    if last:
        torch.save(model, os.path.join(model_dir, 'latest.pth'))
    else:
        torch.save(model, os.path.join(model_dir, '{}.pth'.format(epoch)))

    # remove previous pretrained model if the number of models is too big
    pths = [
        int(pth.split('.')[0]) for pth in os.listdir(model_dir)
        if pth != 'latest.pth'
    ]
    if len(pths) <= 20:
        return
    os.system('rm {}'.format(
        os.path.join(model_dir, '{}.pth'.format(min(pths)))))


def save_part_network(net, model_dir, name):
    model_path = os.path.join(model_dir, name + ".pth")
    print("export model: {}".format(model_path))
    model = net.state_dict()
    torch.save(model, model_path)


def load_part_network(net, model_path):
    print("load model: {}".format(model_path))
    model = torch.load(model_path)
    net.load_state_dict(model, strict=True)


def load_network(net, model_dir, resume=True, epoch=-1, strict=True):
    if not resume:
        return 0

    if not os.path.exists(model_dir):
        print(colored('pretrained model does not exist', 'red'))
        return 0

    if os.path.isdir(model_dir):
        pths = [
            int(pth.split('.')[0]) for pth in os.listdir(model_dir)
            if pth != 'latest.pth'
        ]
        if len(pths) == 0 and 'latest.pth' not in os.listdir(model_dir):
            return 0
        if epoch == -1:
            if 'latest.pth' in os.listdir(model_dir):
                pth = 'latest'
            else:
                pth = max(pths)
        else:
            pth = epoch
        model_path = os.path.join(model_dir, '{}.pth'.format(pth))
    else:
        model_path = model_dir

    print('load model: {}'.format(model_path))
    pretrained_model = torch.load(model_path)
    net.load_state_dict(pretrained_model['net'], strict=strict)
    return pretrained_model['epoch'] + 1


def remove_net_prefix(net, prefix):
    net_ = OrderedDict()
    for k in net.keys():
        if k.startswith(prefix):
            net_[k[len(prefix):]] = net[k]
        else:
            net_[k] = net[k]
    return net_


def add_net_prefix(net, prefix):
    net_ = OrderedDict()
    for k in net.keys():
        net_[prefix + k] = net[k]
    return net_


def replace_net_prefix(net, orig_prefix, prefix):
    net_ = OrderedDict()
    for k in net.keys():
        if k.startswith(orig_prefix):
            net_[prefix + k[len(orig_prefix):]] = net[k]
        else:
            net_[k] = net[k]
    return net_


def remove_net_layer(net, layers):
    keys = list(net.keys())
    for k in keys:
        for layer in layers:
            if k.startswith(layer):
                del net[k]
    return net


def requires_grad(m, req):
    for param in m.parameters():
        param.requires_grad = req
