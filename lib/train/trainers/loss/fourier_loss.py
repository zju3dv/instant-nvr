from turtle import forward
import torch.nn as nn
import torch

class FourierLoss(torch.nn.Module):
    """
    """
    def __init__(self):
        super().__init__()

    def fft(self, img_spa):
        img_freq = torch.fft.fft2(img_spa)
        img_amp = img_freq.abs()
        img_angle = img_freq.angle()
        return img_amp, img_angle, img_freq

    def compute_channel(self, gt, pred):
        gt_amp, gt_angle, gt_freq = self.fft(gt)
        pred_amp, pred_angle, pred_freq = self.fft(pred)
        amp_loss = torch.abs(gt_amp - pred_amp).mean()
        angle_loss = torch.abs(gt_angle - pred_angle).mean()
        return amp_loss + angle_loss

    def forward(self, gt, pred):
        breakpoint()
        H, W, C = gt.shape[-3:]
        # assert pred.shape[-3:] == [H, W, C]

        loss = 0.0

        for c in range(C):
            gt_channel = gt[..., c]
            pred_channel = pred[..., c]
            loss += self.compute_channel(gt_channel, pred_channel)
        
        return loss / C
