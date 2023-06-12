import torch
import torch.nn as nn
import torch.nn.functional as F

class TVImageLoss(nn.Module):
    """
    """
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, img_pred, img_gt, mask):
        # eps = 1e-1
        # eps = 0.0
        diff_x_gt = torch.square(img_gt[:-1, :, :] - img_gt[1:, :, :])
        diff_y_gt = torch.square(img_gt[:, :-1, :] - img_gt[:, 1:, :])
        eps_x = diff_x_gt.max()
        eps_y = diff_y_gt.max()
        diff_x = F.relu(torch.square(img_pred[:-1, :, :] - img_pred[1:, :, :]) - eps_x)[mask[:-1, :]].mean()
        diff_y = F.relu(torch.square(img_pred[:, :-1, :] - img_pred[:, 1:, :]) - eps_y)[mask[:, :-1]].mean()
        loss = (diff_x + diff_y) / 2.0
        return loss