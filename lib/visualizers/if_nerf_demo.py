import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import brier_score_loss
from lib.config import cfg
import cv2
import os
from termcolor import colored
from lib.utils.base_utils import create_link, get_time
import os.path as osp
from pathlib import Path


class Visualizer:
    def __init__(self, name=None):
        if name is None:
            name = get_time()
        self.name = name
        self.result_dir = osp.join(cfg.result_dir, name)
        Path(self.result_dir).mkdir(exist_ok=True, parents=True)
        print(
            colored('the results are saved at {}'.format(self.result_dir),
                    'yellow'))

    def increase_brightness(self, img, value=30.0 / 255.):
        hsv = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        v += value
        v[v > 1.0] = 1.0

        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return img.astype(np.float64)

    def visualize(self, output, batch, split='vis'):
        rgb_pred = output['rgb_map'][0].detach().cpu().numpy()

        mask_at_box = batch['mask_at_box'][0].detach().cpu().numpy()
        H, W = batch['H'].item(), batch['W'].item()
        mask_at_box = mask_at_box.reshape(H, W)

        img_pred = np.zeros((H, W, 3))
        if cfg.white_bkgd:
            img_pred = img_pred + 1
        img_pred[mask_at_box] = rgb_pred
        img_pred = img_pred[..., [2, 1, 0]]
        breakpoint()
        if cfg.add_brightness:
            img_pred = self.increase_brightness(img_pred, value=30. / 255.)

        img_root = self.result_dir
        index = batch['view_index'].item()

        cv2.imwrite(os.path.join(img_root, '{:04d}.png'.format(index)),
                    img_pred * 255)

    def merge_into_video(self, epoch):
        name = cfg.exp_name + "_epoch" + str(epoch)
        if cfg.add_brightness:
            name += "_bright"
        cmd = "ffmpeg -r 20 -i {}/%04d.png -c:v libx264 -vf fps=20 -pix_fmt yuv420p {}.mp4".format(self.result_dir, osp.join(self.result_dir, name))
        print(cmd)
        os.system(cmd)
        cmd2 = "ffmpeg -r 20 -i {}/%04d.png {}.gif".format(self.result_dir, osp.join(self.result_dir, name))
        print(cmd2)
        os.system(cmd2)