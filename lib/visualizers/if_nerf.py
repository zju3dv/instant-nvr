import matplotlib.pyplot as plt
import numpy as np
from lib.config import cfg
import os
import cv2
from termcolor import colored
import os.path as osp
from lib.utils.base_utils import create_link, get_time
from pathlib import Path


class Visualizer:
    def __init__(self, name = None):
        if name is None:
            name = get_time()
        self.name = name
        self.result_dir = osp.join(cfg.result_dir, name)
        Path(self.result_dir).mkdir(exist_ok=True, parents=True)
        print(
            colored('the results are saved at {}'.format(self.result_dir),
                    'yellow'))

    def visualize_image(self, output, batch):
        rgb_pred = output['rgb_map'][0].detach().cpu().numpy()
        rgb_gt = batch['rgb'][0].detach().cpu().numpy()
        print('mse: {}'.format(np.mean((rgb_pred - rgb_gt)**2)))

        if rgb_pred.shape == (1024, 3):
            img_pred = rgb_pred.reshape(32, 32, 3)
            img_gt = rgb_gt.reshape(32, 32, 3)
            breakpoint()
        else:
            mask_at_box = batch['mask_at_box'][0].detach().cpu().numpy()
            H, W = batch['H'].item(), batch['W'].item()
            mask_at_box = mask_at_box.reshape(H, W)

            if cfg.white_bkgd:
                img_pred = np.ones((H, W, 3))
                img_gt = np.ones((H, W, 3))
            else:
                img_pred = np.zeros((H, W, 3))
                img_gt = np.zeros((H, W, 3))

            img_pred[mask_at_box] = rgb_pred
            img_gt[mask_at_box] = rgb_gt

        result_dir = os.path.join(self.result_dir, 'comparison')
        os.system('mkdir -p {}'.format(result_dir))
        frame_index = batch['frame_index'].item()
        view_index = batch['cam_ind'].item()
        error_map = np.abs(img_pred - img_gt).sum(axis = -1)
        cv2.imwrite(
            '{}/frame{:04d}_view{:04d}.png'.format(result_dir, frame_index,
                                                   view_index),
            (img_pred[..., [2, 1, 0]] * 255))
        cv2.imwrite(
            '{}/frame{:04d}_view{:04d}_gt.png'.format(result_dir, frame_index,
                                                      view_index),
            (img_gt[..., [2, 1, 0]] * 255))
        cv2.imwrite("{}/frame{:04d}_view{:04d}_error.png".format(result_dir, frame_index, view_index), (error_map * 255).astype(np.uint8))

        # _, (ax1, ax2) = plt.subplots(1, 2)
        # ax1.imshow(img_pred)
        # ax2.imshow(img_gt)
        # plt.show()

    def visualize_normal(self, output, batch):
        mask_at_box = batch['mask_at_box'][0].detach().cpu().numpy()
        H, W = batch['H'].item(), batch['W'].item()
        mask_at_box = mask_at_box.reshape(H, W)
        surf_mask = mask_at_box.copy()
        surf_mask[mask_at_box] = output['surf_mask'][0].detach().cpu().numpy()

        normal_map = np.zeros((H, W, 3))
        normal_map[surf_mask] = output['surf_normal'][
            output['surf_mask']].detach().cpu().numpy()

        normal_map[..., 1:] = normal_map[..., 1:] * -1
        norm = np.linalg.norm(normal_map, axis=2)
        norm[norm < 1e-8] = 1e-8
        normal_map = normal_map / norm[..., None]
        normal_map = (normal_map + 1) / 2

        plt.imshow(normal_map)
        plt.show()

    def visualize_acc(self, output, batch):
        acc_pred = output['acc_map'][0].detach().cpu().numpy()

        mask_at_box = batch['mask_at_box'][0].detach().cpu().numpy()
        H, W = int(cfg.H * cfg.ratio), int(cfg.W * cfg.ratio)
        mask_at_box = mask_at_box.reshape(H, W)

        acc = np.zeros((H, W))
        acc[mask_at_box] = acc_pred

        plt.imshow(acc)
        plt.show()

        # acc_path = os.path.join(cfg.result_dir, 'acc')
        # i = batch['i'].item()
        # cam_ind = batch['cam_ind'].item()
        # acc_path = os.path.join(acc_path, '{:04d}_{:02d}.jpg'.format(i, cam_ind))
        # os.system('mkdir -p {}'.format(os.path.dirname(acc_path)))
        # plt.savefig(acc_path)

    def visualize_depth(self, output, batch):
        depth_pred = output['depth_map'][0].detach().cpu().numpy()

        mask_at_box = batch['mask_at_box'][0].detach().cpu().numpy()
        H, W = int(cfg.H * cfg.ratio), int(cfg.W * cfg.ratio)
        mask_at_box = mask_at_box.reshape(H, W)

        depth = np.zeros((H, W))
        depth[mask_at_box] = depth_pred

        plt.imshow(depth)
        plt.show()

        # depth_path = os.path.join(cfg.result_dir, 'depth')
        # i = batch['i'].item()
        # cam_ind = batch['cam_ind'].item()
        # depth_path = os.path.join(depth_path, '{:04d}_{:02d}.jpg'.format(i, cam_ind))
        # os.system('mkdir -p {}'.format(os.path.dirname(depth_path)))
        # plt.savefig(depth_path)

    def visualize(self, output, batch, split='vis'):
        if split == 'vis' or split == 'prune':
            self.visualize_image(output, batch)
            if split == 'prune':
                latest_dir = osp.join(cfg.result_dir, 'latest')
                new_link = os.path.basename(self.result_dir)
                if osp.exists(latest_dir) and osp.islink(latest_dir):
                    print("Found old latest dir link {} which link to {}, replacing it to {}".format(latest_dir, os.readlink(latest_dir), self.result_dir))
                    os.unlink(latest_dir)
                os.symlink(new_link, latest_dir)
        elif split == 'tmesh':
            breakpoint()
            target_path = os.path.join(cfg.result_dir, 'tmesh_{}.npy'.format(self.name))
            import mcubes
            import trimesh
            np.save(target_path, output['occ'])
            create_link(osp.join(cfg.result_dir, "latest.npy"), target_path)
            # saving mesh for reference.
            cube = output['occ']
            breakpoint()
            # thresh = np.median(cube[cube > 0].reshape(-1))
            # thresh = 0.05
            # ind = np.argpartition(nonz_error_map, -sample_coord_len)[-sample_coord_len:]
            N = (cube > -1).sum()
            # NN = int(N * 0.15)
            NN = int(N * 0.1)
            ccube = cube.reshape(-1)
            ind = np.argpartition(ccube, -NN)[-NN:]
            thresh = ccube[ind].min()
            thresh = 0.1

            cube = np.pad(cube, 10, mode='constant')
            verts, triangles = mcubes.marching_cubes(cube, thresh) 
            verts = (verts - 10) * cfg.voxel_size[0]
            verts = verts + batch['tbounds'][0, 0].detach().cpu().numpy()
            mesh = trimesh.Trimesh(vertices=verts, faces=triangles)
            mesh.export(os.path.join(cfg.result_dir, "tmesh_{}.ply".format(self.name)))
            print(thresh)
        elif split == 'tdmesh':
            breakpoint()
            target_path = os.path.join(cfg.result_dir, 'tpose_deform_mesh_{}_{}.npy'.format(self.name, batch['frame_dim'][0].item()))
            import mcubes
            import trimesh
            np.save(target_path, output['occ'])

            # saving mesh for reference.
            cube = output['occ']
            cube = np.pad(cube, 10, mode='constant')
            verts, triangles = mcubes.marching_cubes(cube, 0.2) 
            verts = (verts - 10) * cfg.voxel_size[0]
            verts = verts + batch['tbounds'][0, 0].detach().cpu().numpy()
            mesh = trimesh.Trimesh(vertices=verts, faces=triangles)
            mesh.export(os.path.join(cfg.result_dir, "tpose_deform_mesh_{}_{}.ply".format(self.name, batch['frame_dim'][0].item())))
        else:
            raise NotImplementedError
