import os
import pprint
import argparse
import numpy as np
import colored_traceback.auto  # hook

from . import yacs
from .yacs import CfgNode as CN

cfg = CN()

cfg.part3 = False
cfg.part6 = False

cfg.aggr = ""

cfg.ps = [1, 19349663, 83492791]

cfg.ratio = 0.5
cfg.fast_eval = False
cfg.eval_ratio = -1.0
cfg.multi_stream = False
cfg.latent_code_dim = 8
cfg.geo_feature_dim = 16

cfg.dry_run = False
cfg.random_bg = False
cfg.bbox_overlap = 0.2  # 10cm overlap?
cfg.use_batch_bounds = True
cfg.render_chunk = 4096
cfg.detect_anomaly = False
cfg.use_amp = False
cfg.device_prefetch = 8
cfg.n_coarse_knn_ref = -1
cfg.lbs = 'lbs'
cfg.use_pair_reg = True

cfg.profiler = 'torch'
cfg.profiling = False
cfg.profiling_dir = 'data/record/profiling'
cfg.clear_previous_profiling = True

cfg.parent_cfg = 'configs/default.yaml'
cfg.method = ''

cfg.use_time_embedder = False
cfg.no_part = False
cfg.base_resolution = 16
cfg.base_head_resolution = 16

# experiment name
cfg.exp_name = 'hello'

# network
cfg.point_feature = 9
cfg.distributed = False
cfg.num_latent_code = -1
# cfg.train_face_highres = False
cfg.sample_focus = ""

# data
cfg.human = 313
cfg.training_view = [0, 6, 12, 18]
cfg.test_view = []
cfg.begin_ith_frame = 0  # the first smpl
cfg.num_train_frame = 1  # number of smpls
cfg.num_eval_frame = -1  # number of frames to render
cfg.ith_smpl = 0  # the i-th smpl
cfg.frame_interval = 1
cfg.smpl = 'smpl_4views_5e-4'
cfg.vertices = 'vertices'
cfg.params = 'params_4views_5e-4'
cfg.mask_bkgd = True
cfg.sample_smpl = False
cfg.sample_grid = False
cfg.sample_fg_ratio = 0.7
cfg.add_pointcloud = False
cfg.test_on_training_view = False
cfg.sample_using_mse = False
cfg.sample_mse_portion = 0.8
cfg.prune_using_geo = False
cfg.prune_geo_thresh = 0.2
cfg.prune_using_hull = False

cfg.mono_bullet = False

cfg.big_box = False
cfg.box_padding = 0.05
cfg.voxel_size = [0.005, 0.005, 0.005]

cfg.rot_ratio = 0.
cfg.rot_range = np.pi / 32

# mesh
cfg.mesh_th = 50  # threshold of alpha

# task
cfg.task = 'nerf4d'

# gpus
cfg.gpus = list(range(8))
# if load the pretrained network
cfg.pretrained_model = "none"
cfg.resume = True

# epoch
cfg.ep_iter = -1
cfg.save_ep = 100
cfg.save_latest_ep = 5
cfg.eval_ep = 100
cfg.no_save = False

cfg.no_viewdir = False
cfg.part_deform = False

# -----------------------------------------------------------------------------
# train
# -----------------------------------------------------------------------------
cfg.train = CN()

cfg.train.dataset = 'CocoTrain'
cfg.train.epoch = 10000
cfg.train.num_workers = 8
cfg.train.collator = ''
cfg.train.batch_sampler = 'default'
cfg.train.sampler_meta = CN({'min_hw': [256, 256], 'max_hw': [480, 640], 'strategy': 'range'})
cfg.train.shuffle = True

# use adam as default
cfg.train.optim = 'adam'
cfg.train.lr = 1e-4
cfg.train.weight_decay = 0.

cfg.train.scheduler = CN({'type': 'multi_step', 'milestones': [80, 120, 200, 240], 'gamma': 0.5})

cfg.train.batch_size = 4

cfg.train.acti_func = 'relu'

cfg.train.use_vgg = False
cfg.train.vgg_pretrained = ''
cfg.train.vgg_layer_name = [0, 0, 0, 0, 0]

cfg.train.use_ssim = False
cfg.train.use_d = False

# test
cfg.test = CN()
cfg.test.dataset = 'CocoVal'
cfg.test.batch_size = 1
cfg.test.epoch = -1
cfg.test.sampler = 'default'
cfg.test.batch_sampler = 'default'
cfg.test.sampler_meta = CN({'min_hw': [480, 640], 'max_hw': [480, 640], 'strategy': 'origin'})
cfg.test.frame_sampler_interval = 30
cfg.global_test_switch = False

# val
cfg.val = CN()
cfg.val.dataset = 'CocoVal'
cfg.val.batch_size = 1
cfg.val.epoch = -1
cfg.val.sampler = 'FrameSampler'
cfg.val.frame_sampler_interval = 20
cfg.val.batch_sampler = 'default'
cfg.val.sampler_meta = CN({'min_hw': [480, 640], 'max_hw': [480, 640], 'strategy': 'origin'})
cfg.val.collator = ''

# prune
cfg.prune = CN()
cfg.prune.dataset = 'CocoVal'
cfg.prune.batch_size = 1
cfg.prune.epoch = -1
cfg.prune.sampler = 'default'
cfg.prune.batch_sampler = 'default'
cfg.prune.sampler_meta = CN({'min_hw': [480, 640], 'max_hw': [480, 640], 'strategy': 'origin'})
cfg.prune.frame_sampler_interval = 1
cfg.prune.collator = ''

# extract tpose mesh
cfg.tmesh = CN()
cfg.tmesh.dataset = 'Cocotmesh'
cfg.tmesh.batch_size = 1
cfg.tmesh.epoch = -1
cfg.tmesh.sampler = 'FrameSampler'
cfg.tmesh.frame_sampler_interval = 1
cfg.tmesh.batch_sampler = 'default'
cfg.tmesh.sampler_meta = CN({'min_hw': [480, 640], 'max_hw': [480, 640], 'strategy': 'origin'})
cfg.tmesh.collator = ''

# extract deformed tpose mesh
cfg.tdmesh = CN()
cfg.tdmesh.dataset = 'Cocotmesh'
cfg.tdmesh.batch_size = 1
cfg.tdmesh.epoch = -1
cfg.tdmesh.sampler = 'FrameSampler'
cfg.tdmesh.frame_sampler_interval = 1
cfg.tdmesh.batch_sampler = 'default'
cfg.tdmesh.sampler_meta = CN({'min_hw': [480, 640], 'max_hw': [480, 640], 'strategy': 'origin'})
cfg.tdmesh.collator = ''

# bullet
cfg.bullet = CN()
cfg.bullet.dataset = 'CocoVal'
cfg.bullet.batch_size = 1
cfg.bullet.epoch = -1
cfg.bullet.sampler = 'default'
cfg.bullet.batch_sampler = 'default'
cfg.bullet.sampler_meta = CN({'min_hw': [480, 640], 'max_hw': [480, 640], 'strategy': 'origin'})
cfg.bullet.frame_sampler_interval = 1
cfg.bullet.collator = ''

# trained model
cfg.trained_model_dir = 'data/trained_model'

# recorder
cfg.record_dir = 'data/record'
cfg.log_interval = 20
cfg.record_interval = 20

# result
cfg.result_dir = 'exps'

# training
cfg.training_mode = 'default'
cfg.train_nbfusion = False
cfg.train_with_coord = False
cfg.train_init_sdf = False
cfg.train_init_bw = False
cfg.aninerf_animation = False
cfg.tpose_viewdir = True
cfg.color_with_viewdir = True
cfg.color_with_feature = False
cfg.forward_rendering = False
cfg.has_forward_resd = False
cfg.train_forward_resd = False
cfg.train_with_normal = False
cfg.tpose_geometry = True
cfg.erode_edge = True
cfg.num_trained_mask = 3
cfg.bigpose = True
cfg.use_freespace_loss = False
cfg.free_loss_weight = 0.0001
cfg.use_occ_loss = False
cfg.occ_loss_weight = 0.0001
cfg.mlp_weight_decay = 1.0
cfg.reg_loss_weight = 0.0
cfg.use_lpips = False
cfg.use_ssim = False
cfg.use_fourier = False
cfg.use_tv_image = False
cfg.patch_sampling = False
cfg.patch_size = 64
cfg.reg_dist_weight = 0.1
cfg.resd_loss_weight = 0.1
cfg.pair_loss_weight = 1e-4

cfg.use_reg_distortion = False

# evaluation
cfg.eval = False
cfg.skip_eval = False
cfg.test_novel_pose = False
cfg.novel_pose_ni = 100
cfg.vis_novel_pose = False
cfg.vis_novel_view = False
cfg.vis_tpose_mesh = False
cfg.vis_posed_mesh = False

cfg.add_brightness = False

cfg.fix_random = False

cfg.vis = 'mesh'

# data
cfg.body_sample_ratio = 0.5
cfg.face_sample_ratio = 0.

cfg.debug = False

cfg.chunk = 4096

cfg.test_all_other = False
cfg.test_full = True

cfg.semantic_dim = 20
cfg.render_frame = -1
cfg.smpl_thresh = 0.1
cfg.render_remove = ""

cfg.use_knn = True
cfg.knn_k = 4

cfg.smpl_meta = "data/smpl-meta"

cfg.eval_part = ""

cfg.pn_finetune = True

cfg.record_demo = False


def parse_cfg(cfg, args):
    if len(cfg.task) == 0:
        raise ValueError('task must be specified')

    if cfg.num_latent_code < 0:
        cfg.num_latent_code = cfg.num_train_frame

    if cfg.eval_ratio < 0:
        cfg.eval_ratio = cfg.ratio

    # assign the gpus
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = ', '.join([str(gpu) for gpu in cfg.gpus])
    cfg.result_dir = os.path.join(cfg.result_dir, cfg.task, cfg.exp_name)
    cfg.trained_model_dir = os.path.join(cfg.result_dir, 'trained_model')
    cfg.record_dir = os.path.join(cfg.result_dir, 'record')
    # cfg.profiling_dir = os.path.join(cfg.result_dir, 'prof')
    cfg.profiling_dir = os.path.join(cfg.profiling_dir, cfg.task, cfg.exp_name)

    if cfg.forward_rendering:
        cfg.result_dir = cfg.result_dir + '_fw'

    cfg.local_rank = args.local_rank
    cfg.distributed = cfg.distributed or args.launcher not in ['none']

    if cfg.debug:
        os.environ['PYTHONBREAKPOINT'] = "ipdb.set_trace"
        cfg.train.num_workers = 0
    else:
        os.environ['PYTHONBREAKPOINT'] = "0"


def make_cfg(args):
    with open(args.cfg_file, 'r') as f:
        current_cfg = yacs.load_cfg(f)

    if 'parent_cfg' in current_cfg.keys():
        with open(current_cfg.parent_cfg, 'r') as f:
            parent_cfg = yacs.load_cfg(f)
        cfg.merge_from_other_cfg(parent_cfg)

    cfg.merge_from_other_cfg(current_cfg)
    cfg.merge_from_list(args.opts)

    if cfg.train_nbfusion:
        cfg.merge_from_other_cfg(cfg.nbfusion_cfg)

    if cfg.train_init_sdf:
        cfg.merge_from_other_cfg(cfg.train_init_sdf_cfg)

    if cfg.train_init_bw:
        cfg.merge_from_other_cfg(cfg.train_init_bw_cfg)

    if cfg.train_forward_resd:
        cfg.has_forward_resd = True
        cfg.merge_from_other_cfg(cfg.train_forward_resd_cfg)

    if cfg.aninerf_animation:
        cfg.merge_from_other_cfg(cfg.aninerf_animation_cfg)

    if cfg.color_with_feature:
        cfg.merge_from_other_cfg(cfg.color_feature_cfg)

    if cfg.forward_rendering:
        cfg.has_forward_resd = True
        cfg.merge_from_other_cfg(cfg.forward_rendering_cfg)

    if cfg.vis_novel_pose:
        cfg.merge_from_other_cfg(cfg.novel_pose_cfg)

    if cfg.vis_novel_view:
        cfg.merge_from_other_cfg(cfg.novel_view_cfg)

    if cfg.vis_tpose_mesh or cfg.vis_posed_mesh:
        cfg.merge_from_other_cfg(cfg.mesh_cfg)

    cfg.merge_from_list(args.opts)

    parse_cfg(cfg, args)
    # pprint.pprint(cfg)
    return cfg


parser = argparse.ArgumentParser()
parser.add_argument("--cfg_file", default="configs/default.yaml", type=str)
parser.add_argument('--test', action='store_true', dest='test', default=False)
parser.add_argument("--type", type=str, default="vis")
parser.add_argument('--det', type=str, default='')
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--launcher', type=str, default='none', choices=['none', 'pytorch'])
parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
args = parser.parse_args()
if len(args.type) > 0:
    cfg.task = "run"
cfg = make_cfg(args)
