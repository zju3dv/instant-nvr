import pickle
import os.path as osp
import os
import numpy as np
from pathlib import Path
from termcolor import colored
import torch
from typing import Mapping, TypeVar
KT = TypeVar("KT")  # key type
VT = TypeVar("VT")  # value type

def create_dir(name: os.PathLike):
    Path(name).mkdir(exist_ok=True, parents=True)

def create_link(src, tgt):
    new_link = os.path.basename(tgt)
    if osp.exists(src) and osp.islink(src):
        print("Found old latest dir link {} which link to {}, replacing it to {}".format(src, os.readlink(src), tgt))
        os.unlink(src)
    os.symlink(new_link, src)

def dump_cfg(cfg, tgt_path: os.PathLike):
    if os.path.exists(tgt_path):
        if not cfg.silent:
            print(colored("Hey, there exists an experiment with same name before. Please make sure you are continuing.", "green"))
        return
    create_dir(Path(tgt_path).parent)
    cfg_str = cfg.dump()
    with open(tgt_path, "w") as f: 
        f.write(cfg_str)

def git_committed():
    from git import Repo
    modified_index = Repo('.').index.diff(None)
    files = []
    for ind in modified_index:
        file = ind.a_path 
        if not file[-4:] == 'yaml':
            files.append(file)
    return len(files) == 0

def git_hash():
    import git
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    return sha

def get_time():
    from datetime import datetime
    now = datetime.now()
    return '_'.join(now.__str__().split(' '))

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def read_pickle(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


def save_pickle(data, pkl_path):
    os.system('mkdir -p {}'.format(os.path.dirname(pkl_path)))
    with open(pkl_path, 'wb') as f:
        pickle.dump(data, f)


def project(xyz, K, RT):
    """
    xyz: [N, 3]
    K: [3, 3]
    RT: [3, 4]
    """
    xyz = np.dot(xyz, RT[:, :3].T) + RT[:, 3:].T
    xyz = np.dot(xyz, K.T)
    xy = xyz[:, :2] / xyz[:, 2:]
    return xy

def project_torch(xyz, K, RT):
    """
    xyz: [N, 3]
    K: [3, 3]
    RT: [3, 4]
    """
    xyz = torch.mm(xyz, RT[:, :3].T) + RT[:, 3:].T
    depth = xyz[..., -1]
    xyz = torch.mm(xyz, K.T)
    xy = xyz[:, :2] / xyz[:, 2:]
    return xy, depth


def write_K_pose_inf(K, poses, img_root):
    K = K.copy()
    K[:2] = K[:2] * 8
    K_inf = os.path.join(img_root, 'Intrinsic.inf')
    os.system('mkdir -p {}'.format(os.path.dirname(K_inf)))
    with open(K_inf, 'w') as f:
        for i in range(len(poses)):
            f.write('%d\n'%i)
            f.write('%f %f %f\n %f %f %f\n %f %f %f\n' % tuple(K.reshape(9).tolist()))
            f.write('\n')

    pose_inf = os.path.join(img_root, 'CamPose.inf')
    with open(pose_inf, 'w') as f:
        for pose in poses:
            pose = np.linalg.inv(pose)
            A = pose[0:3,:]
            tmp = np.concatenate([A[0:3,2].T, A[0:3,0].T,A[0:3,1].T,A[0:3,3].T])
            f.write('%f %f %f %f %f %f %f %f %f %f %f %f\n' % tuple(tmp.tolist()))

def merge_dicts(dict_a, dict_b, b_append_key):
    dict = {}
    dict2 = {}
    for k in dict_a:
        dict.update({k: dict_a[k]})
        dict2.update({k + b_append_key: dict_b[k]})
    dict.update(dict2)
    return dict


class DotDict(dict, Mapping[KT, VT]):
    """
    a dictionary that supports dot notation 
    as well as dictionary access notation 
    usage: d = DotDict() or d = DotDict({'val1':'first'})
    set attributes: d.val2 = 'second' or d['val2'] = 'second'
    get attributes: d.val2 or d['val2']
    """

    def update(self, dct=None, **kwargs):
        if dct is None:
            dct = kwargs
        else:
            dct.update(kwargs)
        for k, v in dct.items():
            if k in self:
                target_type = type(self[k])
                if not isinstance(v, target_type):
                    # NOTE: bool('False') will be True
                    if target_type == bool and isinstance(v, str):
                        dct[k] = v == 'True'
                    else:
                        dct[k] = target_type(v)
        dict.update(self, dct)

    def __hash__(self):
        return hash(''.join([str(self.values().__hash__())]))

    def __init__(self, dct=None, **kwargs):
        if dct is None:
            dct = kwargs
        else:
            dct.update(kwargs)
        if dct is not None:
            for key, value in dct.items():
                if hasattr(value, 'keys'):
                    value = DotDict(value)
                self[key] = value

    """
    Uncomment following lines and 
    comment out __getattr__ = dict.__getitem__ to get feature:
    
    returns empty numpy array for undefined keys, so that you can easily copy things around
    TODO: potential caveat, harder to trace where this is set to np.array([], dtype=np.float32)
    """

    def __getitem__(self, key):
        try:
            return dict.__getitem__(self, key)
        except KeyError as e:
            raise AttributeError(e)
    # MARK: Might encounter exception in newer version of pytorch
    # Traceback (most recent call last):
    #   File "/home/xuzhen/miniconda3/envs/torch/lib/python3.9/multiprocessing/queues.py", line 245, in _feed
    #     obj = _ForkingPickler.dumps(obj)
    #   File "/home/xuzhen/miniconda3/envs/torch/lib/python3.9/multiprocessing/reduction.py", line 51, in dumps
    #     cls(buf, protocol).dump(obj)
    # KeyError: '__getstate__'
    # MARK: Because you allow your __getattr__() implementation to raise the wrong kind of exception.
    __getattr__ = __getitem__  # overidden dict.__getitem__
    # __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__