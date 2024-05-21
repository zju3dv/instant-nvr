import yaml
import os
import os.path as osp
import sys
import collections
import shutil
import argparse
import time
from pathlib import Path
sys.path.append(os.getcwd())

from lib.config import cfg
from lib.utils.debug_utils import run_cmd, log, mywarn, myerror, mkdir, check_exists

from ruamel.yaml import YAML
yaml = YAML()
yaml.preserve_quotes = True
yaml.indent(mapping=2, sequence=4, offset=2)


def update_dict(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update_dict(d.get(k, {}), v)
        else:
            if v is None:
                d.pop(k)
            else:
                d[k] = v
    return d

def fintune_cfg(src, dst, new_cfg:dict):
    if osp.exists(dst):
        log(f'[INFO] Using cfg {dst}')
        return

    log(f'[INFO] Making cfg {dst}')

    with open(src, 'r') as file:
        data = yaml.load(file)

    data = update_dict(data, new_cfg)

    with open(dst, 'w') as file:
        yaml.dump(data, file)

def symlink(src, dst):
    if not osp.exists(src):
        myerror(f'{src} does not exist')
        return
    if not osp.exists(dst):
        log(f'[INFO] Symlink {osp.abspath(src)} -> {osp.abspath(dst)}')
        os.symlink(osp.abspath(src), osp.abspath(dst))

def check_exists(items, path = cfg.data_root):
    if isinstance(items, str): items = [items]
    for item in items:
        if not osp.exists(osp.join(path, item)):
            return False
    return True

def easymocap():
    # First, we need to extract keypoints from the images
    easymocap_path = osp.expanduser(cfg.easymocap_path)

    # models are defined at https://chingswy.github.io/easymocap-public-doc/quickstart/keypoints.html#yolov4hrnet
    models_path = osp.join(easymocap_path, 'data/models')
    symlink(cfg.models_path, models_path)

    if not check_exists('images'):
        global ed
        assert check_exists('videos'), 'videos not found'
        cmd = f'''python3 {osp.join(easymocap_path, 'apps/preprocess/extract_image.py')} {data_root} --num {ed}'''
        run_cmd(cmd)

        imgs_num = len(os.listdir(osp.join(data_root, 'images', os.listdir(osp.join(data_root, 'images'))[0])))
        if imgs_num < ed: 
            ed = imgs_num
    else:
        log(f'[INFO] Images already exist')

    if not check_exists('annots'):
        cmd = f'''cd {easymocap_path} && python apps/preprocess/extract_keypoints.py {data_root} --mode yolo-hrnet  --gpu {" ".join([f"{i}" for i in cfg.gpus])}'''
        run_cmd(cmd)
    else:
        log(f'[INFO] Keypoints already exist')

    # Second, we need to fit SMPL model to the keypoints
    # Here, we load our SMPL model and regressor, and delete the unnecessary parts in cfg
    smpl_cfg = {
        'args':{
            'at_final':{
                'load_body_model':{
                    'args':{
                        'model_path': smpl_model_path,
                        'regressor_path': smpl_regressor_path,
                    }
                },
                'render': None,
                'make_video': None
            }
        }
    }
    easymocap_cfg_path = osp.join(cfg.result_dir, 'hrnet_pare_finetune.yml')
    easymocap_cfg_path = osp.abspath(easymocap_cfg_path)

    # models are defined at EasyMocap/myeasymocap/backbone/pare/pare.py
    # yolov5 uses torch.hub.load('ultralytics/yolov5', 'yolov5s') to load the model, so we have to connect the internet to download the model
    models_path = osp.join(easymocap_path, 'models')
    symlink(cfg.models_path, models_path)
    symlink(osp.join(cfg.models_path, 'yolov5m.pt'), osp.join(easymocap_path, 'yolov5m.pt'))

    if not check_exists('smpl'):
        fintune_cfg(osp.join(easymocap_path, 'config/1v1p/hrnet_pare_finetune.yml'), easymocap_cfg_path, smpl_cfg)

        cmd = f'''export TORCH_HOME={models_path} && cd {easymocap_path} && emc --data {osp.join(easymocap_path, 'config/datasets/svimage.yml')} --exp {easymocap_cfg_path} --ranges {st} {ed} {interval} --subs {' '.join(os.listdir(osp.join(data_root, 'images')))} --root {data_root} --out {data_root}'''
        run_cmd(cmd)
    else:
        log(f'[INFO] SMPLs already exist')


def schp():
    schp_path = osp.expanduser(cfg.schp_path)
    schp_models_path = osp.join(cfg.models_path, 'schp')
    schp_models_path = osp.abspath(schp_models_path)

    if not check_exists('schp'):
        # backup images
        if not check_exists('images.backup'):            
            log(f'[INFO] Backup images')
            shutil.copytree(osp.join(data_root, 'images'), osp.join(data_root, 'images.backup'))
        
        # remove extra images
        for sub in os.listdir(osp.join(data_root, 'images')):
            images_list = sorted(os.listdir(osp.join(data_root, 'images', sub)))
            if len(images_list) > cfg.num_train_frame:
                log('[INFO] remove extra images')
                range_images = images_list[st:ed:interval]
                other_images = [i for i in images_list if i not in range_images]
                for i in other_images:
                    os.remove(osp.join(data_root, 'images', sub, i))

        try: 
            tmp_path = osp.abspath(cfg.tmp_path)
        except:
            tmp_path = None

        if not check_exists('original_schp'):
            if not check_exists(data_root.split(os.sep)[-1], tmp_path if tmp_path else osp.join(schp_path, 'data')):
                # models are defined at https://chingswy.github.io/easymocap-public-doc/install/install_segment.html#download-the-models
                cmd = f'''python {osp.join(schp_path, 'extract_multi.py')} {data_root} --ckpt_dir {schp_models_path} --subs {' '.join(os.listdir(osp.join(data_root, 'images')))} --gpus {" ".join([f"{i}" for i in cfg.gpus])}'''
                if tmp_path:
                    cmd += f' --tmp {tmp_path}'

                mywarn('[WARN] The code of schp will generate some very big(>100G) outputs. Make sure we have enough space in `data/` or in `tmp_path`,')
                mywarn('Waiting for 5 seconds to check if we have enough space')
                time.sleep(5)
                run_cmd(cmd)
            else:
                log(f'[INFO] SCHP already run')

            log(f'[INFO] Move SCHP to {data_root.split(os.sep)[-1]}')
            shutil.move(osp.join(tmp_path if tmp_path else osp.join(schp_path, 'data'), data_root.split(os.sep)[-1]), osp.join(data_root, 'original_schp'))
        else:
            log(f'[INFO] original SCHPs already exist')
        

        shutil.copytree(osp.join(data_root, 'original_schp', 'mask-schp-parsing'), osp.join(data_root, 'schp'))
        # Crop the images to retain only the main human
        cmd = f'''python tools/cropschp.py --data_root {data_root}'''
        run_cmd(cmd)
    else:
        log(f'[INFO] SCHPs already exist')

def script():
    # convert images, cameras and SMPL to Instant-NVR format
    if not check_exists(['annots.npy', 'smpl_params', 'smpl_vertices']):
        cmd = f'''python tools/easymocap2instant-nvr.py --data_root {data_root} --model_path {smpl_model_path} --regressor_path {smpl_regressor_path} --ranges {st} {ed} {interval}'''
        run_cmd(cmd)
    else:
        log(f'[INFO] Annots already exist')
    
    # convert SMPL to SMPL lbs
    if not check_exists('smpl_lbs'):
        cmd = f'''python tools/prepare_zjumocap.py --data_root {data_root} --output_root {data_root} --smpl_model_path {smpl_model_path} --smpl_uv_path {smpl_uv_path} --ranges {st} {ed} {interval}'''
        run_cmd(cmd)
    else:
        log(f'[INFO] SMPL lbs already exist')

def making_cfg(cfg_path):
    new_cfg = {
        'task': cfg.task,
        'exp_name': cfg.exp_name,
    }

    ann_file = osp.join(data_root, 'annots.npy')

    new_cfg.update({
        f'{split}_dataset': {
            'data_root': data_root,
            'ann_file': ann_file,
        } for split in ['train', 'val', 'test']
    })

    new_cfg.update({
        'num_train_frame': cfg.num_train_frame,
        'begin_ith_frame': cfg.begin_ith_frame,
        'frame_interval': cfg.frame_interval,
        'training_view': [0], # os.listdir(osp.join(data_root, 'images')),
        'test_view':  [0], #os.listdir(osp.join(data_root, 'images')),
    })

    
    fintune_cfg(cfg.default_cfg_path, cfg_path, new_cfg)

def instant_nvr():
    making_cfg(cfg_path)
    if not check_exists('trained_model/latest.pth', Path(cfg_path).parent):
        cmd = f'''python train_net.py --cfg_file {cfg_path}'''
        run_cmd(cmd)
    else:
        log(f'[INFO] Trained model already exist')

    log(f'[INFO] Visualizing')
    cmd = f'''python run.py --type vis --cfg_file {cfg_path}'''
    run_cmd(cmd)

if __name__ == '__main__':
    cfg_path = osp.join(cfg.result_dir, f'{cfg.exp_name}.yml')
    mkdir(Path(cfg_path).parent)
    data_root = osp.abspath(cfg.data_root)
    smpl_path = osp.abspath(cfg.smpl_path)
    smpl_model_path = osp.join(smpl_path, 'SMPL_NEUTRAL.pkl')
    smpl_regressor_path = osp.join(smpl_path, 'J_regressor_body25.npy')
    smpl_uv_path = osp.join(smpl_path, 'smpl_uv.obj')

    st = cfg.begin_ith_frame
    interval = cfg.frame_interval
    ed = st + cfg.num_train_frame * interval

    start_time = time.time()

    easymocap()
    schp()
    script()

    instant_nvr()

    end_time = time.time()
    log(f'[INFO] Time elapsed: {end_time - start_time:.2f}s')