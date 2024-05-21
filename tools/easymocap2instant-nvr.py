import os
import numpy as np
import argparse
import json
import cv2
from easymocap.mytools.camera_utils import read_cameras
from easymocap.bodymodel.smpl import SMPLModel
from termcolor import colored

def myprint(cmd, level):
    color = {'run': 'blue', 'info': 'green', 'warn': 'yellow', 'error': 'red'}[level]
    print(colored(cmd, color))

def log(text):
    myprint(text, 'info')

def mywarn(text):
    myprint(text, 'warn')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data", type=str)
    parser.add_argument("--model_path", default="SMPL_NEUTRAL.pkl", type=str)
    parser.add_argument("--regressor_path", default="J_regressor_body25.npy", type=str)
    parser.add_argument('--ranges', type=int, default=None, nargs=3)
    args = parser.parse_args()

    st, ed, interval = args.ranges
    data_root= args.data_root
    log(f"[INFO] Processing {data_root}...")

    origin_params_dir=os.path.join(data_root, 'smpl')
    params_dir=os.path.join(data_root, 'smpl_params')
    vertices_dir=os.path.join(data_root, 'smpl_vertices')
    if not os.path.exists(params_dir): os.mkdir(params_dir)
    if not os.path.exists(vertices_dir): os.mkdir(vertices_dir)

    cfg={
        'model_path': args.model_path,
        'regressor_path': args.regressor_path,
    }

    smpl = SMPLModel(**cfg)

    # 求 vertices, 并转换 vertices 与 params
    log("[INFO] Converting vertices and params...")
    for name in os.listdir(origin_params_dir):
        origin_params_path=os.path.join(origin_params_dir, name)
        frame=int(name.split('.')[0])
        params_path=os.path.join(params_dir, f"{frame}.npy")
        vertices_path=os.path.join(vertices_dir, f"{frame}.npy")

        with open(origin_params_path, 'r') as f:
            params = json.load(f)
            if 'annots' in params:
                params=params['annots']
            params=params[0]
        params.pop('id')
        for key, item in params.items():
            params[key]=np.array(params[key])

        vertices = smpl.vertices(params, return_tensor=False)[0]
        np.save(vertices_path, vertices)

        params['poses']=smpl.export_full_poses(params['poses'])
        np.save(params_path, params, allow_pickle=True)

    # annots
    annots_path=os.path.join(data_root, 'annots.npy')
    log("[INFO] Converting annots...")

    images_dir=os.path.join(data_root, 'images')
    image_list=[]
    for sub_dir in sorted(os.listdir(images_dir)):
        sub_dir=os.path.join(images_dir, sub_dir)
        for image_path in sorted(os.listdir(sub_dir))[st:ed:interval]:
            image_path=os.path.join(sub_dir, image_path)
            image_path=image_path.split(data_root)[-1][1:]
            image_list.append(image_path)
    num_sub_dirs=len(os.listdir(images_dir))
    num_images=len(image_list)//num_sub_dirs
    ims=[{'ims': image_list[i::num_sub_dirs]} for i in range(num_images)]

    try:
        cameras=read_cameras(data_root)
        num_cameras=len(cameras.keys())
        cameras = dict(sorted(cameras.items(), key=lambda x: int(x[0])))
        cams = {
            'D': [camera['dist'].reshape(1,-1) for camera in cameras.values()],
            'K': [camera['K'] for camera in cameras.values()],
            'R': [camera['R'] for camera in cameras.values()],
            'T': [camera['T']*1000. for camera in cameras.values()]
        }
    except:
        mywarn('[WARN] Camera info error, use default camera')
        imgname = image_list[0]
        img_path = os.path.join(data_root, imgname)
        img = cv2.imread(img_path)
        height, width = img.shape[:2]
        log('[INFO] Read image shape {}'.format(img.shape))
        focal = 1.2*min(height, width) # as colmap
        log('[INFO] Set a fix focal length {}'.format(focal))
        K = np.array([focal, 0., width/2, 0., focal, height/2, 0. ,0., 1.]).reshape(3, 3)
        camera = {'K':K ,'R': np.eye(3), 'T': np.zeros((3, 1)), 'dist': np.zeros((1, 5))}
        for key, val in camera.items():
            camera[key] = val.astype(np.float32)
        cams = {
            'D': [camera['dist'].reshape(1,-1) for _ in range(num_sub_dirs)],
            'K': [camera['K'] for _ in range(num_sub_dirs)],
            'R': [camera['R'] for _ in range(num_sub_dirs)],
            'T': [camera['T']*1000. for _ in range(num_sub_dirs)]
        }

    annots={
        'cams':cams,
        'ims':ims
    }

    np.save(annots_path, annots, allow_pickle=True)