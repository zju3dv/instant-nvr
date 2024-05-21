import numpy as np
import os
import json
import cv2
import imageio
import matplotlib.pyplot as plt
import tqdm
import argparse

def myprint(cmd, level):
    color = {'run': 'blue', 'info': 'green', 'warn': 'yellow', 'error': 'red'}[level]
    print(colored(cmd, color))

def log(text):
    myprint(text, 'info')

def mywarn(text):
    myprint(text, 'warn')

def cal_Square(bbox):
    if bbox is None or bbox[4]<0.6:return 0
    return (bbox[0]-bbox[2])*(bbox[1]-bbox[3])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data", type=str)
    args = parser.parse_args()

    data_root = args.data_root

    annots_path = os.path.join(data_root, 'annots')
    schp_path = os.path.join(data_root, 'schp')

    for sub_dir in os.listdir(schp_path):
        sub_schp_path = os.path.join(schp_path, sub_dir)
        sub_annot_path = os.path.join(annots_path, sub_dir)
        for schp_img in tqdm.tqdm(sorted(os.listdir(sub_schp_path))[0:]):
            full_schp_img = os.path.join(sub_schp_path, schp_img)
            full_annot = os.path.join(sub_annot_path, schp_img.replace('.png', '.json'))

            # print(annot_path)
            with open(full_annot, 'r') as f:
                annot = json.load(f)
            # print(annot['annots'])

            max_bbox = None
            for person in annot['annots']:
                # print(person)
                # print(cal_Square(person['bbox']))
                if(cal_Square(max_bbox)<cal_Square(person['bbox'])):
                    max_bbox = person['bbox']
            max_bbox = np.array(max_bbox).astype(int)
            assert max_bbox is not None
            # print(max_bbox)

            # print(schp_path)
            img = imageio.v2.imread(full_schp_img)
            # fix bbox
            max_bbox[0] = max(max_bbox[0], 0)
            max_bbox[1] = max(max_bbox[1], 0)
            max_bbox[2] = min(max_bbox[2], img.shape[1])
            max_bbox[3] = min(max_bbox[3], img.shape[0])

            img[:max_bbox[1], :] = [0, 0, 0, 255]  # 上
            img[max_bbox[3]:, :] = [0, 0, 0, 255]  # 下
            img[:, :max_bbox[0]] = [0, 0, 0, 255]  # 左
            img[:, max_bbox[2]:] = [0, 0, 0, 255]  # 右

            # imageio.imwrite(full_schp_img, img)
            # plt.imshow(img)
            # plt.scatter([max_bbox[0], max_bbox[2]], [max_bbox[1], max_bbox[3]])
            # plt.show()
            # break
        # break