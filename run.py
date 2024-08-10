#
# CUDA_VISIBLE_DEVICES=7 python run.py configs/infer_occlumesh.yaml n_views=4  infer.ckpt_path=ckpts/epoch=29.ckpt scene_name=000-000/a7ea9a2c734345e19ee0b37cf9674865

from omegaconf import OmegaConf
import os, torch, math, imageio, cv2
import torch.nn as nn
import numpy as np
from tqdm import tqdm

import sys,json
from lightning.system import system
# from torch.utils.data import DataLoader
import pytorch_lightning as L
# from dataLoader import dataset_dict

from pytorch_msssim import ssim
# from tools.gen_video_path import uni_video_path,uni_mesh_path

import torch.nn.functional as F
from tools.depth import acc_threshold,abs_error


import random
import h5py
import imageio.v3 as iio

# def read_image(scene, view_idx, bg_color, scene_name):
def read_image(path, idx, bg_color)  :
    # img = np.array(scene[f'color_single_{view_idx}'])
    path_idx = os.path.join(path, f'colors_single_{idx}.png')
    img = iio.imread(path_idx)
    mask = (img[...,-1] > 0).astype('uint8')
    img = img.astype(np.float32) / 255.
    img = (img[..., :3] * img[..., -1:] + bg_color*(1 - img[..., -1:])).astype(np.float32)

    # if self.cfg.load_normal:

    #     normal = np.array(scene[f'normal_{view_idx}'])
    #     normal = normal.astype(np.float32) / 255. * 2 - 1.0
    #     return img, normal, mask

    return img, None, mask

def fov_to_ixt(fov, reso):
    ixt = np.eye(3, dtype=np.float32)
    ixt[0][2], ixt[1][2] = reso[0]/2, reso[1]/2
    focal = .5 * reso / np.tan(.5 * fov)
    ixt[[0,1],[0,1]] = focal
    return ixt

def read_cam(self, scene, view_idx):
    c2w = np.array(scene[f'c2w_{view_idx}'], dtype=np.float32)
    w2c = np.linalg.inv(c2w)
    fov = np.array(scene[f'fov_{view_idx}'], dtype=np.float32)
    ixt = fov_to_ixt(fov, self.img_size)
    return ixt, c2w, w2c

# def read_views(self, scene, src_views, scene_name):
#     src_ids = src_views
#     bg_colors = []
#     ixts, exts, w2cs, imgs, msks, normals = [], [], [], [], [], []
#     for i, idx in enumerate(src_ids):
        
#         if self.split!='train' or i < self.n_group:
#             bg_color = np.ones(3).astype(np.float32)
#         else:
#             bg_color = np.ones(3).astype(np.float32)*random.choice([0.0, 0.5, 1.0])

#         bg_colors.append(bg_color)
        
#         img, normal, mask = self.read_image(scene, idx, bg_color, scene_name)
#         imgs.append(img)
#         ixt, ext, w2c = self.read_cam(scene, idx)
#         ixts.append(ixt)
#         exts.append(ext)
#         w2cs.append(w2c)
#         msks.append(mask)
#         normals.append(normal)
#     return np.stack(imgs), np.stack(bg_colors), np.stack(normals), np.stack(msks), np.stack(exts), np.stack(w2cs), np.stack(ixts)


def read_views(path):
    datapath = os.path.join(path,'data.h5')
    datah5 = h5py.File(datapath, 'r')
    num_total_view = datah5['fov'][:].shape[0]
    
    ixts, exts, w2cs, imgs, msks, normals = [],[],[],[],[],[]
    bg_colors = []
    for i in range(num_total_view):
        bg_color = np.ones(3).astype(np.float32)
        img, normal, mask = read_image(path, i, bg_color)
        imgs.append(img)
        msks.append(mask)
        bg_colors.append(bg_color)
        ext = datah5['cam_poses'][:][i]
        fov = datah5['fov'][:][i][0]
        reso = np.array(img.shape[:2], dtype = np.float32)
        ixt = fov_to_ixt(fov, reso)
        w2c = np.linalg.inv(ext)
        ixts.append(ixt)
        exts.append(ext)
        w2cs.append(w2c)
        normals.append(normal)
    return np.stack(imgs), np.stack(bg_colors), np.stack(normals), np.stack(msks), np.stack(exts), np.stack(w2cs), np.stack(ixts)


def build_dataset(cfg):
    
    scene_name = cfg.scene_name
    path = os.path.join(cfg.infer.dataset.data_root, scene_name)
    tar_img, bg_colors, tar_nrms, tar_msks, tar_c2ws, tar_w2cs, tar_ixts = read_views(path)
    breakpoint()
    
     
    
    ret ={
        'fovx': None, #
        'foxy': None, #
        'tar_c2w': None, #
        'tar_w2c': None, #
        'tar_ixt': None, #
        'tar_rgb': None, #
        'tar_msk': None,
        'transform_mats': None, #
        'bg_color': None, #
        'render_image_scale' : None, #
        'near_far' : None , #
        'meta' :{
            'tar_h': None, #
            'tar_w' : None #
        } ,
        'tar_rays' : None, #

    }



@torch.no_grad()
def main(cfg):
    data = build_dataset(cfg)
    torch.set_float32_matmul_precision('medium')
    device = 'cuda'
    my_system = system.load_from_checkpoint(cfg.infer.ckpt_path, cfg=cfg, map_location=device)

    sample = None # TODO
    sample = {key: tensor.to(device) if torch.is_tensor(tensor) else tensor for key, tensor in sample.items()} 
    my_system.net.eval()

    return_buffer =True
    output =  my_system.net(sample, with_fine=True, return_buffer=return_buffer)
    images = output['image_fine'][0]
    img_gt = sample['tar_rgb'][0].permute(1,0,2,3).reshape(images.shape)
    alpha = output['acc_map'][0][...,None]
    normal_white = ((output['rend_normal_fine'][0]*alpha+1-alpha) + 1)/2



if __name__ == "__main__":

    base_conf = OmegaConf.load('configs/base.yaml')
    path_config = sys.argv[1]
    cli_conf = OmegaConf.from_cli()
    second_conf = OmegaConf.load(path_config)
    cfg = OmegaConf.merge(base_conf, second_conf, cli_conf)
    main(cfg)