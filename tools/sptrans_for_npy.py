import argparse
import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.io.image import read_image
from torchvision.transforms.functional import normalize, resize, to_pil_image
import imgviz
from PIL import Image
from tqdm import tqdm

import core.models as fcnmodel
from tools.general.Q_util import *

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

parser = argparse.ArgumentParser()


def get_params():

    parser.add_argument(
        '--Qmodel_path', default='/media/ders/mazhiming/SP_CAM_code/SPCAM/experiments/models/train_Q_/00.pth', type=str)  #
    parser.add_argument(
        '--npy_path', default='/media/ders/mazhiming/ReCAM/result_default5/seg_npy', type=str)  #
    parser.add_argument(
        '--save_dir', default='Ablation_exp/recam(irn)_npy_refine40', type=str)
    parser.add_argument("--method", type=str, default='recam')      #

    parser.add_argument("--img_dir", type=str,
                        default='/media/ders/mazhiming/datasets/VOC2012')
    parser.add_argument("--train_list", type=str,
                        default='./metadata/voc12/train.txt')

    parser.add_argument('--refine_time', default=40, type=int)
    parser.add_argument('--bg_thr', default=0.28, type=float)

    args = parser.parse_args()
    return args


def img_pre_porc(img):
    _, _, h, w = img.shape
    H_, W_ = int(np.ceil(round(h)/16.)*16), int(np.ceil(round(w)/16.)*16)
    scaled_images = F.interpolate(
        img, (H_, W_), mode='bilinear', align_corners=False)
    return scaled_images


def get_Q(Q_model, images, ids):
    _, _, h, w = images.shape
    Q_list = []
    affmat_list = []
    scale_list = [1.0]
    for s in scale_list:
        target_size = (round(h * abs(s)), round(w * abs(s)))
        H_, W_ = int(
            np.ceil(target_size[0]/16.)*16), int(np.ceil(target_size[1]/16.)*16)
        scaled_images = F.interpolate(
            images, (H_, W_), mode='bilinear', align_corners=False)
        if (s < 0):
            scaled_images = torch.flip(scaled_images, dims=[3])  # ?dims
        pred = Q_model(scaled_images)
        Q_list.append(pred)
        affmat_list.append(calc_affmat(pred))
    return Q_list, affmat_list


def save_colored_mask(mask, save_path):
    lbl_pil = Image.fromarray(mask.astype(np.uint8), mode="P")
    colormap = imgviz.label_colormap()
    lbl_pil.putpalette(colormap.flatten())
    lbl_pil.save(save_path)


def make_cam(x, epsilon=1e-5):
    # relu(x) = max(x, 0)
    x = F.relu(x)

    b, c, h, w = x.size()

    flat_x = x.view(b, c, (h * w))
    max_value = flat_x.max(axis=-1)[0].view((b, c, 1, 1))

    return F.relu(x - epsilon) / (max_value + epsilon)


def get_refine(args, Q_model):
    img_list = open(args.train_list, encoding='utf-8')
    img_list = img_list.read().split('\n')[:-1]
    os.makedirs(args.save_dir, exist_ok=True)
    for img_name in tqdm(img_list):
        img_path = os.path.join(args.img_dir, 'JPEGImages', img_name+'.jpg')
        npy_path = os.path.join(args.npy_path, img_name+'.npy')
        save_path = os.path.join(args.save_dir, img_name+'.png')

        npy = np.load(npy_path, allow_pickle=True)
        img = read_image(img_path).cuda()

        C, H, W = img.size()
        method = args.method
        if npy.dtype == 'O':
            npy_dict = npy[()]
            pred = np.zeros([21, H, W])
            if method == 'l2g' or method == 'cls' or method == 'eps':
                for i in npy_dict.keys():
                    pred[i+1, :, :] = npy_dict[i]
            elif method == 'recam':
                c = npy_dict['keys'].shape[0]
                for i in range(c):
                    try:
                        pred[npy_dict['keys'][i]+1] = npy_dict['high_res'][i]
                    except:
                        pred[npy_dict['keys'][i]+1] = npy_dict['cam'][i]
            pred = torch.tensor(pred).unsqueeze(0).cuda()
        else:
            pred = torch.tensor(npy).unsqueeze(0).cuda()

        input_tensor = normalize(
            img / 255., [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).unsqueeze(0)
        input_tensor = img_pre_porc(input_tensor)
        Qs, affmats = get_Q(Q_model, input_tensor.cuda(), img_name)
        _, _, ah, aw = affmats[0].shape
        pred = F.interpolate(pred, (int(ah), int(aw)),
                             mode='bilinear', align_corners=False)
        for i in range(args.refine_time):
            pred = refine_with_affmat(pred, affmats[0])
        pred = upfeat(pred, Qs[0], 16, 16)
        pred = F.interpolate(pred, (int(H), int(W)),
                             mode='bilinear', align_corners=False)
        # cam=F.softmax(pred,dim=1)[0].detach().cpu().numpy()
        # cam=make_cam(pred)[0].detach().cpu().numpy()
        cam = pred[0].detach().cpu().numpy()
        cam[0] = args.bg_thr
        cam = np.argmax(cam, axis=0)
        save_colored_mask(cam, save_path=save_path)

    print(args.save_dir)


if __name__ == '__main__':
    args = get_params()

    Q_model = fcnmodel.SpixelNet1l_bn().cuda()
    Q_model.load_state_dict(torch.load(args.Qmodel_path))
    Q_model = nn.DataParallel(Q_model)
    Q_model.eval()

    get_refine(args, Q_model)
