import os
import time
import imageio
import argparse
import importlib
import numpy as np
from PIL import Image

import cv2
import torch
import torchvision
import torch.nn.functional as F
from torch.multiprocessing import Process
import imgviz

from utils import imutils, pyutils
from utils.imutils import HWC_to_CHW
from network.resnet38d import Normalize
from metadata.dataset import load_img_id_list, load_img_label_list_from_npy


start = time.time()
os.environ["CUDA_VISIBLE_DEVICES"] = "7"


def save_colored_mask(mask, save_path):
    lbl_pil = Image.fromarray(mask.astype(np.uint8), mode="P")
    colormap = imgviz.label_colormap()
    lbl_pil.putpalette(colormap.flatten())
    lbl_pil.save(save_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", default="network.resnet38_eps", type=str)
    parser.add_argument(
        "--weights", default='save/eps/checkpoint_cls.pth', type=str)
    parser.add_argument("--n_gpus", type=int, default=2)
    parser.add_argument(
        "--infer_list", default="metadata/voc12/train.txt", type=str)
    parser.add_argument("--n_processes_per_gpu", default=(2, 2), type=int)
    parser.add_argument("--n_total_processes", default=4, type=int)
    parser.add_argument(
        "--img_root", default="/media/ders/mazhiming/datasets/VOC2012_1/JPEGImages/", type=str)
    parser.add_argument("--crf", default='crf_my', type=str)
    parser.add_argument("--crf_alpha", default=[4, 32], type=int)
    parser.add_argument("--crf_t", default=[1], type=int)
    parser.add_argument("--cam_npy", default=None, type=str)
    parser.add_argument("--cam_png", default=None, type=str)
    parser.add_argument(
        "--npy_cam", default='experiments/predictions/voc_dyrenum65_thr0.8_25ep_npy/00/camnpy/0', type=str)
    parser.add_argument("--thr", default=0.3, type=float)
    parser.add_argument("--dataset", default='voc12', type=str)
    args = parser.parse_args()

    if args.dataset == 'voc12':
        args.num_classes = 20
    elif args.dataset == 'coco':
        args.num_classes = 80
    else:
        raise Exception('Error')

    # model information
    if 'cls' in args.network:
        args.network_type = 'cls'
        args.model_num_classes = args.num_classes
    elif 'eps' in args.network:
        args.network_type = 'eps'
        args.model_num_classes = args.num_classes + 1
    else:
        raise Exception('No appropriate model type')

    # save path
    args.save_type = list()
    if args.cam_npy is not None:
        os.makedirs(args.cam_npy, exist_ok=True)
        args.save_type.append(args.cam_npy)
    if args.cam_png is not None:
        os.makedirs(args.cam_png, exist_ok=True)
        args.save_type.append(args.cam_png)
    if args.crf:
        args.crf_list = list()
        for t in args.crf_t:
            for alpha in args.crf_alpha:
                crf_folder = os.path.join(
                    args.crf, 'crf_{}_{}'.format(t, alpha))
                os.makedirs(crf_folder, exist_ok=True)
                args.crf_list.append((crf_folder, t, alpha))
                args.save_type.append(crf_folder)

    # processors
    args.n_processes_per_gpu = [int(_) for _ in args.n_processes_per_gpu]
    args.n_total_processes = sum(args.n_processes_per_gpu)
    return args


def preprocess(image, scale_list, transform):
    img_size = image.size
    num_scales = len(scale_list)
    multi_scale_image_list = list()
    multi_scale_flipped_image_list = list()

    # insert multi-scale images
    for s in scale_list:
        target_size = (round(img_size[0] * s), round(img_size[1] * s))
        scaled_image = image.resize(target_size, resample=Image.CUBIC)
        multi_scale_image_list.append(scaled_image)
    # transform the multi-scaled image
    for i in range(num_scales):
        multi_scale_image_list[i] = transform(multi_scale_image_list[i])
    # augment the flipped image
    for i in range(num_scales):
        multi_scale_flipped_image_list.append(multi_scale_image_list[i])
        multi_scale_flipped_image_list.append(
            np.flip(multi_scale_image_list[i], -1).copy())
    return multi_scale_flipped_image_list


def _crf_with_alpha(image, cam_dict, alpha, t=10):
    v = np.array(list(cam_dict.values()))
    bg_score = np.power(1 - np.max(v, axis=0, keepdims=True), alpha)
    bgcam_score = np.concatenate((bg_score, v), axis=0)
    crf_score = imutils.crf_inference(
        image, bgcam_score, labels=bgcam_score.shape[0], t=t)
    n_crf_al = dict()
    n_crf_al[0] = crf_score[0]
    for i, key in enumerate(cam_dict.keys()):
        n_crf_al[key+1] = crf_score[i+1]
    return n_crf_al


def infer_cam_mp(process_id, image_ids, label_list, cur_gpu):
    print('process {} starts...'.format(os.getpid()))

    print(process_id, cur_gpu)
    print('GPU:', cur_gpu)
    print('{} images per process'.format(len(image_ids)))

    for i, (img_id, label) in enumerate(zip(image_ids, label_list)):

        # load image
        img_path = os.path.join(args.img_root, img_id + '.jpg')
        img = Image.open(img_path).convert('RGB')
        org_img = np.asarray(img)
        npy_cam_ = np.load(os.path.join(
            args.npy_cam, img_id + '.npy'), allow_pickle=True)
        H, W, C = org_img.shape
        method = 'recam'
        if npy_cam_.dtype == 'O':
            pred = np.zeros([21, H, W])
            npy_dict = npy_cam_[()]
            if method == 'l2g' or method == 'cls':
                for i in npy_dict.keys():
                    pred[i+1, :, :] = npy_dict[i]

            elif method == 'recam':
                c = npy_dict['keys'].shape[0]
                for i in range(c):
                    pred[npy_dict['keys'][i]+1] = npy_dict['high_res'][i]
            npy_cam_ = torch.tensor(pred).unsqueeze(0)
            npy_cam_ = npy_cam_.detach().numpy()
        else:
            npy_cam_ = npy_cam_
        npy_cam = np.squeeze(npy_cam_, axis=0)[1:, :, :]
        npy_cam_ = np.squeeze(npy_cam_, axis=0)
        cam_dict = {}
        for j in range(args.num_classes):
            if label[j] > 1e-5:
                cam_dict[j] = npy_cam[j]

        h, w = list(cam_dict.values())[0].shape

        # save cam
        # if args.cam_npy is not None:

        #     np.save(os.path.join(args.cam_npy, img_id + '.npy'), cam_dict)

        # if args.cam_png is not None:
        #     imageio.imwrite(os.path.join(args.cam_png, img_id + '.png'), pred)
        zeromat = np.zeros_like(npy_cam_)
        if args.crf is not None:
            for folder, t, alpha in args.crf_list:
                cam_crf = _crf_with_alpha(org_img, cam_dict, alpha, t=t)
                for i in cam_crf.keys():
                    zeromat[i] = cam_crf.get(i)
                cam_crf_ = np.argmax(zeromat, axis=0).astype(np.uint8)
                # cam_crf_=zeromat
                # np.save(os.path.join(folder, img_id + '.npy'), cam_crf)
                # imageio.imwrite(os.path.join(folder, img_id + '.png'), cam_crf_)
                save_colored_mask(cam_crf_, save_path=os.path.join(
                    folder, img_id + '.png'))
                # print(img_id+' saved' + cv2.imwrite(os.path.join(folder, img_id + '.png'),cam_crf_))
        if i % 10 == 0:
            print('PID{}, {}/{} is complete'.format(process_id, i, len(image_ids)))


def main_mp():
    image_ids = load_img_id_list(args.infer_list)
    label_list = load_img_label_list_from_npy(image_ids, args.dataset)
    n_total_images = len(image_ids)
    assert len(image_ids) == len(label_list)

    saved_list = sorted([file[:-4] for file in os.listdir(args.save_type[0])])
    n_saved_images = len(saved_list)
    new_image_ids = list()
    new_label_list = list()
    for i, name in enumerate(image_ids):
        if name not in saved_list:
            new_image_ids.append(name)
            new_label_list.append(label_list[i])
    image_ids = new_image_ids
    label_list = new_label_list

    n_total_processes = args.n_total_processes
    print('===========================')
    print('OVERALL INFORMATION')
    print('n_gpus:', n_gpus)
    print('n_processes_per_gpu', args.n_processes_per_gpu)
    print('n_total_processes:', n_total_processes)
    print('n_total_images:', n_total_images)
    print('n_saved_images:', n_saved_images)
    print('n_images_to_proceed', len(image_ids))
    print('===========================')

    sub_image_ids = list()
    sub_label_list = list()

    # split model and data
    split_size = len(image_ids) // n_total_processes
    for i in range(n_total_processes):
        # split image ids and labels
        if i == n_total_processes - 1:
            sub_image_ids.append(image_ids[split_size * i:])
            sub_label_list.append(label_list[split_size * i:])
        else:
            sub_image_ids.append(
                image_ids[split_size * i:split_size * (i + 1)])
            sub_label_list.append(
                label_list[split_size * i:split_size * (i + 1)])

    # multi-process
    gpu_list = list()
    for idx, num in enumerate(args.n_processes_per_gpu):
        gpu_list.extend([idx for i in range(num)])
    processes = list()
    for idx, process_id in enumerate(range(n_total_processes)):
        proc = Process(target=infer_cam_mp,
                       args=(process_id, sub_image_ids[idx], sub_label_list[idx], gpu_list[idx]))
        processes.append(proc)
        proc.start()

    for proc in processes:
        proc.join()


if __name__ == '__main__':
    crf_alpha = (4, 32)
    args = parse_args()

    n_gpus = args.n_gpus
    scales = (0.5, 1.0, 1.5, 2.0)
    normalize = Normalize()
    transform = torchvision.transforms.Compose(
        [np.asarray, normalize, HWC_to_CHW])

    main_mp()

    print(time.time() - start)
