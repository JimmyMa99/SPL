from operator import mod
import os
from pickle import FALSE, NONE, TRUE
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from core.datasets import *
from tools.general.io_utils import *
from tools.general.Q_util import *
from tools.dataset.voc_utils import *
from tools.ai.log_utils import *
from tools.ai.torch_utils import *
from tools.ai.evaluate_utils import *
import core.models as fcnmodel
import dataset_root
import importlib

parser = argparse.ArgumentParser()

###################################################################################


def get_params():
    ###############################################################################
    # Dataset
    ###############################################################################
    parser.add_argument('--dataset', default='voc12',
                        type=str, choices=['voc12', 'coco'])
    parser.add_argument('--domain', default='train', type=str)

    parser.add_argument(
        '--Qmodel_path', default='/media/ders/mazhiming/SP_CAM_code/SPCAM/experiments/models/train_Q_/00.pth', type=str)  #
    parser.add_argument(
        '--Cmodel_path', default='log/voc_dyrenum65_thr0.8_25ep/best_checkpoint.pth', type=str)  #

    parser.add_argument('--savepng', default=True, type=str2bool)
    parser.add_argument('--savenpy', default=False, type=str2bool)

    parser.add_argument('--ASAM', default=True, type=str2bool)

    parser.add_argument('--tag', default='train', type=str)
    parser.add_argument('--curtime', default='00', type=str)

    args = parser.parse_args()
    return args


class evaluator:
    def __init__(self, dataset='voc12', domain='train', ASAM=True, save_np_path=None, savepng_path=None, muti_scale=False, th_list=list(np.arange(0.2, 0.5, 0.1)), refine_list=range(0, 50, 10)) -> None:
        self.C_model = None
        self.Q_model = None
        self.args = None
        self.ASAM = ASAM
        if (muti_scale):
            self.scale_list = [0.5, 1, 1.5, 2.0, -
                               0.5, -1, -1.5, -2.0]  # - is flip
        else:
            self.scale_list = [1.0]  # - is flip

        self.th_list = th_list
        self.refine_list = refine_list
        self.parms = []
        for renum in self.refine_list:
            for th in self.th_list:
                self.parms.append((renum, th))
        class_num = 21 if dataset == 'voc12' else 81
        self.meterlist = [Calculator_For_mIoU(
            class_num) for x in self.parms]

        self.save_png_path = savepng_path
        self.save_np_path = save_np_path
        if (self.save_png_path != None):
            if not os.path.exists(self.save_png_path):
                os.mkdir(self.save_png_path)

        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]

        test_transform = transforms.Compose([
            Normalize_For_Segmentation(imagenet_mean, imagenet_std),
            Transpose_For_Segmentation()
        ])
        if (dataset == 'voc12'):
            valid_dataset = Dataset_For_Evaluation(
                dataset_root.VOC_ROOT, domain, test_transform, dataset)
        else:
            print('no that dataset')
            exit()

        self.valid_loader = DataLoader(
            valid_dataset, batch_size=1, num_workers=1, shuffle=False, drop_last=False)

    def get_cam(self, images, ids, Qs):
        with torch.no_grad():
            cam_list = []
            _, _, h, w = images.shape
            for s, q in zip(self.scale_list, Qs):
                target_size = (round(h * abs(s)), round(w * abs(s)))
                scaled_images = F.interpolate(
                    images, target_size, mode='bilinear', align_corners=False)
                H_, W_ = int(
                    np.ceil(target_size[0]/16.)*16), int(np.ceil(target_size[1]/16.)*16)
                scaled_images = F.interpolate(
                    scaled_images, (H_, W_), mode='bilinear', align_corners=False)
                if (s < 0):
                    scaled_images = torch.flip(
                        scaled_images, dims=[3])  # ?dims
                if (self.ASAM):
                    logits, pred, convlist = self.C_model(scaled_images)
                    b, c, h, w = logits.shape
                else:
                    logits, pred, convlist = self.C_model(scaled_images)

                pred = F.softmax(logits, dim=1)

                cam_list.append(torch.roll(pred, 1, 1))
        return cam_list

    def get_Q(self, images, ids):
        _, _, h, w = images.shape
        Q_list = []
        affmat_list = []

        for s in self.scale_list:
            target_size = (round(h * abs(s)), round(w * abs(s)))
            H_, W_ = int(
                np.ceil(target_size[0]/16.)*16), int(np.ceil(target_size[1]/16.)*16)
            scaled_images = F.interpolate(
                images, (H_, W_), mode='bilinear', align_corners=False)
            if (s < 0):
                scaled_images = torch.flip(scaled_images, dims=[3])  # ?dims
            pred = self.Q_model(scaled_images)
            Q_list.append(pred)
            affmat_list.append(calc_affmat(pred))
        return Q_list, affmat_list

    def get_mutiscale_cam(self, cam_list, Q_list, affmat_list, refine_time=0):
        _, _, h, w = Q_list[self.scale_list.index(1.0)].shape
        refine_cam_list = []
        for cam, Q, affmat, s in zip(cam_list, Q_list, affmat_list, self.scale_list):
            if (self.ASAM):
                for i in range(refine_time):
                    cam = refine_with_affmat(cam, affmat)
                cam = upfeat(cam, Q, 16, 16)

            cam = F.interpolate(cam, (int(h), int(w)),
                                mode='bilinear', align_corners=False)
            if (s < 0):
                cam = torch.flip(cam, dims=[3])  # ?dims
            refine_cam_list.append(cam)

        refine_cam = torch.sum(torch.stack(refine_cam_list), dim=0)
        return refine_cam

    def getbest_miou(self, clear=True):
        iou_list = []
        for parm, meter in zip(self.parms, self.meterlist):
            cur_iou = meter.get(clear=clear)[-2]
            iou_list.append((cur_iou, parm))
        iou_list.sort(key=lambda x: x[0], reverse=True)
        return iou_list

    def evaluate(self, C_model, Q_model=None, args=None):
        self.C_model, self.Q_model, self.args = C_model, Q_model, args
        self.C_model.eval()
        if (self.ASAM):
            self.Q_model.eval()
        with torch.no_grad():
            length = len(self.valid_loader)
            for step, (images, image_ids, tags, gt_masks) in enumerate(self.valid_loader):
                images = images.cuda()
                gt_masks = gt_masks.cuda()
                _, _, h, w = images.shape
                if (self.ASAM):
                    Qs, affmats = self.get_Q(images, image_ids)
                else:
                    Qs = [images for x in range(len(self.scale_list))]
                    affmats = [None for x in range(len(self.scale_list))]
                cams_list = self.get_cam(images, image_ids, Qs)
                mask = tags.unsqueeze(2).unsqueeze(3).cuda()
                # if args['network_type']==cls:

                for renum in self.refine_list:
                    refine_cams = self.get_mutiscale_cam(
                        cams_list, Qs, affmats, renum)
                    cams = (make_cam(refine_cams) * mask)
                    cams = F.interpolate(
                        cams, (int(h), int(w)), mode='bilinear', align_corners=False)
                    if (self.save_np_path != None):
                        np.save(os.path.join(self.save_np_path,
                                image_ids[0]+'.npy'), cams.cpu().numpy())
                    for th in self.th_list:
                        cams[:, 0] = th  # predictions.max()
                        predictions = torch.argmax(cams, dim=1)
                        for batch_index in range(images.size()[0]):
                            pred_mask = get_numpy_from_tensor(
                                predictions[batch_index])
                            gt_mask = get_numpy_from_tensor(  # cv2.imwrite("1.png",pred_mask*10)
                                gt_masks[batch_index])
                            gt_mask = cv2.resize(
                                gt_mask, (pred_mask.shape[1], pred_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
                            # self.getbest_miou(clear=False) #,self.meterlist[10].get(clear=False)
                            self.meterlist[self.parms.index((renum, th))].add(
                                pred_mask, gt_mask)
                            if (self.save_png_path != None):
                                cur_save_path = os.path.join(
                                    self.save_png_path, str(th))
                                if not os.path.exists(cur_save_path):
                                    os.mkdir(cur_save_path)
                                cur_save_path = os.path.join(
                                    cur_save_path, str(renum))
                                if not os.path.exists(cur_save_path):
                                    os.mkdir(cur_save_path)
                                img_path = os.path.join(
                                    cur_save_path, image_ids[batch_index]+'.png')
                                save_colored_mask(pred_mask, img_path)

                sys.stdout.write(
                    '\r# Evaluation [{}/{}] = {:.2f}%'.format(step + 1, length, (step + 1) / length * 100))
                sys.stdout.flush()
        self.C_model.train()
        if (self.save_png_path != None):
            savetxt_path = os.path.join(self.save_png_path, "result.txt")
            with open(savetxt_path, 'wb') as f:
                for parm, meter in zip(self.parms, self.meterlist):
                    cur_iou = meter.get(clear=False)[-2]
                    f.write('{:>10.2f} {:>10.2f} {:>10.2f}\n'.format(
                        cur_iou, parm[0], parm[1]).encode())
        ret = self.getbest_miou()

        return ret


if __name__ == "__main__":
    args = get_params()

    log_tag = create_directory(f'./experiments/logs/{args.tag}/')
    log_path = log_tag + f'/{args.curtime}.txt'
    if (args.savepng or args.savenpy):
        prediction_tag = create_directory(
            f'./experiments/predictions/{args.tag}/')
        prediction_path = create_directory(prediction_tag + f'{args.curtime}/')

    log_func = lambda string='': log_print(string, log_path)
    log_func('[i] {}'.format(args.tag))
    log_func(str(args))

    class_num = 21 if args.dataset == 'voc12' else 81

    args.network = 'models.resnet38_eps'
    args.num_classes = 20
    args.network_type = 'eps'
    model = getattr(importlib.import_module(args.network), 'Net')(args)

    model = model.cuda()
    model.train()
    model.load_state_dict(torch.load(args.Cmodel_path))
    model = nn.DataParallel(model)

    if (args.ASAM):
        Q_model = fcnmodel.SpixelNet1l_bn().cuda()
        Q_model.load_state_dict(torch.load(args.Qmodel_path))
        Q_model = nn.DataParallel(Q_model)
        Q_model.eval()
    else:
        Q_model = None
    _savepng_path = None
    _savenpy_path = None
    if (args.savepng):
        _savepng_path = create_directory(prediction_path+'pseudo/')
    if (args.savenpy):
        _savenpy_path = create_directory(prediction_path+'camnpy/')

    evaluatorA = evaluator(dataset='voc12', domain=args.domain, muti_scale=True, ASAM=args.ASAM,
                           save_np_path=_savenpy_path, savepng_path=_savepng_path, refine_list=[0, 20, 30, 40], th_list=[0.2, 0.3, 0.4])
    ret = evaluatorA.evaluate(model, Q_model)
    log_func(ret)
