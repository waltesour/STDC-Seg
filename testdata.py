#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import cv2
import os.path as osp
import os
from PIL import Image
import numpy as np
import json

from transform import *



class CityScapes(Dataset):
    def __init__(self, rootpth, cropsize=(1024, 512), mode='train', 
    randomscale=(0.125, 0.25, 0.375, 0.5, 0.675, 0.75, 0.875, 1.0, 1.25, 1.5), *args, **kwargs):
        super(CityScapes, self).__init__(*args, **kwargs)
        assert mode in ('train', 'val', 'test', 'trainval')
        self.mode = mode
        print('self.mode', self.mode)
        self.ignore_lb = 255

        # with open('./cityscapes_info.json', 'r') as fr:
        #     labels_info = json.load(fr)
        # self.lb_map = {el['id']: el['trainId'] for el in labels_info}
        

        ## parse img directory
        self.imgs = {}
        imgnames = []
        impth = osp.join(rootpth, 'images', mode)
        print(impth)
        # folders = os.listdir(impth)
        # for fd in folders:
        #     fdpth = osp.join(impth, fd)
        im_names = os.listdir(impth)
        names = [el for el in im_names]
        impths = [osp.join(impth, el) for el in im_names]
        imgnames.extend(names)
        self.imgs.update(dict(zip(names, impths)))

        ## parse gt directory
        self.labels = {}
        gtnames = []
        gtpth = osp.join(rootpth, 'labels_png', mode)
        print(gtpth)
        # folders = os.listdir(gtpth)
        # for fd in folders:
        #     fdpth = osp.join(gtpth, fd)
        lbnames = os.listdir(gtpth)
        lbnames = [el for el in lbnames]
        names = [el for el in lbnames]
        lbpths = [osp.join(gtpth, el) for el in lbnames]
        gtnames.extend(names)
        self.labels.update(dict(zip(names, lbpths)))
        
        self.imnames = imgnames
        self.len = len(self.imnames)
        print('self.len', self.mode, self.len)
        print('self.imgnames: ',self.imnames)
        print('self.imgs.keys(): ',self.imgs.keys())
        print('self.labels.keys(): ',self.labels.keys())
        # assert set(imgnames) == set(gtnames) #检查图像文件名和标签文件名是否完全匹配
        # assert set(self.imnames) == set(self.imgs.keys())
        # assert set(self.imnames) == set(self.labels.keys())

        ## pre-processing
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0, 0, 0), (1, 1, 1)),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        self.trans_train = Compose([
            ColorJitter(
                brightness = 0.5,
                contrast = 0.5,
                saturation = 0.5),
            HorizontalFlip(),
            # RandomScale((0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)),
            RandomScale(randomscale),
            # RandomScale((0.125, 1)),
            # RandomScale((0.125, 0.25, 0.375, 0.5, 0.675, 0.75, 0.875, 1.0)),
            # RandomScale((0.125, 0.25, 0.375, 0.5, 0.675, 0.75, 0.875, 1.0, 1.125, 1.25, 1.375, 1.5)),
            RandomCrop(cropsize)
            ])


    def __getitem__(self, idx):
        fn  = self.imnames[idx]
        impth = self.imgs[fn]
        lbpth = get_label_path(impth)
        # lbpth = self.labels[fn]

        img = Image.open(impth).convert('RGB')
        label = Image.open(lbpth)
        # print(label.format, label.size, label.mode)
        # p = label.getpixel((1024, 1024))
        # print(p)
        # label = label.convert("P")
        # label = cv2.cvtColor(np.asarray(label), cv2.COLOR_RGB2BGR)
        # new_label = np.zeros(label.shape)
        # new_label[:, :, 0][label[:, :, 0]>0] = 1
        # new_label[:, :, 1][label[:, :, 1]>0] = 1
        # new_label[:, :, 2][label[:, :, 2]>0] = 1
        # print(label.shape)
        # label = Image.fromarray(cv2.cvtColor(new_label, cv2.COLOR_BGR2RGB))

        if self.mode == 'train' or self.mode == 'trainval':
            im_lb = dict(im = img, lb = label) #创建一个以 'im' 和 'lb' 为键，分别以 img 和 label 为值的字典
            im_lb = self.trans_train(im_lb)
            img, label = im_lb['im'], im_lb['lb']
        img = self.to_tensor(img)
        label = np.array(label).astype(np.int64)[np.newaxis, :]
        # print(label.shape)
        # print(label[0][200, 1216])
        # label = self.convert_labels(label)
        return img, label


    def __len__(self):
        return self.len


    # def convert_labels(self, label):
    #     for k, v in self.lb_map.items():
    #         label[label == k] = v
    #     return label

def get_label_path(img_path):
    # 将路径分隔成列表
    parts = img_path.split(os.sep)
    # 检查是否路径格式正确
    if len(parts) < 4:
        raise ValueError("提供的路径格式不正确")
    # 替换相关部分以形成新的路径
    parts[-3] = 'labels_png'  # 修改目录名从原始图像目录 'images' 到标签目录 'labels_png'
    file_name, file_extension = os.path.splitext(parts[-1])
    parts[-1] = file_name + '.png'  # 修改文件扩展名到 .png
    # 使用os.sep连接各部分形成新路径，确保与操作系统兼容
    png_path = os.sep.join(parts)
    return png_path

if __name__ == "__main__":
    from tqdm import tqdm
    ds = CityScapes('./data/', n_classes=19, mode='val') #n_classes=类别+1
    uni = []
    for im, lb in tqdm(ds):
        lb_uni = np.unique(lb).tolist()
        uni.extend(lb_uni)
    print(uni)
    print(set(uni))

