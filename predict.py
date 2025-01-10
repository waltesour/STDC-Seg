# @Author  : SONG SHUXIANG
# @Time    : 2025/1/10
# @Func    : STDC-Seg pth模型与onnx模型进行推理
# @Usage    :
#       修改pth模型和onnx模型所在路径
#       修改验证数据集所在路径和保存结果所在路径

from skimage import morphology
import math
import cv2
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import os
import sys
import torch.nn.functional as F
import onnxruntime as ort
sys.path.append('/home/inspur/workspace/SONGSX/STDC-Seg') # 以便正确导入models文件夹
from models.model_stages import BiSeNet
from transform import *

def get_color_map_list(num_classes, custom_color=None):
    """
    Returns the color map for visualizing the segmentation mask,
    which can support arbitrary number of classes.

    Args:
        num_classes (int): Number of classes.
        custom_color (list, optional): Save images with a custom color map. Default: None, use paddleseg's default color map.

    Returns:
        (list). The color map.
    """

    num_classes += 1
    color_map = num_classes * [0, 0, 0]
    for i in range(0, num_classes):
        j = 0
        lab = i
        while lab:
            color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
            color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
            color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
            j += 1
            lab >>= 3
    color_map = color_map[3:]

    if custom_color:
        color_map[:len(custom_color)] = custom_color
    return color_map

class PreDataload(Dataset):
    def __init__(self, rootpth, dataname, cropsize=(640, 480), *args, **kwargs):
        super(PreDataload, self).__init__(*args, **kwargs)

        self.imgs = {}
        imgnames = []
        impth = os.path.join(rootpth, dataname)
        folders = os.listdir(impth)
        im_names = folders
        names = [x.split('.')[0].split('/')[-1] for x in im_names]
        impths = [os.path.join(impth, el) for el in im_names]
        imgnames.extend(names)
        self.imgs.update(dict(zip(names, impths)))

        self.imnames = imgnames
        self.len = len(self.imnames)

        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def __getitem__(self, idx):
        fn = self.imnames[idx]
        impth = self.imgs[fn]
        img = Image.open(impth).convert('RGB')
        img = self.to_tensor(img)
        # im_test = cv2.imread(impth)
        # print(img.shape, im_test.shape)
        return img, impth

    def __len__(self):
        return self.len

def prediect(respth='./pretrained', dspth='./data', backbone='CatNetSmall', scale=0.75, use_boundary_2=False,
             use_boundary_4=False, use_boundary_8=False, use_boundary_16=False, use_conv_last=False,
             outpth="/home/inspur/workspace/SONGSX/STDC-Seg/result/"):
    print('scale', scale)
    print('use_boundary_2', use_boundary_2)
    print('use_boundary_4', use_boundary_4)
    print('use_boundary_8', use_boundary_8)
    print('use_boundary_16', use_boundary_16)

    batchsize = 1
    n_workers = 2
    # CityScapes = dataload_configuration('DataGeneral')
    dataname = os.path.basename(dspth)
    dirname = os.path.dirname(dspth)
    dsval = PreDataload(dirname, dataname)
    dl = DataLoader(dsval,
                    batch_size=batchsize,
                    shuffle=False,
                    num_workers=n_workers,
                    drop_last=False)

    os.system('rm -rf {}'.format(outpth))
    if not os.path.exists(outpth):
        os.makedirs(outpth)
    # kernel = np.ones((20, 20), np.uint8)
    # max_pool = nn.MaxPool2d(kernel_size=20, stride=1, padding=2)
    color_map = get_color_map_list(256)
    color_map = [color_map[i:i + 3] for i in range(0, len(color_map), 3)]
    color_map = np.array(color_map).astype("uint8")
    color_map[0, 0] = 255
    color_map[0, 1] = 255
    color_map[0, 2] = 255
    skeleton_out_dir = os.path.join(outpth, 'skeleton')
    if not os.path.exists(skeleton_out_dir):
        os.makedirs(skeleton_out_dir)

    # 加载pth模型
    n_classes = 3
    print("backbone:", backbone)
    net = BiSeNet(backbone=backbone, n_classes=n_classes,
                  use_boundary_2=use_boundary_2, use_boundary_4=use_boundary_4,
                  use_boundary_8=use_boundary_8, use_boundary_16=use_boundary_16,
                  use_conv_last=use_conv_last)
    net.load_state_dict(torch.load(respth))
    net.cuda()
    net.eval()

    # 加载onnx模型
    output_onnx_path='/home/inspur/workspace/SONGSX/STDC-Seg/checkpoints/bbdd1211/pths/model_maxmIOU100_feat_out.onnx'
    ort_session = ort.InferenceSession(output_onnx_path)

    with torch.no_grad():
        for i, (imgs, img_pth) in enumerate(dl):
            img_pth = img_pth[0]
            # print(img_pth)
            basename = os.path.basename(img_pth)
            N, C, H, W = imgs.size()
            # new_hw = [int(H * scale), int(W * scale)]
            new_hw = [2432, 2048] # 输入尺寸大效果更好,比例要和训练图片比例一致效果好
            new_imgs_cpu = F.interpolate(imgs, new_hw, mode='bilinear', align_corners=True)

            #获取 pth 模型推理结果
            new_imgs = new_imgs_cpu.cuda()
            logits = net(new_imgs) # [0]是经过 FFM - 8x 的结果
            logits_numpy = logits[0]
            logits = F.interpolate(logits_numpy, [H, W], mode='bilinear')
            logits = torch.softmax(logits, dim=1)   # 沿着类别维度,得到每个像素属于不同类别的概率分布
            preds = torch.argmax(logits, dim=1)     # 对于每个像素位置，选出概率最大的类别索引作为预测类别
            preds = preds.squeeze(dim=0)            # 如果preds有一个多余的单一维度（例如，形状为(1,H,W)），那么这一行会移除这个维度，使得形状变为(H,W)
            preds = preds.cpu().numpy().astype('uint8')
            skeleton = preds * 122
            skeleton_out_path = os.path.join(skeleton_out_dir, basename)
            skeleton_out_path = skeleton_out_path[:-4]+'_pth.jpg'
            cv2.imwrite(skeleton_out_path, skeleton)


            # 获取 ONNX 推理结果
            dummy_input_numpy2 = new_imgs_cpu.numpy()
            logits = ort_session.run(None, {'img': dummy_input_numpy2})
            logits_onnx = logits[0]
            logits_onnx = torch.from_numpy(logits_onnx)
            logits = F.interpolate(logits_onnx, [H, W], mode='bilinear')
            logits = torch.softmax(logits, dim=1)   # 沿着类别维度,得到每个像素属于不同类别的概率分布
            preds = torch.argmax(logits, dim=1)     # 对于每个像素位置，选出概率最大的类别索引作为预测类别
            preds = preds.squeeze(dim=0)            # 如果preds有一个多余的单一维度（例如，形状为(1,H,W)），那么这一行会移除这个维度，使得形状变为(H,W)
            preds = preds.cpu().numpy().astype('uint8')
            skeleton = preds * 122
            skeleton_out_path = os.path.join(skeleton_out_dir, basename)
            skeleton_out_path = skeleton_out_path[:-4]+'_onnx.jpg'
            cv2.imwrite(skeleton_out_path, skeleton)
            
            print(i)


if __name__ == '__main__':
    prediect("/home/inspur/workspace/SONGSX/STDC-Seg/checkpoints/bbdd1211/pths/model_maxmIOU100.pth",
             dspth="/home/inspur/workspace/SONGSX/Dataset/baddall2cls-seg/images/val", backbone='STDCNet813',
             scale=1, use_boundary_2=False, use_boundary_4=False, use_boundary_8=True, use_boundary_16=False)
