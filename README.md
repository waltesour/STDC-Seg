### Author  : SONG SHUXIANG
### Time    : 2024/12/23
### Func    : 记录STDC-Seg语义分割网络的使用方式

## 1、数据格式(按照如下文件夹层级)：
    dspth
    ├── images
    │    └── train
    │    └── val
    └── labels_png (png中背景灰度值0,其他类别灰度值按照 1、2 顺延)
    │    └── train
    │    └── val
注：
    json_to_png.py脚本可以将labelme标注的 json格式分割数据 转换成 png格式的分割图像,使用EiSeg标注工具可以标注png格式的数据
    (或者根据json中数据信息由通义千问生成合适的代码)

## 2、训练命令：train.sh
    相比原始代码改动如下：
    修改 model_stages.py中 InPlaceABNSync表示的BatchNorm2d函数为torch.nn.BatchNorm2d
    修改 model_stages.py中 F.avg_pool2d 为 F.adaptive_avg_pool2d (adaptive_avg_pool2d不需要变量做输入,这样导出动态onnx的时候不报错)
    修改 testdata.py 中get_label_path函数中png_path路径(test_data.py是重写的数据读取部分) 
    修改 train_bbdd.py(train.py是源码中原始文件,不用)中如下
    --- n_classes 为类别数目
    --- respath 模型训练完成后的保存路径
    --- dspth 训练数据所在路径
    --- max_iter 最大迭代次数,修改成100000以上有较好效果
    --- pretrain_path 提前下载的预训练模型文件
注：
    预训练模型的下载地址： 
    - BaiduYun Link: https://pan.baidu.com/s/1OdMsuQSSiK1EyNs6_KiFIw Password: q7dt
    - GoogleDrive Link: https://drive.google.com/drive/folders/1wROFwRt8qWHD4jSo8Zu1gp1d6oYJ3ns1?usp=sharing
    
## 3、推理命令：predict.sh
    修改 predict.py 中如下 
    --- respth 为训练得到的模型路径
    --- dspth 为测试数据集的路径
    --- outpth 为验证结果的保存路径

## 4、转onnx推理
    export.py 修改：
    --- from models.model_stages import BiSeNet 修改为 models.model_stages_export import BiSeNet 
    --- save_pth_path为训练得到的pth模型路径, onnx模型将保存在pth同级目录下
    --- output_names 网络有多少输出 output_names 就设定几个名称
    --- dynamic_axes 参数可以指定输出中哪个维度是可变的
*************************************************************************************************************

# Rethinking BiSeNet For Real-time Semantic Segmentation[[PDF](https://openaccess.thecvf.com/content/CVPR2021/papers/Fan_Rethinking_BiSeNet_for_Real-Time_Semantic_Segmentation_CVPR_2021_paper.pdf)]

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Mingyuan Fan, Shenqi Lai, Junshi Huang, Xiaoming Wei, Zhenhua Chai, Junfeng Luo, Xiaolin Wei

In CVPR 2021.

## Overview

<p align="center">
  <img src="images/overview-of-our-method.png" alt="overview-of-our-method" width="600"/></br>
  <span align="center">Speed-Accuracy performance comparison on the Cityscapes test set</span> 
</p>
We present STDC-Seg, an mannully designed semantic segmentation network with not only state-of-the-art performance but also faster speed than current methods.

Highlights:

* **Short-Term Dense Concatenation Net**: A task-specific network for dense prediction task.
* **Detail Guidance**: encode spatial information without harming inference speed.
* **SOTA**: STDC-Seg achieves extremely fast speed (over 45\% faster than the closest automatically designed competitor on CityScapes)  and maintains competitive accuracy.
  - see our Cityscapes test set submission [STDC1-Seg50](https://www.cityscapes-dataset.com/anonymous-results/?id=805e22f63fc53d1d0726cefdfe12527275afeb58d7249393bec6f483c3342b3b)  [STDC1-Seg75](https://www.cityscapes-dataset.com/anonymous-results/?id=6bd0def75600fd0f1f411101fe2bbb0a2be5dba5c74e2f7d7f50eecc23bae64c)  [STDC2-Seg50](https://www.cityscapes-dataset.com/anonymous-results/?id=b009a595f0d4e10a7f10ac25f29962b67995dc11b059f0c733ddd212a56b9ee0)  [STDC2-Seg75](https://www.cityscapes-dataset.com/anonymous-results/?id=9012a16cdeb9d52aaa9ad5fb9cc1c6284efe8a3daecee85b4413284364ff3f45).
  - Here is our speed-accuracy comparison on Cityscapes test&val set.

<p align="center">
<img src="images/comparison-cityscapes.png" alt="Cityscapes" width="400"/></br>
</p>

## Methods

<p align="center">
<img src="images/stdc-architecture.png" alt="stdc-architecture" width="600"/></br>
</p>

<p align="center">
<img src="images/stdcseg-architecture.png" alt="stdcseg-artchitecture" width="800"/></br>
  <span align="center">Overview of the STDC Segmentation network</span> 
</p>

## Prerequisites

- Pytorch 1.1
- Python 3.5.6
- NVIDIA GPU
- TensorRT v5.1.5.0 (Only need for testing inference speed)

This repository has been trained on Tesla V100. Configurations (e.g batch size, image patch size) may need to be changed on different platforms. Also, for fair competition, we test the inference speed on NVIDIA GTX 1080Ti.

## Installation

* Clone this repo:

```bash
git clone https://github.com/MichaelFan01/STDC-Seg.git
cd STDC-Seg
```

* Install dependencies:

```bash
pip install -r requirements.txt
```

* Install [PyCuda](https://wiki.tiker.net/PyCuda/Installation) which is a dependency of TensorRT.
* Install [TensorRT](https://github.com/NVIDIA/TensorRT) (v5.1.5.0): a library for high performance inference on NVIDIA GPUs with [Python API](https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/index.html#python).

## Usage

### 0. Prepare the dataset

* Download the [leftImg8bit_trainvaltest.zip](https://www.cityscapes-dataset.com/file-handling/?packageID=3) and [gtFine_trainvaltest.zip](https://www.cityscapes-dataset.com/file-handling/?packageID=1) from the Cityscapes.
* Link data to the  `data` dir.

  ```bash
  ln -s /path_to_data/cityscapes/gtFine data/gtFine
  ln -s /path_to_data/leftImg8bit data/leftImg8bit
  ```

### 1. Train STDC-Seg

Note: Backbone STDCNet813 denotes STDC1, STDCNet1446 denotes STDC2.

* Train STDC1Seg:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2
python -m torch.distributed.launch \
--nproc_per_node=3 train.py \
--respath checkpoints/train_STDC1-Seg/ \
--backbone STDCNet813 \
--mode train \
--n_workers_train 12 \
--n_workers_val 1 \
--max_iter 60000 \
--use_boundary_8 True \
--pretrain_path checkpoints/STDCNet813M_73.91.tar
```

* Train STDC2Seg:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2
python -m torch.distributed.launch \
--nproc_per_node=3 train.py \
--respath checkpoints/train_STDC2-Seg/ \
--backbone STDCNet1446 \
--mode train \
--n_workers_train 12 \
--n_workers_val 1 \
--max_iter 60000 \
--use_boundary_8 True \
--pretrain_path checkpoints/STDCNet1446_76.47.tar
```

We will save the model's params in model_maxmIOU50.pth for input resolution 512x1024，and model_maxmIOU75.pth for input resolution 768 x 1536.

ImageNet Pretrained STDCNet Weights for training and Cityscapes trained STDC-Seg weights for evaluation:

BaiduYun Link: https://pan.baidu.com/s/1OdMsuQSSiK1EyNs6_KiFIw  Password: q7dt

GoogleDrive Link:[https://drive.google.com/drive/folders/1wROFwRt8qWHD4jSo8Zu1gp1d6oYJ3ns1?usp=sharing](https://drive.google.com/drive/folders/1wROFwRt8qWHD4jSo8Zu1gp1d6oYJ3ns1?usp=sharing)

###

### 2. Evaluation

Here we use our pretrained STDCSeg as an example for the evaluation.

* Choose the evaluation model in evaluation.py:

```python
#STDC1-Seg50 mIoU 0.7222
evaluatev0('./checkpoints/STDC1-Seg/model_maxmIOU50.pth', dspth='./data', backbone='STDCNet813', scale=0.5, 
           use_boundary_2=False, use_boundary_4=False, use_boundary_8=True, use_boundary_16=False)

#STDC1-Seg75 mIoU 0.7450
evaluatev0('./checkpoints/STDC1-Seg/model_maxmIOU75.pth', dspth='./data', backbone='STDCNet813', scale=0.75, 
           use_boundary_2=False, use_boundary_4=False, use_boundary_8=True, use_boundary_16=False)

#STDC2-Seg50 mIoU 0.7424
evaluatev0('./checkpoints/STDC2-Seg/model_maxmIOU50.pth', dspth='./data', backbone='STDCNet1446', scale=0.5, 
           use_boundary_2=False, use_boundary_4=False, use_boundary_8=True, use_boundary_16=False)

#STDC2-Seg75 mIoU 0.7704
evaluatev0('./checkpoints/STDC2-Seg/model_maxmIOU75.pth', dspth='./data', backbone='STDCNet1446', scale=0.75, 
           use_boundary_2=False, use_boundary_4=False, use_boundary_8=True, use_boundary_16=False)
```

* Start the evaluation process:

```bash
CUDA_VISIBLE_DEVICES=0 python evaluation.py
```

### 3. Latency

#### 3.0 Latency measurement tools

* If you have successfully installed [TensorRT](https://github.com/chenwydj/FasterSeg#installation), you will automatically use TensorRT for the following latency tests (see [function](https://github.com/chenwydj/FasterSeg/blob/master/tools/utils/darts_utils.py#L167) here).
* Otherwise you will be switched to use Pytorch for the latency tests  (see [function](https://github.com/chenwydj/FasterSeg/blob/master/tools/utils/darts_utils.py#L184) here).

#### 3.1 Measure the latency of the FasterSeg

* Choose the evaluation model in run_latency:

```python
# STDC1Seg-50 250.4FPS on NVIDIA GTX 1080Ti
backbone = 'STDCNet813'
methodName = 'STDC1-Seg'
inputSize = 512
inputScale = 50
inputDimension = (1, 3, 512, 1024)

# STDC1Seg-75 126.7FPS on NVIDIA GTX 1080Ti
backbone = 'STDCNet813'
methodName = 'STDC1-Seg'
inputSize = 768
inputScale = 75
inputDimension = (1, 3, 768, 1536)

# STDC2Seg-50 188.6FPS on NVIDIA GTX 1080Ti
backbone = 'STDCNet1446'
methodName = 'STDC2-Seg'
inputSize = 512
inputScale = 50
inputDimension = (1, 3, 512, 1024)

# STDC2Seg-75 97.0FPS on NVIDIA GTX 1080Ti
backbone = 'STDCNet1446'
methodName = 'STDC2-Seg'
inputSize = 768
inputScale = 75
inputDimension = (1, 3, 768, 1536)
```

* Run the script:

```bash
CUDA_VISIBLE_DEVICES=0 python run_latency.py
```

## Citation

```
@InProceedings{Fan_2021_CVPR,
    author    = {Fan, Mingyuan and Lai, Shenqi and Huang, Junshi and Wei, Xiaoming and Chai, Zhenhua and Luo, Junfeng and Wei, Xiaolin},
    title     = {Rethinking BiSeNet for Real-Time Semantic Segmentation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {9716-9725}
}
```

## Acknowledgement

* Segmentation training and evaluation code from [BiSeNet](https://github.com/CoinCheung/BiSeNet).
* Latency measurement from the [Faster-Seg](https://github.com/VITA-Group/FasterSeg).
