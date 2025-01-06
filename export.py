# @Author  : SONG SHUXIANG
# @Time    : 2024/12/27
# @Func    : STDC-Seg语义分割网络转onnx
# @Usage    :
#       修改save_pth_path为训练得到的pth模型路径
#       确认input_names和output_names

import torch
import torch.onnx
import argparse
from models.model_stages import BiSeNet
import os
import onnx
import onnxruntime as ort
import numpy as np

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

parse = argparse.ArgumentParser(description='export STDC model to ONNX.')
parse.add_argument(
        '--local_rank',
        dest = 'local_rank',
        type = int,
        default = -1,
        )
parse.add_argument(
        '--n_classes',
        dest = 'n_classes',
        type = int,
        default = 3,
        )
parse.add_argument(
    '--cropsize',
    dest='cropsize',
    type=str,
    default='2048,2048',
)
parse.add_argument(
        '--n_workers_train',
        dest = 'n_workers_train',
        type = int,
        default = 1,
        )
parse.add_argument(
        '--n_workers_val',
        dest = 'n_workers_val',
        type = int,
        default = 1,
        )
parse.add_argument(
        '--n_img_per_gpu',
        dest = 'n_img_per_gpu',
        type = int,
        default = 4,
        )
parse.add_argument(
        '--max_iter',
        dest = 'max_iter',
        type = int,
        default = 100000,
        )
parse.add_argument(
        '--save_iter_sep',
        dest = 'save_iter_sep',
        type = int,
        default = 1000,
        )
parse.add_argument(
        '--warmup_steps',
        dest = 'warmup_steps',
        type = int,
        default = 1000,
        )      
parse.add_argument(
        '--mode',
        dest = 'mode',
        type = str,
        default = 'train',
        )
parse.add_argument(
        '--ckpt',
        dest = 'ckpt',
        type = str,
        default = None,
        )
parse.add_argument(
        '--respath',
        dest = 'respath',
        type = str,
        default = 'checkpoints/bbdd1211/',
        )
parse.add_argument(
        '--dspth',
        dest = 'dspth',
        type = str,
        default = '/home/inspur/workspace/SONGSX/Dataset/baddall2cls-seg',
        )
parse.add_argument(
        '--backbone',
        dest = 'backbone',
        type = str,
        default = 'STDCNet813',
        )
parse.add_argument(
        '--pretrain_path',
        dest = 'pretrain_path',
        type = str,
        default = 'checkpoints/STDCNet813M_73.91.tar',
        )
parse.add_argument(
        '--use_conv_last',
        dest = 'use_conv_last',
        type = str2bool,
        default = False,
        )
parse.add_argument(
        '--use_boundary_2',
        dest = 'use_boundary_2',
        type = str2bool,
        default = False,
        )
parse.add_argument(
        '--use_boundary_4',
        dest = 'use_boundary_4',
        type = str2bool,
        default = False,
        )
parse.add_argument(
        '--use_boundary_8',
        dest = 'use_boundary_8',
        type = str2bool,
        default = True,
        )
parse.add_argument(
        '--use_boundary_16',
        dest = 'use_boundary_16',
        type = str2bool,
        default = False,
        )

def main():
    args = parse.parse_args()
    model = BiSeNet(backbone=args.backbone, n_classes=args.n_classes, pretrain_model=args.pretrain_path, 
    use_boundary_2=args.use_boundary_2, use_boundary_4=args.use_boundary_4, use_boundary_8=args.use_boundary_8, 
    use_boundary_16=args.use_boundary_16, use_conv_last=args.use_conv_last) #加载模型结构

    save_pth_path = os.path.join(args.respath, 'pths/model_maxmIOU100.pth') #训练得到的pth模型文件
    state_dict = torch.load(save_pth_path,  weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    # 创建一个虚拟输入张量，即一批量为1的RGB图像
#     dummy_input = torch.randn(1, 3, int(args.cropsize.split(',')[0]), int(args.cropsize.split(',')[1]))
    dummy_input = torch.randn(1, 3, 1024,1024)

    # 设置导出的ONNX文件名
    output_onnx_path = os.path.join(args.respath, 'pths/model_maxmIOU100.onnx')

    # 导出模型到ONNX
    torch.onnx.export(model, dummy_input, output_onnx_path,
                    export_params=True,        # 存储已训练参数
                    verbose = False,           # 是否打印详细的ONNX模型信息
                    opset_version=11,          # ONNX操作集版本
                    do_constant_folding=True,  # 是否执行常量折叠优化
                    input_names = ['img'],     # 输入名称
                    output_names = ['feat_out','feat_out16','feat_out32','feat_out_sp8'], # 定义输出名称,输出的数目与网络实际输出对应起来
                    dynamic_axes =  {
                        'img':{0: 'batch_size', 2: 'height', 3: 'width'},  # 输入的批次大小、高度和宽度是动态的
                        'feat_out':{0: 'batch_size', 2: 'height', 3: 'width'}, # 输出的批次大小、高度和宽度是动态的
                        'feat_out16':{0: 'batch_size', 2: 'height', 3: 'width'},
                        'feat_out32':{0: 'batch_size', 2: 'height', 3: 'width'},
                        'feat_out_sp8':{0: 'batch_size', 2: 'height', 3: 'width'}}
    )
    model_onnx = onnx.load(output_onnx_path)  # 加载 ONNX 模型
    onnx.checker.check_model(model_onnx)  # 检查 ONNX 模型

    dummy_input_numpy = dummy_input.numpy()
    # 获取 PyTorch 推理结果
    with torch.no_grad():
        pytorch_outputs = model(dummy_input)
    # 获取 ONNX 推理结果（如上文所示）
    ort_session = ort.InferenceSession(output_onnx_path)

    dummy_input2 = torch.randn(2, 3, 2048,2048)
    dummy_input_numpy2 = dummy_input2.numpy()
    onnx_outputs = ort_session.run(None, {'img': dummy_input_numpy2}) # 要和输入名称匹配起来,开启后dynamic_axes,ort_session.run的输入尺寸动态变化

    # 比较输出
    print("PyTorch output:", pytorch_outputs)
    print("ONNX output:", onnx_outputs)

    # 计算两个输出之间的差异（因为启用了use_boundary参数,模型输出多个张量尺寸不同,所以只取第一个进行使用）
    difference = np.abs(pytorch_outputs[0] - onnx_outputs[0]).max()
    print(f"最大差异: {difference}")


if __name__ == '__main__':
    main()

