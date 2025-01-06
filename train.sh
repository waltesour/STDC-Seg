# nohup python3 train_bbdd.py >train.log &
# 主干网络 STDCNet813  代表 STDC1
# 主干网络 STDCNet1446 代表 STDC2
# 保存后的参数
# model_maxmIOU50.pth 将验证集resize到原始尺寸 0.5后 验证得到的最好模型
# model_maxmIOU75.pth 将验证集resize到原始尺寸 0.75后 验证得到的最好模型
# model_maxmIOU100.pth 将验证集resize到原始尺寸 1后 验证得到的最好模型
# 
# GPU训练
CUDA_VISIBLE_DEVICES=0 nohup python train_bbdd.py >train.log &
