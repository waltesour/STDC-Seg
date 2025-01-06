# 将labelme标注的json格式分割数据、转换成png格式的图像

import os
import json
from PIL import Image, ImageDraw
import numpy as np

# 定义函数来处理单个JSON文件
def process_json_file(json_path, output_folder):
    # 读取JSON文件
    with open(json_path, 'r') as f:
        data = json.load(f)

    # 获取图像尺寸
    width = data['imageWidth']
    height = data['imageHeight']

    # 创建空白图像，模式为L表示8位灰度图
    img_array = np.zeros((height, width), dtype=np.uint8)  # 背景初始化为0

    # 定义分割类别的灰度值
    gray_values = {
        "edge": 1,  # 第一类分割区域
        "cable": 2  # 第二类分割区域
    }

    # 绘制多边形
    for shape in data['shapes']:
        points = [(int(x), int(y)) for x, y in shape['points']]
        mask = Image.new('L', (width, height), 0)
        draw = ImageDraw.Draw(mask)     # 创建一个绘图对象draw
        draw.polygon(points, fill=255)  # 使用白色填充多边形，用于掩码
        mask = np.array(mask)
        # 创建布尔索引，找到mask为255且img_array为0的位置
        condition = (mask == 255) & (img_array == 0)
        # 更新img_array中满足条件的位置
        img_array[condition] = gray_values[shape['label']]

    # 将numpy数组转换为PIL图像
    img = Image.fromarray(img_array, mode='L')

    # 构造输出文件路径（与输入JSON文件同名，但扩展名为.png，并保存在指定的输出文件夹中）
    output_filename = os.path.join(output_folder, os.path.splitext(os.path.basename(json_path))[0] + '.png')
    # 保存图像
    img.save(output_filename)

# 指定要处理的文件夹路径
input_folder_path = 'D:\\Datasets\\baddall2cls-seg\\labels'  # 替换为你的输入文件夹路径
# 指定输出文件夹路径
output_folder_path = 'D:\\Datasets\\baddall2cls-seg\\labels_png'  # 替换为你想要保存PNG文件的新文件夹路径

# 确保输出文件夹存在
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

# 遍历文件夹中的所有JSON文件并处理
for filename in os.listdir(input_folder_path):
    if filename.endswith('.json'):
        file_path = os.path.join(input_folder_path, filename)
        process_json_file(file_path, output_folder_path)