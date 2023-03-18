from PIL import Image
import numpy as np
import torch

# 保存灰度图像
def save_gray_image(tensor, file_path):
    # 将tensor转换为numpy数组
    array = tensor.squeeze().cpu().numpy()
    # 将像素值缩放到0-255之间
    array = (array * 255).astype(np.uint8)
    # 创建图像对象
    image = Image.fromarray(array, mode='L')
    # 保存图像
    image.save(file_path)

# 保存彩色图像
def save_color_image(tensor, file_path):
    # 将tensor转换为numpy数组
    array = tensor.squeeze().cpu().numpy()
    # 将像素值缩放到0-255之间
    array = (array * 255).astype(np.uint8)
    # 转换为RGB顺序
    array = np.transpose(array, (1, 2, 0))
    # 创建图像对象
    image = Image.fromarray(array, mode='RGB')
    # 保存图像
    image.save(file_path)

# 创建1*1*1024*1024的灰度图像tensor
gray_tensor = torch.rand(1, 1, 1024, 1024)
# 保存为灰度图像
save_gray_image(gray_tensor, 'gray_image.jpg')

# 创建1*3*1024*1024的彩色图像tensor
color_tensor = torch.rand(1, 3, 1024, 1024)
# 保存为彩色图像
save_color_image(color_tensor, 'color_image.jpg')
