import pickle

import cv2
import numpy as np
import torch
from PIL import Image
import pyvista as pv
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


def keep_image_size_open(path, encoder, size=(256, 256)):
    img = Image.open(path)
    # temp = max(img.size)
    # mask = Image.new('RGB', (temp, temp), (0, 0, 0))
    # mask.paste(img, (0, 0))
    # mask = mask.resize(size)
    encoded_image = encoder.encode(img)
    return encoded_image

def keep_image_size_open2(path, size=(256, 256)):
    img = Image.open(path)
    # temp = max(img.size)
    # mask = Image.new('RGB', (temp, temp), (0, 0, 0))
    # mask.paste(img, (0, 0))
    # mask = mask.resize(size)
    return img

def keep_image_size_open_rgb(path, size=(256, 256)):
    img = Image.open(path)
    temp = max(img.size)
    mask = Image.new('RGB', (temp, temp))
    mask.paste(img, (0, 0))
    mask = mask.resize(size)
    return mask


class ColorEncoder:
    def __init__(self):
        self.color_map = {}
        self.color_list = []

    def fit(self, image):
        # 打开图像并获取唯一颜色
        img = image
        img_colors = img.getcolors(maxcolors=len(img.getdata()) // 3)
        unique_colors = [rgb_value for count, rgb_value in img_colors]

        # 创建颜色映射和列表
        self.color_map = {color: idx for idx, color in enumerate(unique_colors)}
        self.color_list = list(self.color_map.keys())
        a = 1

    def encode(self, image):
        # 将整个图像转换为一个热编码形式
        img = image
        width, height = img.size
        encoded_image = np.zeros((height, width, len(self.color_list)), dtype=np.float32)
        for y in range(height):
            for x in range(width):
                rgb_value = img.getpixel((x, y))
                if rgb_value not in self.color_map:
                    raise ValueError(f"Color {rgb_value} not found in the color map.")
                index = self.color_map[rgb_value]
                encoded_image[y, x, index] = 1.0
        # img.show()
        # aaa=self.decode(torch.from_numpy(encoded_image))
        # # 将tensor转换为grid图像
        # grid = torchvision.utils.make_grid(aaa, nrow=2)
        #
        # # 显示grid图像
        # plt.imshow(grid.permute(1, 2, 0))
        # plt.show()
        return encoded_image

    def decode(self, encoded_image):
        # 将一个热编码形式转换回RGB图像
        num_colors, height, width = encoded_image.shape
        decoded_image = np.zeros((height, width, 3), dtype=np.uint8)
        for y in range(height):
            for x in range(width):
                target = encoded_image[:, y, x]
                # 获取最大值及其索引
                max_value, color_index = torch.max(target, dim=0)
                rgb_value = self.color_list[color_index]
                decoded_image[x, y, :] = rgb_value
        return decoded_image

    def get_color_map(self,color_index):
        rgb_value = self.color_list[color_index]
        return rgb_value

    def get_color_all(self):
        rgb_value = np.zeros(len(self.color_list))
        for i in range(len(self.color_list)):
            rgb_value[i] = self.color_list[i]
        return rgb_value

# 保存ImageEncoder对象
def save_encoder(encoder, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(encoder, f)


# 加载ImageEncoder对象
def load_encoder(file_path):
    with open(file_path, 'rb') as f:
        encoder = pickle.load(f)
    return encoder


if __name__ == '__main__':
    # 读取图片序列
    image_sequence = []
    num_images = 256
    for i in range(1, num_images + 1):  # num_images是你图片序列的长度
        name_pic = (f'data/数字岩心数据库/rock_01_slice_{str(i).zfill(2)}.png')  # 替换成你的图片路径和命名规则
        ima = keep_image_size_open_rgb(name_pic)
        ima = np.array(ima)
        image_sequence.append(ima)
    # 将图片序列转换成三维体
    volume = np.stack(image_sequence, axis=2)
    # 假设 imgs 是你的图像序列组，维度为 (50, 256, 256, 3)
    # 将彩色图像转换为灰度图像
    gray_imgs = np.squeeze(volume[:,:,:,0])
    # 可视化三维体
    # 使用ipyvolume进行三维可视化
    # 创建结构化网格
    grid = pv.ImageData()

    # 设置网格的尺寸
    grid.dimensions = tuple(x + 1 for x in gray_imgs.shape)

    # 将矩阵数据展平并存储为单元数据（cell data）
    grid.cell_data["values"] = gray_imgs.flatten(order="F")

    # 对网格进行阈值处理
    threshed = grid.threshold(value=0.5, invert=True)

    # 创建绘图窗口
    plotter = pv.Plotter()

    # 添加阈值处理后的网格对象
    # plotter.add_mesh(threshed, color='lightskyblue', interpolate_before_map=True, show_edges=0, smooth_shading=True, metallic=1)
    plotter.add_mesh(grid, scalars='values', cmap='gray')
    # 设置窗口大小
    plotter.window_size = [800, 600]

    # 显示图形
    plotter.show()
