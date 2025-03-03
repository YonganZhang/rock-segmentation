import os

import numpy as np
import torch
from torch.utils.data import Dataset
from utils import *
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor()
])


class MyDataset(Dataset):
    def __init__(self, path,encoder_):
        self.path = path
        self.name = os.listdir(os.path.join(path, '标签'))
        self.encoder = encoder_
    def __len__(self):
        return len(self.name)

    def __getitem__(self, index):
        segment_name = self.name[index]  # xx.png
        image_path = os.path.join(self.path, '样本', segment_name.replace('.jpg', '.png'))
        segment_path = os.path.join(self.path, '标签', segment_name)
        image = keep_image_size_open_rgb(image_path)
        segment_image = keep_image_size_open(segment_path, self.encoder)
        image_path2 = keep_image_size_open_rgb(segment_path)
        image , image2, image3 = transform(image), transform(image_path2), transform(segment_image)
        # 在第一个维度上叠加
        image25 = torch.cat((image, image2), dim=0)
        c = torch.add(image, image2)
        return c, image,image3


if __name__ == '__main__':
    #先建立独热编码对象
    encoder = ColorEncoder()
    encoder.fit(Image.open('data/样本/image1.png'))  # 从图像中获取唯一颜色
    # 保存编码器
    save_encoder(encoder, 'params/encoder2.pkl')
    # 可以使用 PIL 或其他库来保存或显示解码后的图像
    data = MyDataset('data',encoder)
    print(data[0][0].shape)
    print(data[0][1].shape)

