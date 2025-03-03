import numpy as np
import torch
from matplotlib import pyplot as plt
import porespy as ps
import openpnm as op
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def porosity(phi, i):
    r"""
    Computes the porosity of a sample (image, real or generated).
    Here 0 values correspond to the voids, and 1 values correspond to the grains.

    Parameters
    ----------
    phi: input array corresponding to the pixels of the image.


    Returs
    ------
    porosity: float
              Returns the computed porosity, it is equal to the sum of voids (0), over
              the sum of the voids(0) and solids(1)
    """
    voids = torch.sum(torch.round(phi) == i, dim=(-3, -2, -1))
    solids = torch.sum(torch.round(phi) != 1, dim=(-3, -2, -1))
    phi_computed = voids / (voids + solids)

    return phi_computed


def culculate_permeability(fake_images, i1):
    part = []
    for i in range((len(fake_images[:, 0, 0, 0]))):
        sample = np.round(fake_images[i])
        phi = porosity(torch.from_numpy(sample), i1).numpy()
        part.append(phi)
        # part = np.concatenate((part, phi), axis=0)

    return part


if __name__ == "__main__":
    loaded_data = np.load("fake_rocks/" + str(0) + ".0.npy")
    real_4d = np.empty((0, loaded_data.shape[0], loaded_data.shape[1], loaded_data.shape[2]), dtype=loaded_data.dtype)
    for i in range(7, 39):
        # 加载数据文件 "data.npy"
        loaded_data = np.load("fake_rocks/" + str(i) + ".0.npy")
        # 将 loaded_data 添加到四维数组中
        real_4d = np.concatenate((real_4d, loaded_data[np.newaxis, :, :, :]), axis=0)

    loaded_data = np.load("fake_rocks/" + str(0) + ".0.npy")
    fake_4d = np.empty((0, loaded_data.shape[0], loaded_data.shape[1], loaded_data.shape[2]), dtype=loaded_data.dtype)
    for i in range(12, 44):
        # 加载数据文件 "data.npy"
        loaded_data = np.load("fake_rocks/" + str(i) + ".0.npy")
        # 将 loaded_data 添加到四维数组中
        fake_4d = np.concatenate((fake_4d, loaded_data[np.newaxis, :, :, :]), axis=0)

    name = ["Nothing",
            "Sodium feldspar",
            "Dolomite",
            "Hard gypsum",
            "Quartz",
            "Pyrite",
            "Chlorite",
            "Other minerals",
            "Calcite",
            "Illite",
            "rutile",
            "Pore"]
    name2 = [
            "Nothing",
            "Sodium feldspar",
            "Dolomite",
            "-11",
            "Hard gypsum",
        ",11",
            "Pyrite",
            "Rutile",
            ]
    for i in range(8):
        if i == 12:
            i = - 1
        perm_real = culculate_permeability(real_4d, i)
        perm_fake = culculate_permeability(fake_4d, i)

        # 设置字体参数
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.hist(perm_fake, bins=6, color='skyblue', edgecolor='black', alpha=0.7, label='real')
        plt.hist(perm_real, bins=6, color='salmon', edgecolor='black', alpha=0.7, label='fake')
        plt.xlabel(f'{str(name2[i])}, %')
        plt.ylabel('sample numbers')
        plt.tight_layout()
        plt.legend()
        plt.savefig(f'--{i}_pic.png', dpi=300)  # 将图像保存为 PNG 格式，dpi 参数指定图像分辨率
        plt.close()
