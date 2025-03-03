import numpy as np
import torch
from matplotlib import pyplot as plt
import porespy as ps
import openpnm as op
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


def porosity(phi):
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
    voids = torch.sum(torch.round(phi) == 0, dim=(-3, -2, -1))
    solids = torch.sum(torch.round(phi) == 1, dim=(-3, -2, -1))
    phi_computed = voids / (voids + solids)

    return phi_computed
def culculate_permeability(fake_images):
    resolution = 4.44e-6
    settings = {'pore_shape': 'pyramid',
                'throat_shape': 'cuboid',
                'pore_diameter': 'equivalent_diameter',
                'throat_diameter': 'inscribed_diameter'}
    # Initialize matrices to store
    k_mD = []
    snow = []
    for i in range((len(fake_images[:, 0, 0, 0]))):
        print(i)
        sample = np.round(fake_images[i])
        phi = porosity(torch.from_numpy(sample))
        # Extracting Network
        snow = ps.networks.snow2(sample, voxel_size=resolution)
        pn, geo = op.io.PoreSpy.import_data(snow.network, settings=settings)
        # # pore rate
        # Vol_void = np.sum(pn['pore.volume']) + np.sum(pn['throat.volume'])
        # Vol_bulk = 128 * 128 * 128
        # Poro = Vol_void / Vol_bulk
        # print(f'The value of Porosity is: {Poro:.2f}')

        # Network health
        h = pn.check_network_health()
        op.topotools.trim(network=pn, pores=h['trim_pores'])

        # Adding phase
        water = op.phases.Water(network=pn)

        # conductance
        water.add_model(propname='throat.hydraulic_conductance',
                        model=op.models.physics.hydraulic_conductance.classic_hagen_poiseuille)

        # Stokes
        perm = op.algorithms.StokesFlow(network=pn, phase=water)
        perm.set_value_BC(pores=pn.pores('zmax'), values=0)
        perm.set_value_BC(pores=pn.pores('zmin'), values=1000)
        perm.run()
        water.update(perm.results())

        # Permeability
        Q = np.absolute(perm.rate(pores=pn.pores('zmin')))
        A = (sample.shape[0] * sample.shape[1]) * resolution ** 2
        L = sample.shape[2] * resolution
        mu = water['pore.viscosity'].max()
        delta_P = 1000 - 0

        K = Q * L * mu / (A * delta_P)
        K = K / 0.98e-12 / 1000
        k_mD = np.concatenate((k_mD, K), axis=0)
        # print('--------- Completed perm for window size:', j)

        del snow, pn, geo, water, perm, sample, h, Q, A, L, mu, delta_P, K

    return k_mD

if __name__ == "__main__":
    loaded_data = np.load("real_rock/" + str(0) + ".0.npy")
    real_4d = np.empty((0, loaded_data.shape[0], loaded_data.shape[1], loaded_data.shape[2]), dtype=loaded_data.dtype)
    for i in range(0,32):
    # 加载数据文件 "data.npy"
        loaded_data = np.load("real_rock/"+str(i)+".0.npy")
        # 将 loaded_data 添加到四维数组中
        real_4d = np.concatenate((real_4d, loaded_data[np.newaxis, :, :, :]), axis=0)

    loaded_data = np.load("fake_rocks/" + str(0) + ".0.npy")
    fake_4d = np.empty((0, loaded_data.shape[0], loaded_data.shape[1], loaded_data.shape[2]), dtype=loaded_data.dtype)
    for i in range(0, 32):
        # 加载数据文件 "data.npy"
        loaded_data = np.load("fake_rocks/" + str(i) + ".0.npy")
        # 将 loaded_data 添加到四维数组中
        fake_4d = np.concatenate((fake_4d, loaded_data[np.newaxis, :, :, :]), axis=0)

    # 将所有非 -1 的值转为 1，所有 -1 的值转为 0
    real_4d_processed = np.where(real_4d != -1, 1, 0)
    # 将所有非 -1 的值转为 1，所有 -1 的值转为 0
    fake_4d_processed = np.where(fake_4d != -1, 1, 0)
    perm_real = culculate_permeability(real_4d_processed)
    perm_fake = culculate_permeability(fake_4d_processed)

    # 设置字体参数
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.hist(perm_fake, bins=6, color='skyblue', edgecolor='black', alpha=0.7, label='real')
    plt.hist(perm_real, bins=6, color='salmon', edgecolor='black', alpha=0.7, label='fake')
    plt.axvline(x=np.mean(perm_fake), color='skyblue', linestyle='dashed', linewidth=3, label='x = Average Permeability of fake',clip_on=True)
    plt.axvline(x=np.mean(perm_real), color='salmon', linestyle='dashed', linewidth=3, label='x = Average Permeability of real',clip_on=True)
    print(np.mean(perm_real))
    print(np.mean(perm_fake))
    plt.xlabel('Permeability [mD]')
    plt.ylabel('sample numbers')
    plt.tight_layout()
    plt.legend()
    plt.show()
    plt.savefig('plot.png')