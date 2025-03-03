import imageio

from utils import *


def one_hot_to_int(arr):
    """
    将独热编码的最后一维转换为整数。
    """
    return np.argmax(arr, axis=-1)  # 因为索引是从0开始的，所以加1得到实际的类别编号


def int_to_one_hot(arr, num_classes):
    """
    将整数转换为独热编码。
    """
    return np.eye(num_classes)[arr - 1]  # 因为类别编号是从1开始的，所以减1得到正确的索引


def get_pore(volume_int):
    # 指定包含TIFF文件的目录
    directory = r'data\孔隙结构/1/'

    # 初始化一个列表来存储所有读取的图像
    image_list = []

    # 图像的起始编号和结束编号
    start_number = 0
    end_number = 255

    # 读取指定范围内的TIFF图像
    for i in range(start_number, end_number + 1):
        # 生成文件名
        filename = f'数字岩心数据库{i:04d}.tif'
        # 读取图像并转换为NumPy数组
        image = imageio.v2.imread(directory + filename)
        # 将图像数组添加到列表中
        image_list.append(image)

    # 将列表转换为四维NumPy数组
    # 假设每个图像具有相同的维度，例如 (height, width, channels)
    volume = np.stack(image_list, axis=0)
    # volume 现在是一个四维数组，其维度应该是 (256, height, width, channels)
    # 使用布尔索引来选择 volume 中的 0 值
    zero_indices = volume == 0

    # 使用布尔索引来赋值，只改变 volume_int 中对应于 volume 中 0 值的位置
    volume_int[zero_indices] = -1
    return volume_int


def get_pore_2(volume_int, i1,flag):
    # 指定包含TIFF文件的目录
    directory = r'data\孔隙结构/2/'

    # 初始化一个列表来存储所有读取的图像
    image_list = []

    # 图像的起始编号和结束编号
    start_number = i1
    end_number = i1 + 255

    # 读取指定范围内的TIFF图像
    for i in range(start_number, end_number + 1):
        # 生成文件名
        filename = f'1 ({i}).tif'
        # 读取图像并转换为NumPy数组
        image = imageio.v2.imread(directory + filename)
        if flag == 1:
            image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_NEAREST)
            if i == start_number + 128:
                break
        # 将图像数组添加到列表中
        image_list.append(image)

    # 将列表转换为四维NumPy数组
    # 假设每个图像具有相同的维度，例如 (height, width, channels)
    volume = np.stack(image_list, axis=0)
    # volume 现在是一个四维数组，其维度应该是 (256, height, width, channels)
    # 使用布尔索引来选择 volume 中的 0 值
    zero_indices = volume == 0

    # 使用布尔索引来赋值，只改变 volume_int 中对应于 volume 中 0 值的位置
    volume_int[zero_indices] = -1
    return volume_int, end_number


def plot_3d_ROCK(volume_int, encoder):
    # 创建结构化网格
    grid = pv.ImageData()
    # 设置网格的尺寸
    grid.dimensions = tuple(x + 1 for x in volume_int.shape)
    # 允许绘制空网格
    pv.global_theme.allow_empty_mesh = True
    # 将矩阵数据展平并存储为单元数据（cell data）
    grid.cell_data["values"] = volume_int.flatten(order="F")
    # 创建绘图窗口
    plotter = pv.Plotter()
    # 对网格进行阈值处理
    # 设置阈值区间
    for i in range(11):
        lower_threshold = i - 0.5  # 假设我们要选择大于或等于1的值
        upper_threshold = lower_threshold + 0.8  # 假设我们要选择小于或等于3的值
        threshed = grid.threshold(value=[lower_threshold, upper_threshold])
        # 添加阈值处理后的网格对象
        color = encoder.get_color_map(i)
        plotter.add_mesh(threshed, color=color, interpolate_before_map=False, show_edges=0, smooth_shading=False, metallic=0)
    # 绘制孔隙
    threshed = grid.threshold(value=[-1.5, -0.5])
    plotter.add_mesh(threshed, color=[0, 0, 0], interpolate_before_map=False, show_edges=0, smooth_shading=False, metallic=0)
    pv.global_theme.allow_empty_mesh = True
    outline = grid.outline()
    plotter.add_mesh(outline, color="k")
    return plotter


def plot_fake_digital_rock(start_images , flag = 0):
    num_images = 256
    image_sequence = []
    encoder = load_encoder("params/encoder2.pkl")
    for i in range(1, num_images + 1):  # num_images是你图片序列的长度
        print(i)
        name_pic = (f'result/{str(start_images + num_images - i)}.png')  # 替换成你的图片路径和命名规则
        ima = keep_image_size_open(name_pic, encoder)
        if flag == 1:
            ima = cv2.resize(ima, (128, 128), interpolation=cv2.INTER_NEAREST)
            if i == 129:
                break
        image_sequence.append(ima)
    # 将图片序列转换成三维体
    volume = np.stack(image_sequence, axis=2)
    # 将最后一维从独热编码转为整数
    volume_int = one_hot_to_int(volume)
    volume_int, pore_num = get_pore_2(volume_int, start_images ,flag)
    np.save(f"fake_rocks/{str(start_images/256)}.npy", volume_int)
    return pore_num

def plot_real_digital_rock(start_images , flag = 0):
    num_images = 256
    image_sequence = []
    encoder = load_encoder("params/encoder2.pkl")
    for i in range(1, num_images + 1):  # num_images是你图片序列的长度
        print(i)
        name_pic = (f'result/{str(start_images + num_images - i)}.png')  # 替换成你的图片路径和命名规则
        ima = keep_image_size_open(name_pic, encoder)
        if flag == 1:
            ima = cv2.resize(ima, (128, 128), interpolation=cv2.INTER_NEAREST)
            if i == 129:
                break
        image_sequence.append(ima)
    # 将图片序列转换成三维体
    volume = np.stack(image_sequence, axis=2)
    # 将最后一维从独热编码转为整数
    volume_int = one_hot_to_int(volume)
    volume_int, pore_num = get_pore_2(volume_int, start_images ,flag)
    np.save(f"real_rock/{str(start_images/256)}.npy", volume_int)
    # plotter = plot_3d_ROCK(volume_int, encoder)
    # # 设置窗口大小
    # plotter.window_size = [1000, 1000]
    # # 显示图形
    # plotter.show()
    return pore_num

if __name__ == '__main__':
    # 读取图片序列s
    # for i in range(0,33):
    #     start_images = 256*i
    #     plot_fake_digital_rock(start_images, 1)
        # plot_real_digital_rock(start_images, 0)
    volume_int = np.load(f"fake_rocks/{str(32.0)}.npy")
    encoder = load_encoder("params/encoder2.pkl")
    plotter = plot_3d_ROCK(volume_int, encoder)
    # 设置窗口大小
    plotter.window_size = [1000, 1000]
    # 显示图形
    plotter.show()


