import tqdm
from torch import optim
from torch.utils.data import DataLoader
from data_2 import *
from net import *
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weight_path = r'params'
weight_path_2 = r'params/11111.pth'
data_path = r'data'
data_path_2 = r'data'
save_path = 'train_image'
encoder = load_encoder("params/encoder.pkl")


def find_rate(output):
    mask = F.one_hot(output.argmax(dim=1), num_classes=11)
    mask = mask.permute(0, 3, 1, 2)
    output_class = output * mask
    output_sum = torch.sum(output_class, dim=(-1, -2))
    rate = output_sum / (mask.size(-1) * mask.size(-1))
    return rate


def my_loss_function(output, target, flag = 0):
    if flag == 1:
        loss_fun = nn.CrossEntropyLoss()
        # 假设输入张量为 tensor_input 形状为 (8, 11, 256, 256)
        # 将one-hot编码转换为普通编码
        rate = find_rate(output)
        ture_rate = find_rate(target)
        # value = [1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]  # 假设每个元素都是 [1, 2, 3, 4, 5]
        # ture_rate = torch.tensor([value] * rate.size(0)).to(device)
        ture_rate[:, 7] = 1
        KE_loss = loss_fun(rate, ture_rate)
        class_loss = loss_fun(output, target)
        class_loss = class_loss + loss_fun(output, target)
        print(f'{epoch}-{i}---class_loss===>>{class_loss.item()}---KE_loss===>>{KE_loss.item()}')
        # totoal_loss = loss_fun(output, target)
        # z_output = output_class[1,:,1,1].detach().cpu().numpy()
        # encoded = rate.detach().cpu().numpy()
        # tensor_output = tensor_output.detach().cpu().numpy()
    else:
        loss_fun = nn.CrossEntropyLoss()
        class_loss = loss_fun(output, target)
        print(f'{epoch}-{i}---class_loss===>>{class_loss.item()}')
    return class_loss


def my_model(net):
    ## 设置参数--------------------------------------------------------------------------------------------------------------------------------
    _input = 'data'
    encoder = load_encoder("params/encoder2.pkl")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_loader = DataLoader(MyDataset(_input, encoder), batch_size=1, shuffle=False)
    # 遍历DataLoader
    for i, (image, _, segment_image) in enumerate(data_loader):
        # 将输入和目标数据移动到相同设备(CPU或GPU)
        image, segment_image = image.to(device), segment_image.to(device)
        # 前向传播获取预测结果
        outputs = net(image)
        break
    return outputs


# 初始化列表来存储训练损失
train_losses = []
# 本代码是论文的正常Unet，训练输出为3通道（而非11通道），但输入标签
if __name__ == '__main__':
    num_classes = 10 + 1  # +1是背景也为一类
    data_loader = DataLoader(MyDataset(data_path_2, encoder), batch_size=16, shuffle=False)
    net = UNet(num_classes).to(device)
    if os.path.exists(weight_path_2):
        # net.load_state_dict(torch.load(weight_path_2))
        print('successful load weight！')
    else:
        print('not successful load weight')
    opt = optim.Adam(net.parameters())

    epoch = 1
    while epoch < 200:
        for i, (_, image, segment_image) in enumerate(tqdm.tqdm(data_loader)):
            image, segment_image = image.to(device), segment_image.to(device)
            out_image = net(image)
            train_loss = my_loss_function(out_image, segment_image)
            opt.zero_grad()
            train_loss.backward()
            opt.step()
            out_image2 = my_model(net)
            _out_image = encoder.decode(out_image2[0])
            # 垂直方向拼接
            img_vstack = _out_image
            # 保存垂直拼接图像
            img_vstack = cv2.cvtColor(img_vstack, cv2.COLOR_BGR2RGB)
            img_vstack = img_vstack.transpose((1, 0, 2))
            cv2.imwrite(f'{save_path}/{i}-{epoch}.png', img_vstack)

            # 记录当前迭代的损失
            train_losses.append(train_loss.item())

            if i % 1 == 0:
                torch.save(net.state_dict(), f'{weight_path}/常规模型3号_{epoch}_{i}.pth')
                # print('save successfully!')
                # 绘制训练损失曲线
                # 创建一个新的figure
                # fig = plt.figure(figsize=(10, 6))
                # plt.plot(train_losses)
                # plt.title('Training Loss Curve')
                # plt.xlabel('Iteration')
                # plt.ylabel('Loss')
                # 保存图像到本地
                # fig.savefig(f'data/测试集/{epoch}.tif', dpi=300, bbox_inches='tight')
        epoch += 1
