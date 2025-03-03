import tqdm
from torch import optim
from torch.utils.data import DataLoader
from data import *
from net import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weight_path = 'params/100.PTH'
weight_path_2 = 'params'
data_path = r'data'
save_path = 'train_image'
encoder = load_encoder("params/encoder.pkl")
# 初始化列表来存储训练损失
train_losses = []


def my_loss_function(output, target):
    loss_fun = nn.CrossEntropyLoss()
    # 假设输入张量为 tensor_input 形状为 (8, 11, 256, 256)
    # 将one-hot编码转换为普通编码
    mask = F.one_hot(output.argmax(dim=1), num_classes=11)
    mask = mask.permute(0, 3, 1, 2)
    output_class = output * mask
    output_sum = torch.sum(output_class, dim=(-1, -2))
    rate = output_sum / (mask.size(-1) * mask.size(-1))
    value = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0.1]  # 假设每个元素都是 [1, 2, 3, 4, 5]
    ture_rate = torch.tensor([value] * rate.size(0)).to(device)
    class_loss = loss_fun(rate, ture_rate)
    total_loss = class_loss + loss_fun(output, target)
    # print(f'{epoch}-{i}---train_loss===>>{total_loss.item()}---class_loss===>>{class_loss.item()}')
    # print(f'{epoch}-{i}--class_loss===>>{class_loss.item()}')
    # totoal_loss = loss_fun(output, target)
    # z_output = output_class[1,:,1,1].detach().cpu().numpy()
    # encoded = rate.detach().cpu().numpy()
    # tensor_output = tensor_output.detach().cpu().numpy()
    return class_loss

# 本代码是正常Unet，训练输出为11通道，用不了
if __name__ == '__main__':
    num_classes = 10 + 1  # +1是背景也为一类
    data_loader = DataLoader(MyDataset(data_path, encoder), batch_size=8, shuffle=False)
    net = UNet(num_classes).to(device)
    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path))
        print('successful load weight！')
    else:
        print('not successful load weight')

    opt = optim.Adam(net.parameters(), lr=1e-3)

    epoch = 1
    while epoch < 200:
        for i, (image, segment_image) in enumerate(tqdm.tqdm(data_loader)):
            image, segment_image = image.to(device), segment_image.to(device)
            out_image = net(image)
            train_loss = my_loss_function(out_image, segment_image)
            opt.zero_grad()
            train_loss.backward()
            opt.step()
            print(f'{epoch}-{i}--class_loss===>>{train_loss.item()}')
            _image = image[0].cpu().numpy()
            _segment_image = encoder.decode(segment_image[0])
            _out_image = encoder.decode(out_image[0])
            # 垂直方向拼接
            _image = np.swapaxes(_image, 0, 2) * 255
            img_vstack = np.vstack((_segment_image, _image, _out_image))
            # 保存垂直拼接图像
            img_vstack = cv2.cvtColor(img_vstack, cv2.COLOR_BGR2RGB)
            cv2.imwrite(f'{save_path}/{(epoch) * i + 1}.png', img_vstack)
            # 记录当前迭代的损失
            train_losses.append(train_loss.item())
        if epoch % 5 == 0:
            torch.save(net.state_dict(), f'{weight_path_2}/{epoch}_train1.pth')
            print('save successfully!')
            # 绘制训练损失曲线
            # 创建一个新的figure
            fig = plt.figure(figsize=(10, 6))
            plt.plot(train_losses)
            plt.title('Training Loss Curve')
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            # 保存图像到本地
            fig.savefig('result/train_loss_curve.tif', dpi=300, bbox_inches='tight')
        epoch += 1
