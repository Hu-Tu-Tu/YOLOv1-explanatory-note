from yoloData import yoloDataset
from yoloLoss import yoloLoss
from new_resnet import resnet50
from torchvision import models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
import os

os.chdir('/root/workspace/YOLOV1-pytorch')


device = 'cuda'
file_root = 'VOCdevkit/VOC2007/JPEGImages/'
batch_size = 2   # 若显存较大可以调大此参数 4，8，16，32等等
learning_rate = 0.001
num_epochs = 100

train_dataset = yoloDataset(img_root=file_root, list_file='voctrain.txt', train=True, transform=[transforms.ToTensor()])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)# shuffle=True：这意味着数据加载器会打乱数据集中的样本顺序
test_dataset = yoloDataset(img_root=file_root, list_file='voctest.txt', train=False, transform=[transforms.ToTensor()])
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
print('the train_dataset has %d images' % (len(train_dataset)))

"""
因之前定义的网络后面一部分与ResNet50的结构略有差异，所以并不能完全使用使用torchvision的models库中的resnet50导入权重参数。
需要对其权重参数进行一定的筛选。

权重参数导入方法：自己定义的网络以及models库内的网络各自创建一个对象。接着使用state_dict()导入各自的权重参数。
网络结构相同的部分将new_state_dict的值赋给op。但是如果自己定义的网络结构的键值与torch自带的库不一致的话，导入权重参数会稍微麻烦一点。
这里给出了一种解决办法，具体参考代码。

"""
net = resnet50()  # 自己定义的网络
net = net.cuda()
resnet = models.resnet50(pretrained=True)  # torchvison库中的网络
new_state_dict = resnet.state_dict()
op = net.state_dict()

# for i in new_state_dict.keys():   # 查看网络结构的名称 并且得出一共有320个key
#     print(i)

# 若定义的网络结构的key()名称与torchvision库中的ResNet50的key()相同则可以使用此方法
# for k in new_state_dict.keys():
#     # print(k)                    # 输出层的名字
#     if k in op.keys() and not k.startswith('fc'):  # startswith() 方法用于检查字符串是否是以指定子字符串开头，如果是则返回 True，否则返回 False
#         op[k] = new_state_dict[k]  # 与自定义的网络比对 相同则把权重参数导入 不同则不导入
# net.load_state_dict(op)

# 无论名称是否相同都可以使用；enumerate: for循环中经常用到，既可以遍历元素又可以遍历索引
for new_state_dict_num, new_state_dict_value in enumerate(new_state_dict.values()):
    for op_num, op_key in enumerate(op.keys()):
        if op_num == new_state_dict_num and op_num <= 317:  # 320个key中不需要最后的全连接层的两个参数
            op[op_key] = new_state_dict_value
net.load_state_dict(op)  # 更改了state_dict的值记得把它导入网络中

print('cuda', torch.cuda.current_device(), torch.cuda.device_count())   # 确认一下cuda的设备

criterion = yoloLoss(7, 2, 5, 0.5)
criterion = criterion.to(device)
net.train()  # 训练前需要加入的语句

params = []  # 里面存字典
# net网络的参数名称和参数对象的元祖，通过named_parameters()方法获取，返回的事一个字典
params_dict = dict(net.named_parameters()) # 返回各层中key(只包含weight and bias) and value
for key, value in params_dict.items():
    params += [{'params': [value], 'lr':learning_rate}]  # value和学习率相加，其实是append

optimizer = torch.optim.SGD(    # 定义优化器  “随机梯度下降”
    params,   # net.parameters() 为什么不用这个???
    lr=learning_rate,
    momentum=0.9,   # 即更新的时候在一定程度上保留之前更新的方向  可以在一定程度上增加稳定性，从而学习地更快
    weight_decay=5e-4)     # L2正则化理论中出现的概念
# torch.multiprocessing.freeze_support()  # 多进程相关 猜测是使用多显卡训练需要

for epoch in range(num_epochs):
    net.train()
    # 更平滑的衰减，可以考虑使用学习率调度器（如torch.optim.lr_scheduler）
    if epoch == 60:
        learning_rate = 0.0001
    if epoch == 80:
        learning_rate = 0.00001
    for param_group in optimizer.param_groups:   # 其中的元素是2个字典；optimizer.param_groups[0]： 长度为6的字典，包括[‘amsgrad’, ‘params’, ‘lr’, ‘betas’, ‘weight_decay’, ‘eps’]这6个参数；
                                                # optimizer.param_groups[1]： 好像是表示优化器的状态的一个字典；
        param_group['lr'] = learning_rate      # 更改全部的学习率
    print('\n\nStarting epoch %d / %d' % (epoch + 1, num_epochs))
    print('Learning Rate for this epoch: {}'.format(learning_rate))

    # 训练阶段
    total_loss = 0.
    for i, (images, target) in enumerate(train_loader):
        images, target = images.cuda(), target.cuda()
        pred = net(images)# 前向传播
        loss = criterion(pred, target)# 计算损失
        total_loss += loss.item()# 累积损失

        optimizer.zero_grad()# 梯度清零
        loss.backward()# 反向传播
        optimizer.step()# 更新权重
        if (i + 1) % 5 == 0:
            print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f, average_loss: %.4f' % (epoch +1, num_epochs,
                                                                                 i + 1, len(train_loader), loss.item(), total_loss / (i + 1)))
    # 验证阶段，每次训练完成之后，都验证一下模型的准确性，并且保存
    validation_loss = 20.0
    net.eval()# net.eval() 方法用于将模型设置为评估模式（evaluation mode）
    for i, (images, target) in enumerate(test_loader):  # 导入dataloader 说明开始训练了  enumerate 建立一个迭代序列
        images, target = images.cuda(), target.cuda()
        pred = net(images)    # 将图片输入
        loss = criterion(pred, target)
        validation_loss += loss.item()   # 累加loss值  （固定搭配）
    validation_loss /= len(test_loader)  # 计算平均loss

    best_test_loss = validation_loss
    print('get best test loss %.5f' % best_test_loss)
    torch.save(net.state_dict(), 'yolo.pth')


