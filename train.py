import glob
import os
from datetime import datetime
import numpy as np
import torch.optim as optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F
from resnet14 import MyModel
from utils.ensemble_train import ensemble
from utils.evaluate import test
from utils.pseudo_train import pseudo_label
from utils.train_util import train_epoch
from utils.utils import *


def lbl_un_lbl_split(labels, num_lbl, n_class):
    lbl_per_class = num_lbl // n_class
    labels = np.array(labels)
    lbl_idx = []
    un_lbl_idx = []
    for i in range(n_class):
        idx = np.where(labels == i)[0]
        np.random.shuffle(idx)
        lbl_idx.extend(idx[:lbl_per_class])
        un_lbl_idx.extend(idx[lbl_per_class:])
    return lbl_idx, un_lbl_idx


if __name__ == '__main__':
    # 数据集设置
    # 使用torchvision可以很方便地下载Cifar10数据集，而torchvision下载的数据集为[0,1]的PILImage格式
    # 我们需要将张量Tensor归一化到[-1,1]
    norm_mean = [0.485, 0.456, 0.406]  # 均值
    norm_std = [0.229, 0.224, 0.225]  # 方差
    train_transform_cifar = transforms.Compose([transforms.ToTensor(),  # 将PILImage转换为张量
                                                # 将[0,1]归一化到[-1,1]
                                                transforms.Normalize(norm_mean, norm_std),
                                                transforms.RandomHorizontalFlip(),  # 随机水平镜像
                                                transforms.RandomErasing(scale=(0.04, 0.2), ratio=(0.5, 2)),  # 随机遮挡
                                                transforms.RandomCrop(32, padding=4)  # 随机中心裁剪
                                                ])

    test_transform_cifar = transforms.Compose([transforms.ToTensor(),
                                               transforms.Normalize(norm_mean, norm_std)])
    # 超参数：
    num_epochs = 200
    batch_size_train = 128
    batch_size_test = 100
    random_seed = 1
    num_workers = 4
    lr = 0.01
    test_freq = 10
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
    else:
        n_gpu = 0
    out = './log'
    run_started = datetime.today().strftime('%d-%m-%y_%H%M')
    exp_name = f'exp_{run_started}'
    out = os.path.join(out, exp_name)

    device = torch.device('cuda:3') if torch.cuda.is_available() else torch.device('cpu')
    # args.device = device

    print('n_gpu:', n_gpu)
    print('device:', device)

    train_set_cifar = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                   download=True, transform=train_transform_cifar)
    test_set_cifar = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                  download=True, transform=test_transform_cifar)

    # Get 400 idx of each class randomly
    train_lbl_idx, train_un_lbl_idx = lbl_un_lbl_split(train_set_cifar.targets, 1000, 10)

    # Create a Subset of the dataset with the selected samples
    label_subset_cifar = torch.utils.data.Subset(train_set_cifar, train_lbl_idx)
    un_label_subset_cifar = torch.utils.data.Subset(train_set_cifar, train_un_lbl_idx)

    # 创建dataloader
    train_label_dataloader_cifar = torch.utils.data.DataLoader(label_subset_cifar, batch_size=batch_size_train,
                                                               shuffle=True,
                                                               num_workers=num_workers)
    un_label_dataloader_cifar = torch.utils.data.DataLoader(un_label_subset_cifar, batch_size=batch_size_train,
                                                            shuffle=True,
                                                            num_workers=num_workers)
    test_label_dataloader_cifar = torch.utils.data.DataLoader(test_set_cifar, batch_size=batch_size_test, shuffle=False,
                                                              num_workers=num_workers, drop_last=True)

    label_data_size = len(label_subset_cifar)
    unlabel_data_size = len(un_label_subset_cifar)

    print('label_size: {:4d}  un_label_size:{:4d}'.format(label_data_size, unlabel_data_size))

    train_transform_mnist = transforms.Compose([
        transforms.Resize([32, 32]),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_transform_mnist = transforms.Compose([
        transforms.Resize([32, 32]),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_set_mnist = torchvision.datasets.MNIST(root='./data', train=True,
                                                 download=True, transform=train_transform_mnist)
    test_set_mnist = torchvision.datasets.MNIST(root='./data', train=False,
                                                download=True, transform=test_transform_mnist)

    # Get 400 idx of each class randomly
    train_lbl_idx, train_un_lbl_idx = lbl_un_lbl_split(train_set_mnist.targets, 1000, 10)

    # Create a Subset of the dataset with the selected samples
    label_subset_mnist = torch.utils.data.Subset(train_set_mnist, train_lbl_idx)
    un_label_subset_mnist = torch.utils.data.Subset(train_set_mnist, train_un_lbl_idx)

    # 创建dataloader
    train_label_dataloader_mnist = torch.utils.data.DataLoader(label_subset_mnist, batch_size=batch_size_train,
                                                               shuffle=True,
                                                               num_workers=num_workers)
    un_label_dataloader_mnist = torch.utils.data.DataLoader(un_label_subset_mnist, batch_size=batch_size_train,
                                                            shuffle=True,
                                                            num_workers=num_workers)
    test_label_dataloader_mnist = torch.utils.data.DataLoader(test_set_mnist, batch_size=batch_size_test, shuffle=False,
                                                              num_workers=num_workers)

    model = MyModel(res=True, in_channel=3)
    checkpoint = torch.load('log/exp_11-04-23_0913/checkpoint/model_best_iteration_3.pth.tar',
                            map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    # 定义损失函数和优化器
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-3)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                  milestones=[int(num_epochs * 0.56), int(num_epochs * 0.78)],
                                                  gamma=0.1, last_epoch=-1)

    # # 基于labeled训练一个初始模型
    # train_label_dataloader = train_label_dataloader_mnist
    # test_label_dataloader = test_label_dataloader_mnist
    # print('------label train start-------')
    # checkpoint_regular = os.path.join(out, 'checkpoint_regular')
    # if not os.path.exists(checkpoint_regular):
    #     os.makedirs(checkpoint_regular)
    # # 开始训练有标签数据
    # best_acc = 0.0
    # for epoch in range(num_epochs):
    #     train_acc, train_loss = train_epoch(model, loss_function, train_label_dataloader, optimizer, device,
    #                                         batch_size=batch_size_train)
    #     test_acc = 0.0
    #     if epoch % 10 == 0:
    #         print('train loss: ', train_loss, 'train acc:', train_acc)
    #     if epoch > (num_epochs + 1) / 2 and epoch % test_freq == 0:
    #         test_acc, test_loss = test(model, loss_function, test_label_dataloader, device, batch_size_test)
    #         print('test loss: ', test_loss, 'test acc:', test_acc)
    #         print('lr:', lr_scheduler.get_lr()[0])
    #     elif epoch == (num_epochs - 1):
    #         test_acc, test_loss = test(model, loss_function, test_label_dataloader, device, batch_size_test)
    #         print('test loss: ', test_loss, 'test acc:', test_acc)
    #         print('lr:', lr_scheduler.get_last_lr()[0])
    #     lr_scheduler.step()
    #
    #     is_best = test_acc > best_acc
    #     best_acc = max(test_acc, best_acc)
    #     if is_best:
    #         model_to_save = model.module if hasattr(model, "module") else model
    #         save_checkpoint({
    #             'state_dict': model_to_save.state_dict(),
    #         }, False, checkpoint_regular, f'epoch_{str(epoch + 1)}')
    # with open(os.path.join(out, 'log.txt'), 'a+') as ofile:
    #     ofile.write(f'Last Test Acc: {test_acc}, Best Test Acc: {best_acc}\n')
    # print('------label train Finish-------')
    # 开始伪标签训练多个网络
    # pseudo_label(train_label_dataloader, un_label_dataloader, loss_function, device, num_epochs, lr, out, batch_size,
    #              test_label_dataloader)
    # 多数投票集成
    # test_acc, test_loss = test(model, loss_function, test_label_dataloader_cifar, device, batch_size_test)
    # print('test loss: ', test_loss, 'test acc:', test_acc)
    ensemble(device, test_label_dataloader_mnist, batch_size_test)
