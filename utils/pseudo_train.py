# auther  : qian_yu wang
# content :
# data    : 2023/4/9
import os
from torch import nn, tensor
import torch
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from tqdm import tqdm

from resnet14 import MyModel
from utils.utils import acc_for_every_class, save_checkpoint


def pseudo_label(train_label_dataloader, un_label_dataloader, loss_function, device, num_epochs, lr, out, batch_size,
                 test_label_dataloader, checkpoint_pseudo, itrs):
    # 开始MCL训练
    print('------MCL pseudo label train start-------')

    model = MyModel(res=True)
    checkpoint = torch.load('log/exp_11-04-23_0913/checkpoint/model_best_iteration_3.pth.tar',
                            map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-3)
    # # 学习率调整策略 MultiStep：
    # optimizer.load_state_dict(checkpoint['optimizer'])

    model1 = MyModel(cfg=[64, 'M', 128, 'M', 256, 'M', 512, 'M'], res=True)
    # checkpoint1 = torch.load('log/exp_11-04-23_0913/checkpoint/checkpoint_iteration_0.pth.tar',
    #                          map_location=torch.device('cpu'))
    optimizer1 = optim.SGD(model1.parameters(), lr=lr, momentum=0.9, weight_decay=5e-3)
    lr_scheduler1 = optim.lr_scheduler.MultiStepLR(optimizer=optimizer1,
                                                   milestones=[int(num_epochs * 0.56), int(num_epochs * 0.78)],
                                                   gamma=0.1, last_epoch=-1)

    model2 = MyModel(cfg=[64, 'M', 128, 128, 'M', 256, 'M', 512, 'M'], res=True)
    # checkpoint2 = torch.load('log/exp_11-04-23_0913/checkpoint/checkpoint_iteration_1.pth.tar',
    #                          map_location=torch.device('cpu'))
    optimizer2 = optim.SGD(model2.parameters(), lr=lr, momentum=0.9, weight_decay=5e-3)
    lr_scheduler2 = optim.lr_scheduler.MultiStepLR(optimizer=optimizer2,
                                                   milestones=[int(num_epochs * 0.56), int(num_epochs * 0.78)],
                                                   gamma=0.1, last_epoch=-1)

    model3 = MyModel(cfg=[64, 'M', 128, 'M', 256, 256, 'M', 512, 'M'], res=True)
    # checkpoint3 = torch.load('log/exp_11-04-23_0913/checkpoint/checkpoint_iteration_2.pth.tar',
    #                          map_location=torch.device('cpu'))
    optimizer3 = optim.SGD(model3.parameters(), lr=lr, momentum=0.9, weight_decay=5e-3)
    lr_scheduler3 = optim.lr_scheduler.MultiStepLR(optimizer=optimizer3,
                                                   milestones=[int(num_epochs * 0.56), int(num_epochs * 0.78)],
                                                   gamma=0.1, last_epoch=-1)

    model4 = MyModel(cfg=[64, 'M', 128, 'M', 256, 'M', 512, 512, 'M'], res=True)
    # checkpoint4 = torch.load('log/exp_11-04-23_0913/checkpoint/checkpoint_iteration_3.pth.tar',
    #                          map_location=torch.device('cpu'))
    optimizer4 = optim.SGD(model4.parameters(), lr=lr, momentum=0.9, weight_decay=5e-3)
    lr_scheduler4 = optim.lr_scheduler.MultiStepLR(optimizer=optimizer4,
                                                   milestones=[int(num_epochs * 0.56), int(num_epochs * 0.78)],
                                                   gamma=0.1, last_epoch=-1)

    model5 = MyModel(cfg=[64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M'], res=True)
    # checkpoint5 = torch.load('log/exp_11-04-23_0913/checkpoint/checkpoint_iteration_4.pth.tar',
    #                          map_location=torch.device('cpu'))
    optimizer5 = optim.SGD(model5.parameters(), lr=lr, momentum=0.9, weight_decay=5e-3)
    lr_scheduler5 = optim.lr_scheduler.MultiStepLR(optimizer=optimizer5,
                                                   milestones=[int(num_epochs * 0.56), int(num_epochs * 0.78)],
                                                   gamma=0.1, last_epoch=-1)

    # Checkpoint = {'checkpoint1': checkpoint1, 'checkpoint2': checkpoint2, 'checkpoint3': checkpoint3,
    #               'checkpoint4': checkpoint4, 'checkpoint5': checkpoint5}
    Model_five = {'model1': model1, 'model2': model2, 'model3': model3, 'model4': model4, 'model5': model5}
    Optimizer_five = {'optimizer1': optimizer1, 'optimizer2': optimizer2, 'optimizer3': optimizer3,
                      'optimizer4': optimizer4, 'optimizer5': optimizer5}
    # MultiStepLR
    Lr_scheduler_five = {'lr_scheduler1': lr_scheduler1, 'lr_scheduler2': lr_scheduler2, 'lr_scheduler3': lr_scheduler3,
                         'lr_scheduler4': lr_scheduler4, 'lr_scheduler5': lr_scheduler5}
    # for i in range(5):
    #     Checkpoint['checkpoint' + str(i + 1)] = torch.load(
    #         'log/exp_11-04-23_0913/checkpoint/checkpoint_iteration_' + str(i) + '.pth.tar',
    #         map_location=torch.device('cpu'))
    #     Model_five['model' + str(i + 1)].load_state_dict(Checkpoint['checkpoint' + str(i + 1)]['state_dict'])
    #     Optimizer_five['optimizer' + str(i + 1)].load_state_dict(Checkpoint['checkpoint' + str(i + 1)]['optimizer'])
    #     # 学习率调整策略 MultiStep：
    #     Lr_scheduler_five['lr_scheduler' + str(i + 1)].load_state_dict(
    #         Checkpoint['checkpoint' + str(i + 1)]['scheduler'])
    #
    # for i in range(5):
    #     print(Model_five['model' + str(i + 1)])
    #     print(Optimizer_five['optimizer' + str(i + 1)])
    #     print(Lr_scheduler_five['lr_scheduler' + str(i + 1)])

    threshold = 0.755
    print(threshold)

    for itr in range(itrs):
        threshold += 0.025
        for epoch in range(num_epochs):
            cnt_for_true = 0
            cnt_for_train = 0
            iter_list = [iter(train_label_dataloader), iter(un_label_dataloader)]
            # 迭代整个训练集
            for i in range(len(iter_list)):
                iter_dataloader = iter_list[i]
                # for images, target in iter_dataloader:
                for step, (inputs, targets) in enumerate(tqdm(iter_dataloader)):
                    # 测试，单纯打标签

                    model.eval()
                    inputs = inputs.to(device)
                    targets = targets.to(device)

                    outputs = model(inputs)

                    out_prob = F.softmax(outputs, dim=1)  # for selecting positive pseudo-labels

                    # 返回最大概率和其类别
                    max_value, max_idx = torch.max(out_prob, dim=1)
                    # print('max_value:', max_value, 'max_idx:', max_idx, 'targets:', targets)
                    # max_value, max_idx: torch.Size([batch_size])
                    # 取Tensor中符合条件的坐标 维度不变 其中元素根据条件变化 这里是一维
                    select_idx = torch.where(max_value > threshold)[0]
                    if select_idx.numel() > 0:
                        # 所有用于训练的标签
                        cnt_for_train += select_idx.numel()
                        # 剔除不满足的idx
                        max_idx = torch.index_select(max_idx, 0, select_idx)
                        targets = torch.index_select(targets, 0, select_idx)
                        inputs = torch.index_select(inputs, 0, select_idx)

                        Model_five['model' + str(itr + 1)].to(device)
                        Model_five['model' + str(itr + 1)].train()
                        inputs = inputs.to(device)
                        outputs = Model_five['model' + str(itr + 1)](inputs)
                        cnt_for_true += torch.eq(max_idx, targets).sum().item()

                        loss = loss_function(outputs, targets)
                        # loss = loss1 / accumulation_steps
                        Optimizer_five['optimizer' + str(itr + 1)].zero_grad()
                        loss.backward()
                        Optimizer_five['optimizer' + str(itr + 1)].step()
            if (epoch + 1) % 20 == 0:
                with open(os.path.join(out, 'cnt.txt'), 'a+') as f1:
                    f1.write(f'##################### MCL Itr: {itr + 1}-Epoch: {epoch + 1}#####################\n')
                    f1.write(f'cnt_for_train: {cnt_for_train}, '
                             f'cnt_for_ture: {cnt_for_true}, cnt_for_false: {cnt_for_train - cnt_for_true}\n')
                    f1.write(f'scheduler:{Lr_scheduler_five["lr_scheduler" + str(itr + 1)].get_last_lr()[0]}\n')

            Lr_scheduler_five['lr_scheduler' + str(itr + 1)].step()
        # 计算模型acc
        acc = acc_for_every_class(Model_five['model' + str(itr + 1)], device, test_label_dataloader)
        with open(os.path.join(out, 'acc.txt'), 'a+') as f2:
            f2.write(f'itr:{itr + 1}-model{itr + 1}-acc: {acc}\n')

        save_checkpoint({
            'itr': itr + 1,
            'state_dict': Model_five['model' + str(itr + 1)].state_dict(),
            'optimizer': Optimizer_five['optimizer' + str(itr + 1)].state_dict(),
            'scheduler': Lr_scheduler_five['lr_scheduler' + str(itr + 1)].state_dict(),
        }, False, checkpoint_pseudo, f'model{str(itr + 1)}_threshold{threshold}_batch_size{batch_size}')

    print('------MCL train Finish-------')
