import glob
import os
import random
import shutil
import sys

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.append('../..')


def save_checkpoint(state, is_best, checkpoint, itr):
    filename = f'checkpoint_{itr}.pth.tar'
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, f'model_best_{itr}.pth.tar'))


def set_seed(seed, n_gpu):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def enable_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()


def calculate_loss(inputs, max_idx, model1, model2, model3, model4, model5, loss_function, device):
    # 用于训练
    model = {'model1': model1, 'model2': model2, 'model3': model3, 'model4': model4, 'model5': model5}
    for i in range(5):
        model['model' + str(i + 1)].to(device)
    for i in range(5):
        model['model' + str(i + 1)].eval()

    inputs = inputs.to(device)
    max_idx = max_idx.to(device)
    outputs = []
    for i in range(5):
        outputs.append(model['model' + str(i + 1)](inputs))

    # print(outputs)
    loss = []
    for i in range(5):
        loss.append(loss_function(outputs[i], max_idx))
    # print(loss)
    return loss


def acc_for_every_class(model, device, test_label_dataloader):
    acc = []
    num_classes = 10  # 类别数目
    # 初始化为 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    class_correct = list(0. for i in range(num_classes))
    class_total = list(0. for i in range(num_classes))
    model.to(device)
    model.eval()
    # test
    # total_correct = 0
    # total_num = 0
    for _, (x, label) in enumerate(tqdm(test_label_dataloader)):
        x, label = x.to(device), label.to(device)
        outputs = model(x)
        pred = outputs.argmax(dim=1)
        # total_correct += torch.eq(pred, label).float().sum().item()
        # total_num += x.size(0)  # 即batch_size

        c = (pred == label).squeeze()
        print('c:', c)
        for i in range(len(label)):
            _label = label[i]
            class_correct[_label] += c[i].item()
            class_total[_label] += 1

    for i in range(num_classes):
        acc.append(class_correct[i] / class_total[i])
    return acc
