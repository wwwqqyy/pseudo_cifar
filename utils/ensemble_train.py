# auther  : qian_yu wang
# content :
# data    : 2023/4/19
from collections import Counter

import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
from resnet14 import MyModel
from utils.utils import acc_for_every_class


def ensemble(device, test_label_dataloader, batch_size):
    print('------ensemble train start-------')

    model1 = MyModel(cfg=[64, 'M', 128, 'M', 256, 'M', 512, 'M'], res=True, in_channel=1)
    checkpoint1 = None

    model2 = MyModel(cfg=[64, 64, 'M', 128, 'M', 256, 'M', 512, 'M'], res=True, in_channel=1)
    checkpoint2 = None

    model3 = MyModel(cfg=[64, 'M', 128, 128, 'M', 256, 'M', 512, 'M'], res=True, in_channel=1)
    checkpoint3 = None

    model4 = MyModel(cfg=[64, 'M', 128, 'M', 256, 256, 'M', 512, 'M'], res=True, in_channel=1)
    checkpoint4 = None

    model5 = MyModel(cfg=[64, 'M', 128, 'M', 256, 'M', 512, 512, 'M'], res=True, in_channel=1)
    checkpoint5 = None


    checkpoint = {'checkpoint1': checkpoint1, 'checkpoint2': checkpoint2, 'checkpoint3': checkpoint3,
                  'checkpoint4': checkpoint4, 'checkpoint5': checkpoint5}
    model_five = {'model1': model1, 'model2': model2, 'model3': model3, 'model4': model4, 'model5': model5}

    for i in range(5):
        checkpoint['checkpoint' + str(i + 1)] = torch.load(
            'log/exp_23-05-23_1008/checkpoint_pseudo/checkpoint_model' + str(i + 1) +
            '_threshold0.78_batch_size128.pth.tar',
            map_location=torch.device('cpu'))

        model_five['model' + str(i + 1)].load_state_dict(checkpoint['checkpoint' + str(i + 1)]['state_dict'])
        # print('model' + str(i + 1) + ':', model_five['model' + str(i + 1)])
        model_five['model' + str(i + 1)].to(device)
        model_five['model' + str(i + 1)].eval()

    models_correct = [0 for i in range(5)]
    pred = []
    vote_correct = 0
    with torch.no_grad():
        # for _, (inputs, labels) in enumerate(test_label_dataloader):
        for _, (inputs, labels) in enumerate(tqdm(test_label_dataloader)):
            inputs = inputs.to(device)
            labels = labels.to(device)
            for i in range(5):
                # inputs: shape: torch.Size([128, 3, 32, 32])
                # print('inputs:', inputs, 'shape:', inputs.shape)
                outputs = model_five['model' + str(i + 1)](inputs)

                _, prediction = torch.max(outputs, 1)  # 按行取最大值
                pre_num = prediction.cpu().numpy()
                models_correct[i] += (pre_num == labels.cpu().numpy()).sum()

                pred.append(pre_num)
            arr = np.array(pred)
            # print(arr)
            pred.clear()
            # arr[:, i] 在不同维度上切片  most_common(1) 返回最多票类别字典
            result = [Counter(arr[:, i]).most_common(1)[0][0] for i in range(batch_size)]
            # print('result:', result)
            vote_correct += (result == labels.cpu().numpy()).sum()
        print("集成的正确率" + str(vote_correct / len(test_label_dataloader) / batch_size))
        for idx, correct in enumerate(models_correct):
            print("网络" + str(idx) + "的正确率为：" + str(correct / len(test_label_dataloader) / batch_size))
