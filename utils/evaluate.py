import torch
from tqdm import tqdm
import torch.nn.functional as F


def test(model, loss_function, test_loader, device, batch_size):
    model.to(device)
    valid_data_size = len(test_loader)
    valid_loss = 0.0
    valid_acc = 0.0
    with torch.no_grad():
        model.eval()
        # for _, (inputs, labels) in enumerate(test_loader):
        for _, (inputs, labels) in enumerate(tqdm(test_loader)):
            inputs = inputs.to(device)
            labels = labels.to(device)
            # inputs: shape: torch.Size([8, 3, 32, 32])
            # print('inputs:', inputs, 'shape:', inputs.shape)
            outputs = model(inputs)

            # 一个batch-size的loss
            loss = loss_function(outputs, labels)
            # print('loss:', loss, 'shape:', loss.shape)
            # view_as返回和原tensor数据个数相同，但size不同的tensor
            # correct_counts = predictions.eq(labels.data.contiguous().view_as(predictions))

            #         print(outputs.shape)
            # 二维tensor shape: torch.Size([8, 10]) 10是一共10个类别的输出
            # out_prob = F.softmax(outputs, dim=1)  # for selecting positive pseudo-labels
            # # print('out_prob:', out_prob, 'shape:', out_prob.shape)
            #
            # # 返回最大概率和其类别 value, idx: shape: torch.Size([8])
            # max_value, max_idx = torch.max(out_prob, dim=1)
            # # print('max_value:', max_value, 'max_idx:', max_idx)
            # select_idx = torch.where(max_value > 0.75)[0]
            # print('select_idx', select_idx)
            #
            # max_idx = torch.index_select(max_idx, 0, select_idx)
            # labels = torch.index_select(labels, 0, select_idx)
            # inputs = torch.index_select(inputs, 0, select_idx)
            # print('max_idx:', max_idx, 'shape:', max_idx.shape)
            # print('labels:', labels, 'shape:', labels.shape)
            # print('inputs:', inputs, 'shape:', inputs.shape)
            # ret shape: torch.Size([8]) 最大值，并不是概率
            ret, pred = torch.max(outputs.data, dim=1)
            # .data是把Variable里的tensor取出来
            acc = pred.eq(labels.data).cpu().sum()
            # print('labels:', labels, 'labels-data:', labels.data)
            valid_loss += loss.item()
            valid_acc += acc.item()

        avg_valid_loss = valid_loss * 1.0 / valid_data_size
        avg_valid_acc = valid_acc * 100.0 / valid_data_size / batch_size

    return avg_valid_acc, avg_valid_loss
