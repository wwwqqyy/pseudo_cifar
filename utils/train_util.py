import torch
from tqdm import tqdm


def train_epoch(model, loss_function, train_label_dataloader, optimizer, device, batch_size):
    model.to(device)

    train_data_size = len(train_label_dataloader)

    model.train()

    train_loss = 0.0
    train_acc = 0.0

    for batch_idx, (inputs, labels) in enumerate(tqdm(train_label_dataloader)):
        # 128,3,32,32
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)

        loss = loss_function(outputs, labels)

        # 因为这里梯度是累加的，所以每次记得清零
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        # inputs.size(0)表示第0维度的数据数量,128
        # train_loss += loss.item() * inputs.size(0)

        ret, pred = torch.max(outputs.data, dim=1)
        acc = pred.eq(labels.data).cpu().sum()
        # batch-size = inputs.size(0)

        train_loss += loss.item()
        train_acc += acc.item()

    avg_train_loss = train_loss * 1.0 / train_data_size
    avg_train_acc = train_acc * 100.0 / train_data_size / batch_size
    return avg_train_acc, avg_train_loss
