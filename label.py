import io
import base64
from collections import Counter

import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

from resnet14 import MyModel

device = torch.device('cuda:3') if torch.cuda.is_available() else torch.device('cpu')
model1 = MyModel(cfg=[64, 'M', 128, 'M', 256, 'M', 512, 'M'], res=True, in_channel=3)
checkpoint1 = None

model2 = MyModel(cfg=[64, 'M', 128, 128, 'M', 256, 'M', 512, 'M'], res=True, in_channel=3)
checkpoint2 = None

model3 = MyModel(cfg=[64, 'M', 128, 'M', 256, 256, 'M', 512, 'M'], res=True, in_channel=3)
checkpoint3 = None

model4 = MyModel(cfg=[64, 'M', 128, 'M', 256, 'M', 512, 512, 'M'], res=True, in_channel=3)
checkpoint4 = None

model5 = MyModel(cfg=[64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M'], res=True, in_channel=3)
checkpoint5 = None

checkpoint = {'checkpoint1': checkpoint1, 'checkpoint2': checkpoint2, 'checkpoint3': checkpoint3,
              'checkpoint4': checkpoint4, 'checkpoint5': checkpoint5}
model_five = {'model1': model1, 'model2': model2, 'model3': model3, 'model4': model4, 'model5': model5}

for i in range(5):
    checkpoint['checkpoint' + str(i + 1)] = torch.load(
        'log/exp_08-05-23_1127/checkpoint_pseudo/checkpoint_model'+str(i + 1)+'_threshold0.78_batch_size128.pth.tar',
        map_location=torch.device('cpu'))
    model_five['model' + str(i + 1)].load_state_dict(checkpoint['checkpoint' + str(i + 1)]['state_dict'])
    model_five['model' + str(i + 1)].to(device)
    model_five['model' + str(i + 1)].eval()


# Pre-process image
def transform_image(image_bytes):
    norm_mean = [0.485, 0.456, 0.406]  # 均值
    norm_std = [0.229, 0.224, 0.225]  # 方差
    my_transforms = transforms.Compose([transforms.Resize([32, 32]),
                                        transforms.ToTensor(),
                                       transforms.Normalize(norm_mean, norm_std)])
    image = Image.open(io.BytesIO(image_bytes))

    if image.mode != "RGB":
        raise ValueError("Input file is not RGB image...")

    return my_transforms(image)


cifar10_classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def get_label(image_bytes):
    pred = []
    # <class 'PIL.PngImagePlugin.PngImageFile'>
    img = Image.open(io.BytesIO(image_bytes))
    # print(type(img))
    # <class 'PIL.Image.Image'>
    img = img.resize((380, 400), Image.ANTIALIAS)
    # print(type(img))
    # <class 'numpy.ndarray'>
    img = np.array(img)
    # print(type(img))
    image = transform_image(image_bytes)
    image = image.reshape(1, 3, 32, 32)
    for j in range(5):
        outputs = model_five['model' + str(i + 1)](image)

        _, prediction = torch.max(outputs, 1)  # 按行取最大值
        pred.append(prediction)
    print(pred)
    result = Counter(pred).most_common(1)[0][0]
    # outputs = model1(image)
    # _, prediction = torch.max(outputs, 1)  # 按行取最大值
    # print(cifar10_classes[prediction.item()])
    # <class 'numpy.ndarray'>
    cv2.putText(img, cifar10_classes[result.item()], (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
    # print(type(img))
    img = Image.fromarray(img.astype('uint8')).convert('RGB')
    # print(type(img))
    img.save(r'static/images/labeled/labeled.png')
    # print(type(img))
    im = Image.open('static/images/labeled/labeled.png')
    # 创建一个 BytesIO 对象
    img_byte = io.BytesIO()
    im.save(img_byte, format='PNG')  # format: PNG or JPEG
    binary_content = img_byte.getvalue()  # im对象转为二进制流
    # img.show('img', img)
    # cv2.waitKey(0)
    img_stream = base64.b64encode(binary_content).decode()

    return img_stream
