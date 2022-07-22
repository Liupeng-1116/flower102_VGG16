import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from model import VGG


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # load image
    img_path = "../tulip.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)
    # 对指定维度插入1。也就是在这里将0维变成1.意思就是表示只有一张输入图片
    # 之后，图像tensor形状为1*3*224*224（或者说这里的作用就是添加表征数量的1）

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # create model
    model = vgg(model_name="vgg16", num_classes=5).to(device)
    # load model weights
    weights_path = "./vgg16Net.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))

    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        # 指定维度，判断是否为1，如果是1就删除。不是则保留。
        # 不指定维度，就是删除整个维度数据中的1维处。
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    plt.title(print_res)
    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  predict[i].numpy()))
    plt.show()


if __name__ == '__main__':
    main()