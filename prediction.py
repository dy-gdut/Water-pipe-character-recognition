# -*- coding: utf-8 -*
from flyai.framework import FlyAI
from path import *
from model import *
import torch
import PIL.Image as Image
import torchvision.transforms as transforms
import numpy as np
import torchvision
from repvgg import get_RepVGG_func_by_name
from cnn_models.densenet import densenet121
from cnn_models.resnet import resnet18
from efficientnet_pytorch import EfficientNet
from config import image_std,image_mean
import cv2
import matplotlib.pyplot as plt
mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

test_train_trans = transforms.Compose([
                            #transforms.ToPILImage(),
                            # transforms.RandomCrop(224),
                            # transforms.RandomHorizontalFlip(),
                            # transforms.RandomRotation(15),
                            # transforms.Resize(224),
                            transforms.ToTensor(),
                            transforms.Normalize(image_mean, image_std)
                        ])
class Prediction(FlyAI):
    def __init__(self):
        super(Prediction, self).__init__()
        ##########
        self.net0 = torchvision.models.resnet18(pretrained=False).cuda()
        feature = self.net0.fc.in_features
        # self.net0.fc = nn.Linear(in_features=feature, out_features=20, bias=True).cuda()
        self.net0.layer4 = nn.Sequential(*(list(self.net0.layer4.children()) + [nn.Dropout2d(0.3)]))
        self.net0.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=feature, out_features=20, bias=True).cuda()
        )

        self.net1 = torchvision.models.resnet18(pretrained=False).cuda()
        feature = self.net1.fc.in_features
        # self.net1.fc = nn.Linear(in_features=feature, out_features=20, bias=True).cuda()
        self.net1.layer4 = nn.Sequential(*(list(self.net1.layer4.children()) + [nn.Dropout2d(0.5)]))
        self.net1.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=feature, out_features=20, bias=True).cuda()
        )

        self.net2 = torchvision.models.resnet18(pretrained=False).cuda()
        feature = self.net2.fc.in_features
        # self.net2.fc = nn.Linear(in_features=feature, out_features=20, bias=True).cuda()
        self.net2.layer4 = nn.Sequential(*(list(self.net2.layer4.children()) + [nn.Dropout2d(0.5)]))
        self.net2.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=feature, out_features=20, bias=True).cuda()
        )

        self.net3 = torchvision.models.resnet18(pretrained=False).cuda()
        feature = self.net3.fc.in_features
        # self.net3.fc = nn.Linear(in_features=feature, out_features=20, bias=True).cuda()
        self.net3.layer4 = nn.Sequential(*(list(self.net3.layer4.children()) + [nn.Dropout2d(0.5)]))
        self.net3.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=feature, out_features=20, bias=True).cuda()
        )

        self.net4 = torchvision.models.resnet18(pretrained=False).cuda()
        feature = self.net4.fc.in_features
        # self.net4.fc = nn.Linear(in_features=feature, out_features=20, bias=True).cuda()
        self.net4.layer4 = nn.Sequential(*(list(self.net4.layer4.children()) + [nn.Dropout2d(0.5)]))
        self.net4.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=feature, out_features=20, bias=True).cuda()
        )


    def load_model(self):
        '''
        模型初始化，必须在此方法中加载模型
        '''
        MODEL_PATH0=MODEL_PATH+'/'+'0/'
        # print(MODEL_PATH0)
        for _,i,model_path in sorted(os.walk(MODEL_PATH0)):
            model_score = []
            for index in model_path:
                model_score.append(float(index.split('.')[0]+'.'+index.split('.')[1]))
            score_max=np.max(model_score)
            self.model_path0=MODEL_PATH0+'/'+str(score_max)+'.pth'
        print('Loda weights from {}'.format(self.model_path0))
        self.net0.load_state_dict(torch.load(self.model_path0))
        self.net0.eval()

        MODEL_PATH1=MODEL_PATH+'/'+'1/'
        for _,i,model_path in sorted(os.walk(MODEL_PATH1)):
            model_score = []
            for index in model_path:
                model_score.append(float(index.split('.')[0]+'.'+index.split('.')[1]))
            score_max=np.max(model_score)
            self.model_path1=MODEL_PATH1+'/'+str(score_max)+'.pth'
        print('Loda weights from {}'.format(self.model_path1))
        self.net1.load_state_dict(torch.load(self.model_path1))
        self.net1.eval()

        MODEL_PATH2=MODEL_PATH+'/'+'2/'
        for _,i,model_path in sorted(os.walk(MODEL_PATH2)):
            model_score = []
            for index in model_path:
                model_score.append(float(index.split('.')[0]+'.'+index.split('.')[1]))
            score_max=np.max(model_score)
            self.model_path2=MODEL_PATH2+'/'+str(score_max)+'.pth'
        print('Loda weights from {}'.format(self.model_path2))
        self.net2.load_state_dict(torch.load(self.model_path2))
        self.net2.eval()

        MODEL_PATH3=MODEL_PATH+'/'+'3/'
        for _,i,model_path in sorted(os.walk(MODEL_PATH3)):
            model_score = []
            for index in model_path:
                model_score.append(float(index.split('.')[0]+'.'+index.split('.')[1]))
            score_max=np.max(model_score)
            self.model_path3=MODEL_PATH3+'/'+str(score_max)+'.pth'
        print('Loda weights from {}'.format(self.model_path3))
        self.net3.load_state_dict(torch.load(self.model_path3))
        self.net3.eval()

        MODEL_PATH4=MODEL_PATH+'/'+'4/'
        for _,i,model_path in sorted(os.walk(MODEL_PATH4)):
            model_score = []
            for index in model_path:
                model_score.append(float(index.split('.')[0]+'.'+index.split('.')[1]))
            score_max=np.max(model_score)
            self.model_path4=MODEL_PATH4+'/'+str(score_max)+'.pth'
        print('Loda weights from {}'.format(self.model_path4))
        self.net4.load_state_dict(torch.load(self.model_path4))
        self.net4.eval()

    def predict(self, data):
        '''
        模型预测返回结果
        :param input: 评估传入样例 data为要预测的图片路径. 示例 "./data/input/image/0.jpg"
        :return: 模型预测成功中, 直接返回预测的结果 ，返回示例 浮点类型 如3.11
        '''
        pred_label_final=""
        image = cv2.imread(data, -1)
        h, w = image.shape[:2]
        w = w * 1.0 / 5  # five numbers in a image
        for index in range(5):
            label_list = []
            image1 = image[:, int(w * index): int(w * (index + 1)), :]
            image1 = cv2.resize(image1, dsize=(200, 200))
            image1 = Image.fromarray(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
            image2=test_train_trans(image1)
            #####
            out0 = self.net0(image2.unsqueeze(dim=0).cuda())
            pred_label0 = np.argmax(out0.cpu().detach().numpy())
            label_list.append(pred_label0)
            out1 = self.net1(image2.unsqueeze(dim=0).cuda())
            pred_label1 = np.argmax(out1.cpu().detach().numpy())
            label_list.append(pred_label1)
            out2 = self.net2(image2.unsqueeze(dim=0).cuda())
            pred_label2 = np.argmax(out2.cpu().detach().numpy())
            label_list.append(pred_label2)
            out3 = self.net3(image2.unsqueeze(dim=0).cuda())
            pred_label3 = np.argmax(out3.cpu().detach().numpy())
            label_list.append(pred_label3)
            out4 = self.net4(image2.unsqueeze(dim=0).cuda())
            pred_label4 = np.argmax(out4.cpu().detach().numpy())
            label_list.append(pred_label4)
            pred_label=max(label_list,key=label_list.count)
            pred_label=str(pred_label)
            pred_label_final=pred_label_final+pred_label
            if index<4:
                pred_label_final=pred_label_final+','
        print(pred_label_final)
        return {"label":pred_label_final}






def main():
    p=Prediction()
    p.load_model()
    path=os.listdir('./data/input/WaterMeterNumberRecognition/image')
    for i in path:
        # print(i)
        p.predict('./data/input/WaterMeterNumberRecognition/image/{}'.format(i))

if __name__=='__main__':
    main()