# -*- coding: utf-8 -*-
import argparse
import os
from flyai.data_helper import DataHelper
from flyai.framework import FlyAI
from path import MODEL_PATH, DATA_PATH
import json
import pandas as pd
from PIL import Image
import numpy as np
import cv2
import torch
from data_loder1 import  Appearance_scores
from torchvision import transforms
from torch.utils.data import DataLoader
from model import vgg11_bn,vgg13_bn,vgg16_bn,vgg19_bn
from torch import nn
import torch.optim as optim
from metrics_new import Metrics
import numpy as np
from path import *
from cnn_models import densenet121
from cnn_models import resnet18,resnet34
from torch.optim.lr_scheduler import StepLR
import torchvision
from utils.seg_metrics import Seg_metrics
from torchvision.models import resnet34
from cnn_models.senet import seresnet18,seresnet34
from flyai.utils import remote_helper
from cnn_models.resnext import resnext50
from efficientnet_pytorch import EfficientNet
from cnn_models.googlenet import googlenet
from Resnet import ResNet_Tri
from loss import emd_loss
from focal_loss import focal_loss,FocalLoss,LabelSmoothing
from repvgg import get_RepVGG_func_by_name
'''
此项目为FlyAI2.0新版本框架，数据读取，评估方式与之前不同
2.0框架不再限制数据如何读取
样例代码仅供参考学习，可以自己修改实现逻辑。
模版项目下载支持 PyTorch、Tensorflow、Keras、MXNET、scikit-learn等机器学习框架
第一次使用请看项目中的：FlyAI2.0竞赛框架使用说明.html
使用FlyAI提供的预训练模型可查看：https://www.flyai.com/models
学习资料可查看文档中心：https://doc.flyai.com/
常见问题：https://doc.flyai.com/question.html
遇到问题不要着急，添加小姐姐微信，扫描项目里面的：FlyAI小助手二维码-小姐姐在线解答您的问题.png
'''
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

# 项目的超参，不使用可以删除
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=50, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=16, type=int, help="batch size")
parser.add_argument('-gpu', default=True, help='use gpu or not')
parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate')
parser.add_argument('-net', type=str, default='resnet18_pretrain', help='Net of project')
parser.add_argument('-loss', type=str, default='CE', help='Net of project')
args = parser.parse_args()
mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
image_mean=[0.485, 0.456, 0.406]
image_std=[0.229, 0.224, 0.225]
class Main(FlyAI):
    '''
    项目中必须继承FlyAI类，否则线上运行会报错。
    '''
    def __init__(self,data_path,label_path):
        self.data_path=data_path
        self.label_path=label_path
        self.metrics=Metrics()
        self.train_trans = transforms.Compose([
                            #transforms.ToPILImage(),
                            # transforms.RandomCrop(224),
                            # transforms.RandomHorizontalFlip(),
                            # transforms.RandomCrop(200),
                            # transforms.RandomRotation(15),
                            # transforms.Resize(224),
                            transforms.ToTensor(),
                            transforms.Normalize(mean, std)])

    def init_net(self, ):
        if args.net=='vgg19_bn':
            self.net = vgg19_bn(num_class=1).cuda()
        if args.net == 'densenet121':
            self.net=densenet121(num_class=1).cuda()
        if args.net == 'resnet18':
            self.net=resnet18(num_classes=1).cuda()
        if args.net == 'resnet34':
            self.net = resnet34(num_classes=1).cuda()
        #  pretrain_model
        if args.net == 'resnet50_pretrain':
            self.net=torchvision.models.resnet50(pretrained=False).cuda()
        if args.net == 'resnet34_pretrain':
            self.net = torchvision.models.resnet34(pretrained=False).cuda()
        if args.net == 'resnet18_pretrain':
            self.net = torchvision.models.resnet18(pretrained=True).cuda()
            feature=self.net.fc.in_features
            self.net.layer4 = nn.Sequential(*(list(self.net.layer4.children()) + [nn.Dropout2d(0.5)]))
            self.net.fc= nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(in_features=feature, out_features=20, bias=True).cuda()
            )
        if args.net == 'googglenet_pretrain':
            net = torchvision.models.googlenet(pretrained=True)
            net.fc = nn.Linear(1024, 20)
            self.net = net.cuda()
        if args.net == 'RepVGG':
            arch = 'RepVGG-A0'
            repvgg_build_func = get_RepVGG_func_by_name(arch)
            self.net = repvgg_build_func(deploy=False).cuda()
            train_path = './pre_weights/RepVGG-20210429T164212Z-001/RepVGG/RepVGG-A0-train.pth'
            self.net.load_state_dict(torch.load(train_path))
            self.net.stage4 = nn.Sequential(*(list(self.net.stage4.children()) + [nn.Dropout2d(0.3)]))
            self.net.linear = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(in_features=1280, out_features=20).cuda()
            )
        if args.net == 'EfficientNet':
            self.net = EfficientNet.from_name('efficientnet-b5').cuda()
            pretrain_path = './pre_weights/efficientnet-b5-b6417697.pth'
            self.net.load_state_dict(torch.load(pretrain_path))
            print('model loder sucess!')
            feature=self.net._fc.in_features
            self.net.fc=nn.Linear(in_features=feature,out_features=20,bias=True).cuda()
        # resnet18_pretrain用于线上训练
        # pretrain_path = remote_helper.get_remote_data('https://www.flyai.com/m/resnet18-5c106cde.pth')
        # self.net = torchvision.models.resnet18(pretrained=False).cuda()
        # self.net.load_state_dict(torch.load(pretrain_path))
        # feature = self.net.fc.in_features
        # self.net.layer4 = nn.Sequential(*(list(self.net.layer4.children()) + [nn.Dropout2d(0.5)]))
        # self.net.fc = nn.Sequential(
        #     nn.Dropout(0.5),
        #     nn.Linear(in_features=feature, out_features=20, bias=True).cuda()
        # )
        if args.loss=='MSE':
            self.loss_function=nn.MSELoss(reduction='mean')
        if args.loss == 'CE':
            self.loss_function = nn.CrossEntropyLoss()
        if args.loss == 'focal_loss':
            self.loss_function=focal_loss(num_classes=20)
        if args.loss == 'LabelSmoothing':
            self.loss_function=LabelSmoothing(smoothing=0.1)
    def download_data(self):
        # 根据数据ID下载训练数据
        data_helper = DataHelper()
        # data_helper.download_from_ids("FacialBeautyPrediction")
        data_helper.download_from_ids("WaterMeterNumberRecognition")


    def deal_with_data(self,k,p):
        '''
        处理数据，没有可不写。
        :return:
        '''

        data_loder=Appearance_scores(self.data_path,self.label_path, transform=self.train_trans,k=k,p=p)
        # for test
        train_data_lo, val_data_lo = data_loder.get_data()
        self.train_data_loder = DataLoader(dataset=train_data_lo,shuffle=False,num_workers=1,batch_size=args.BATCH,
                                           collate_fn=data_loder.collate_fn)
        self.val_data_loder = DataLoader(dataset=val_data_lo, num_workers=2,
                                         batch_size=args.BATCH, collate_fn=data_loder.collate_fn)


        #################数据分析用
        # class_dict = {}
        # sort_dict = {}
        # for i in data_loder.read_csv(train_vsc_path):
        #     labels = i[1].split(',')
        #     for label_single in labels:
        #         if label_single not in class_dict.keys():
        #             class_dict[label_single] = 1
        #         else:
        #             class_dict[label_single] += 1
        # for indax in range(20):
        #     sort_dict[str(indax)] = class_dict[str(indax)]
        # print('数据集长度{}'.format(len(data_loder.read_csv(train_vsc_path))))
        # print(class_dict)
        # print(sort_dict)
        # print(sum(class_dict.values()))
        ##########
        #   判断数据是否泄露
        print('数据泄露检测')
        data_out_list=[]
        for index,(x,y) in enumerate(data_loder.train_data_list):
            data_out_list.append(x)
        for data_test in data_loder.valid_data_list:
            if data_test[0] in data_out_list:
                raise ('数据泄露')
        print('数据未泄露')

    def eval_training(self,epoch,vis_save_path,epoch_vil,is_val=False):
        with torch.no_grad():
            self.net.eval()
            acc_metrics = Seg_metrics(num_classes=20)
            t=0
            f=0
            for batch_index, (images, labels) in enumerate(self.val_data_loder):
                if args.gpu:
                    labels = labels.cuda()
                    images = images.cuda()
                outputs = self.net(images)
                output_np = outputs.cpu().detach().numpy()
                labels_np = labels.cpu().detach().numpy()
                label_list = torch.chunk(labels.squeeze(dim=1), chunks=int(len(labels) / 5), dim=0)
                outputs_list = torch.chunk(torch.argmax(outputs, dim=1), chunks=(int(len(outputs) / 5)), dim=0)
                for show_index in range(len(label_list)):
                    if all(label_list[show_index].cpu().detach().numpy() == outputs_list[show_index].cpu().detach().numpy()):
                        t += 1
                    else:
                        image_test = images[show_index * 5:(show_index + 1) * 5, :, :, :].cpu().detach().numpy()
                        a = np.concatenate([np.array([image_test[i]][0]) for i in range(5)], axis=2).transpose(
                            [1, 2, 0])
                        a = a * std + mean
                        f += 1
                        # 测试过程中的错误分析,变量is_val是否进行错误分析,epoch_vil进行错误分析的epoch
                        if is_val and epoch%epoch_vil==0:
                            if not os.path.exists(os.path.join(vis_save_path,str(epoch))):
                                os.makedirs(os.path.join(vis_save_path,str(epoch)))
                            cv2.imwrite(vis_save_path+str(epoch)+'/'+
                                        ','.join([str(i) for i in outputs_list[show_index].cpu().detach().numpy()])
                                        +'_label_'+','.join([str(j) for j in label_list[show_index].cpu().detach().numpy()])+'.jpg',a*255)
                self.metrics.add_batch(y_pred=np.argmax(output_np,axis=1), y=labels_np)
                a = torch.argmax(outputs, dim=1).flatten().cpu().numpy()
                b = (labels.flatten().cpu().numpy())
                for x_index in range(int(outputs.shape[0] / 5)):
                    a1 = a[x_index * 5: (x_index + 1) * 5]
                    b1 = b[x_index * 5: (x_index + 1) * 5]
                    if np.array_equal(a1, b1):
                        acc_metrics.add_batch(np.array([0]), np.array([0]))
                    else:
                        acc_metrics.add_batch(np.array([1]), np.array([0]))
            val_acc = acc_metrics.pixelAccuracy()
            print("验证集整幅图片精度为：{}".format(round(val_acc , 4)))
            score_test = self.metrics.apply()
            print('验证集单个字符精度: {}'.format(score_test))
            print('验证集字符预测个数: Ture = {} / {} ,Flase = {} / {}'.format(t,  len(self.val_data_loder.dataset),f,
                                                                      len(self.val_data_loder.dataset)))
        return score_test,val_acc
    def train(self):
        '''train
        训练模型，必须实现此方法
        :return:
        '''
        # np.random.seed(0)
        # torch.manual_seed(0)
        k=5
        for i in range(k):
            self.init_net()
            print("Init Net With fold {}".format(i))
            self.deal_with_data(k=k,p=i)
            best_train_scores=0
            # optimizer = optim.SGD(self.net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
            optimizer=optim.Adam(self.net.parameters(),lr=args.lr, weight_decay=1e-4)
            # optimizer=optim.RMSprop(self.net.parameters(),lr=args.lr, weight_decay=1e-4) 精度85左右
            scheduler =StepLR(optimizer, step_size=10, gamma=0.5)
            iter_per_epoch = len(self.train_data_loder)
            print('迭代次数{}'.format(iter_per_epoch))
            for epoch in range(1, args.EPOCHS+1):
                scheduler.step()
                self.net.train()
                acc_metrics = Seg_metrics(num_classes=20)
                t,f=0,0
                for batch_index, (images, labels) in enumerate(self.train_data_loder):
                    # print(labels)
                    if args.gpu:
                        labels = labels.cuda()
                        images = images.cuda()
                    optimizer.zero_grad()
                    outputs = self.net(images)
                    loss = self.loss_function(outputs, labels.squeeze(dim=1).long())
                    loss.backward()
                    optimizer.step()
                    label_list=torch.chunk(labels.squeeze(dim=1),chunks=int(len(labels)/5),dim=0)
                    outputs_list=torch.chunk(torch.argmax(outputs,dim=1),chunks=(int(len(outputs)/5)),dim=0)
                    for show_index in range(len(label_list)):
                        if all(label_list[show_index].cpu().detach().numpy()==outputs_list[show_index].cpu().detach().numpy()):
                            t+=1
                        else:
                            image_test=images[show_index*5:(show_index+1)*5,:,:,:].cpu().detach().numpy()
                            a = np.concatenate([np.array([image_test[i]][0]) for i in range(5)], axis=2).transpose([1,2,0])
                            a=a*std+mean
                            f+=1
                            # 此处也可以做一下训练错误可视化,但是我没有做了
                    output_np=outputs.cpu().detach().numpy()
                    labels_np=labels.cpu().detach().numpy()
                    self.metrics.add_batch(y_pred=np.argmax(output_np,axis=1),y=labels_np)
                    print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
                    loss.item(),
                    optimizer.param_groups[0]['lr'],
                    epoch=epoch,
                    trained_samples=batch_index * args.BATCH + args.BATCH,
                    total_samples=len(self.train_data_loder.dataset)))
                    score_train=self.metrics.apply()
                    a = torch.argmax(outputs, dim=1).flatten().cpu().numpy()
                    b = (labels.flatten().cpu().numpy())
                    for x_index in range(int(outputs.shape[0] / 5)):
                        a1 = a[x_index * 5: (x_index + 1) * 5]
                        b1 = b[x_index * 5: (x_index + 1) * 5]
                        if np.array_equal(a1, b1):
                            acc_metrics.add_batch(np.array([0]), np.array([0]))
                        else:
                            acc_metrics.add_batch(np.array([1]), np.array([0]))
                train_acc = acc_metrics.pixelAccuracy()
                print("训练集整幅图片精度为：{}".format(round(train_acc, 4)))
                print('训练集单个字符精度: {}'.format(score_train))
                print('训练集字符预测个数: Ture = {} / {} ,Flase = {} / {}'.format(t,len(self.train_data_loder.dataset),f,len(self.train_data_loder.dataset)))


                # 测试过程+保存验证最好的模型
                self.score_test,val_acc=self.eval_training(epoch,vis_save_path='./Visual/False_vis/'+str(i)+'_fold/',epoch_vil=10, is_val=True)
                # if epoch %1==0 and float(val_acc)>float(best_train_scores):
                if epoch % 1 == 0 and val_acc> best_train_scores:
                    print('saving weights file to {} /{}/--{}.pth '.format(MODEL_PATH,i,val_acc))
                    if not os.path.exists(MODEL_PATH+'/'+str(i)):
                        os.makedirs(MODEL_PATH+'/'+str(i))
                    torch.save(self.net.state_dict(), MODEL_PATH+'/'+str(i)+'/'+str(val_acc )+'.pth')
                    best_train_scores = val_acc
                # continue
            print('第{}折叠完成'.format(i+1))

if __name__ == '__main__':
    path=DATA_PATH+'/WaterMeterNumberRecognition/image'
    train_vsc_path=DATA_PATH+'/WaterMeterNumberRecognition/train.csv'
    main = Main(path,train_vsc_path)
    main.download_data()
    # main.deal_with_data()
    main.train()