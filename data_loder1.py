import os
import numpy as np
import torch
import random
import torchvision
from torchvision.transforms import transforms 
from torch.utils.data import Dataset
import pandas
import csv
from PIL import Image
import matplotlib.pyplot as plt
from random import shuffle
from copy import deepcopy
import cv2
from k_floder import get_k_fold_data
# from config import image_mean,image_std

image_mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
image_std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)


class Appearance_scores(Dataset):
    def __init__(self, File_path,train_csv_path, transform=None, rand_mask=True, phase=None,k=5,p=0):
        self.rand_mask = rand_mask
        self.root='./data/input/WaterMeterNumberRecognition'
        self.file_path=File_path
        self.transform=transform
        self.train_csv_path=train_csv_path
        self.k=k
        self.p=p
        # self.paths=None
        # self.labels=None
        self.trans = transforms.ToTensor()
        # self.list_folder(self.file_path)
        self.data_li=self.read_csv(self.train_csv_path)
        self.train_data_list,self.valid_data_list=get_k_fold_data(self.k,self.p,self.data_li)
        # self.train_data_list = self.data_li[:int(0.8*len(self.data_li))]
        # self.valid_data_list = self.data_li[int(0.8*len(self.data_li)):]
        # print(self.data_list)
        # exit()
        # self.data_list_new = self.func(self.data_list)
        # print('数据集长度{}'.format(len(self.data_list_new)))
        # print(self.data_list)
        self.phase=phase
        self.paths, self.labels = self.get_list1()

        self.online_dataset_dict = self.Gen_online_dataset(self.paths, self.labels)
        # print(self.online_dataset_dict)

    def __len__(self):
        if self.phase=='train':
            self.data_list=self.train_data_list
        else:
            self.data_list =self.valid_data_list
        return (len(self.data_list))

    def __getitem__(self, item):
        # image_path, image_index, label =self.data_list[item]
        # print(item)
        if self.phase=='train':
            self.data_list=self.train_data_list
        else:
            self.data_list =self.valid_data_list
        image_path, label = self.data_list[item]
        label = label.replace(",", " ")
        img_arr, label = self.Cut2cat(image_path, label, data_aug=False)
        img_pil = Image.fromarray(img_arr).convert("RGB")
        img_tensor = self.transform(img_pil)
        # print(image.shape)c
        # print(type(label))
        # exit()
        return img_tensor, label
    def get_list1(self):
        path=[]
        label=[]
        for i in self.train_data_list:
            path.append('./image/'+i[0].split('/')[-1])
            label.append(i[1].replace(',',' '))
        return path,label

    def get_list(self):
        assert os.path.exists(self.root)
        paths = []
        label = []
        with open(os.path.join(self.root, "train.csv")) as f:
            lines = f.readlines()
        lines = lines[1:]
        for line in lines:
            paths.append(line.split(",")[0])
            label_num_list = line.split('"')[1].split(',')
            num_cat = ""
            for num in label_num_list:
                pass
                # if int(num) >= 10:
                #     num = '0'
                num_cat = num_cat + ' ' + num
            num_cat = num_cat[1:]
            label.append(num_cat)
        # print(paths)
        # print(len(paths))
        # print(label)
        # print(len(label))
        # exit()
        return paths, label

    def Gen_online_dataset(self, img_paths=None, labels=None):
        dataset_dist = {}
        for i in range(10):
            dataset_dist[str(i)] = []
        for path, label in zip(img_paths, labels):
            label_list = label.split(' ')
            for cnt, label in enumerate(label_list):
                if int(label) < 10:
                    dataset_dist[label].append(path + '_' + str(cnt))
        return dataset_dist

    def gen_img_from_online_dataset(self, online_dataset_dict, size=(64, 200)):
        h = size[0]
        w = size[1] / 5
        rand_label = np.arange(0, 10, 1)
        # 选取数字的概率设定
        # index_p = [0.9, 0.9, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]
        index_p = [0.9, 0.9, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]
        # 生成预选的标签
        rand_labels = np.random.choice(rand_label, 5, index_p)
        img_slice_list = []
        gen_label_list = []
        for cnt, rand_label in enumerate(rand_labels):
            # 0.2的概率生成拼接图
            if random.random() > 0.2:
            # if random.random() > 0.5:
                index1 = str(rand_label)
                index2 = -1 if rand_label == 9 else rand_label
                index2 = str(index2+1)
                img_slices1 = online_dataset_dict[index1]
                img_slices2 = online_dataset_dict[index2]
                img_slice1 = img_slices1[int(random.random() * len(img_slices1))]
                img_slice2 = img_slices2[int(random.random() * len(img_slices2))]
                img1 = cv2.imread(self.root+img_slice1[1:-2], 0)
                img1 = cv2.resize(img1, dsize=(size[1], size[0]))
                index1 = int(img_slice1[-1])
                img_slice1 = img1[0:h, int(index1*w):int((index1+1)*w)]
                img2 = cv2.imread(self.root + img_slice2[1:-2], 0)
                img2 = cv2.resize(img2, dsize=(size[1], size[0]))
                index2 = int(img_slice2[-1])
                img_slice2 = img2[0:h, int(index2 * w):int((index2 + 1) * w)]

                img_v_stack = np.vstack([img_slice1, img_slice2])
                rand_pos = int((random.random()*32)+16)
                img_slice = img_v_stack[rand_pos:int(rand_pos+h), 0:int(w)]
                img_slice_list.append(img_slice)
                # 重新分配标签
                gen_label_list.append(str(rand_label+10))
            else:
                img_slices = online_dataset_dict[str(rand_label)]
                img_slice = img_slices[int(random.random()*len(img_slices))]
                img = cv2.imread(self.root+img_slice[1:-2], 0)
                img = cv2.resize(img, dsize=(size[1], size[0]))
                index = int(img_slice[-1])
                img_slice = img[0:h, int(index*w):int((index+1)*w)]
                img_slice_list.append(img_slice)
                gen_label_list.append(str(rand_label))

        gen_img = np.hstack(img_slice_list)
        gen_label = ""
        for i in gen_label_list:
            gen_label = gen_label + " " + i
        gen_label = gen_label[1:]
        #
        # cv2.imshow("a", gen_img)
        # print(gen_label)
        # cv2.waitKey()
        # exit()

        return gen_img, gen_label


    def gen_img_from_online_dataset1(self, online_dataset_dict, size=(64, 200)):
        h = size[0]
        w = size[1] / 5
        rand_label = np.arange(0, 10, 1)
        # 选取数字的概率设定
        index_p = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]
        # 生成预选的标签
        rand_labels = np.random.choice(rand_label, 5, index_p)
        img_slice_list = []
        gen_label_list = []

        for cnt, rand_label in enumerate(rand_labels):
            p = random.random()
            if p > 0.1:
                # 0.9的概率进行字符向下随机移动且用均值填充区域 和 随机拼接随机裁剪成原尺寸
                # if p > 0.6:  只进行之前的增强效果还不错,p我是直接不小心设置的5
                if p > 5:
                    # 生成向下移动的随机变量(16~32)
                    down_pos = int(h/4 + random.random()*h/4)
                    # 未使用
                    # 生成向左移动的随机变量(0~10)
                    left_pos = int(random.random()*w/4)
                    # 生成向右移动的随机变量(0~10)
                    right_pos = int(random.random()*w/4)

                    # 从数据库中提取单字符图片img_slice
                    img_slices = online_dataset_dict[str(rand_label)]
                    img_slice = img_slices[int(random.random()*len(img_slices))]
                    img = cv2.imread(self.root+img_slice[1:-2], 0)
                    img = cv2.resize(img, dsize=(size[1], size[0]))
                    # print(img.shape)
                    index = int(img_slice[-1])
                    img_slice = img[0:h, int(index*w):int((index+1)*w)]

                    # 向下移动数字
                    pad_mean = np.mean(img_slice[(h-down_pos):h, :])
                    pad_slice = np.ones_like(img_slice[(h-down_pos):h, :])*pad_mean
                    cut_slice = img_slice[0:(h-down_pos), :]
                    # img_slice = cv2.resize(cut_slice, dsize=(img_slice.shape[1],img_slice.shape[0]))
                    # print(img_slice.shape)
                    # print(img_slice.shape)
                    # exit()
                    # cv2.imshow('image',img_slice)
                    # cv2.waitKey()
                    img_slice[down_pos:h, :] = cut_slice
                    img_slice[0:down_pos, :] = pad_slice
                    # print(img_slice.shape)
                    # exit()

                    # 计算新的标签
                    new_label = 10 if rand_label == 0 else rand_label
                    new_label = new_label+9
                    new_label = str(new_label)
                    img_slice_list.append(img_slice)
                    gen_label_list.append(new_label)
                else:
                    index1 = str(rand_label)
                    # 获取下一个的索引标签，9的下一个数字是0,所以得单独判断
                    index2 = -1 if rand_label == 9 else rand_label
                    index2 = str(index2 + 1)
                    img_slices1 = online_dataset_dict[index1]
                    img_slices2 = online_dataset_dict[index2]
                    img_slice1 = img_slices1[int(random.random() * len(img_slices1))]
                    img_slice2 = img_slices2[int(random.random() * len(img_slices2))]
                    img1 = cv2.imread(self.root + img_slice1[1:-2], 0)
                    img1 = cv2.resize(img1, dsize=(size[1], size[0]))
                    index1 = int(img_slice1[-1])
                    img_slice1 = img1[0:h, int(index1 * w):int((index1 + 1) * w)]
                    img2 = cv2.imread(self.root + img_slice2[1:-2], 0)
                    img2 = cv2.resize(img2, dsize=(size[1], size[0]))
                    index2 = int(img_slice2[-1])
                    img_slice2 = img2[0:h, int(index2 * w):int((index2 + 1) * w)]

                    img_v_stack = np.vstack([img_slice1, img_slice2])
                    # 生成一个随机位置进行裁剪
                    rand_pos = int((random.random() * 32) + 16)
                    img_slice = img_v_stack[rand_pos:int(rand_pos + h), 0:int(w)]
                    img_slice_list.append(img_slice)
                    # 重新分配标签
                    gen_label_list.append(str(rand_label + 10))
            else:
                # 从数据库获取图片不做处理
                img_slices = online_dataset_dict[str(rand_label)]
                img_slice = img_slices[int(random.random()*len(img_slices))]
                img = cv2.imread(self.root+img_slice[1:-2], 0)
                img = cv2.resize(img, dsize=(size[1], size[0]))
                index = int(img_slice[-1])
                img_slice = img[0:h, int(index*w):int((index+1)*w)]
                img_slice_list.append(img_slice)
                gen_label_list.append(str(rand_label))

        gen_img = np.hstack(img_slice_list)
        gen_label = ""
        for i in gen_label_list:
            gen_label = gen_label + " " + i
        gen_label = gen_label[1:]
        # print(gen_label)
        # cv2.imshow("a", gen_img)
        # cv2.waitKey()
        # exit()

        return gen_img, gen_label

    # def get_data(self, p):
    #     data_list = deepcopy(self.data_list)
    #     train_data = Appearance_scores(self.file_path, self.train_csv_path, self.transform)
    #     # del train_data.get_data
    #     self.val_transform=transforms.Compose([
    #                         #transforms.ToPILImage(),
    #                         # transforms.RandomCrop(224),
    #                         # transforms.RandomHorizontalFlip(),
    #                         # transforms.RandomRotation(15),
    #                         transforms.Resize(224),
    #                         transforms.ToTensor(),
    #                         transforms.Normalize(image_mean, image_std)
    #                     ])
    #     train_data.data_list = data_list[:int(p*len(data_list))]
    #     val_data = Appearance_scores(self.file_path, self.train_csv_path, self.val_transform)
    #     val_data.data_list = data_list[int(p*len(data_list)):]
    #     # del val_data.get_data
    #     return train_data, val_data
    def get_data(self):
        # data_list = deepcopy(self.data_list)
        train_data = Appearance_scores(self.file_path, self.train_csv_path, self.transform,phase='train',k=self.k,p=self.p)
        # del train_data.get_data
        self.val_transform=transforms.Compose([
                            #transforms.ToPILImage(),
                            # transforms.RandomCrop(224),
                            # transforms.RandomHorizontalFlip(),
                            # transforms.RandomRotation(15),
                            # transforms.Resize(224),
                            # transforms.RandomCrop(200),
                            transforms.ToTensor(),
                            transforms.Normalize(image_mean, image_std)
                        ])

        # train_data.data_list = data_list[:int(p*len(data_list))]
        # val_data = Appearance_scores(self.file_path, self.train_csv_path, self.val_transform)
        val_data = Appearance_scores(self.file_path, self.train_csv_path, self.val_transform,rand_mask=False, phase='valid',k=self.k,p=self.p)
        # train_data.data_list, val_data.data_list =get_k_fold_data(k=k,i=p,X=data_list)

        # train_data.data_list, val_data.data_list=self.train_data_list,self.valid_data_list

        # val_data.data_list = data_list[int(p*len(data_list)):]
        # del val_data.get_data
        return train_data, val_data


    # def transform(self, x):


    def make_data(self,data_list):
        pass


    def list_folder(self,root, use_absPath=False, func=None):
        """
        :param root:  文件夹根目录
        :param func:  定义一个函数，过滤文件
        :param use_absPath:  是否返回绝对路径， false ：返回相对于root的路径
        :return:
        """
        root = os.path.abspath(root)
        if os.path.exists(root):
            print("遍历文件夹【{}】......".format(root))
        else:
            raise Exception("{} is not existing!".format(root))
        files = []
        # 遍历根目录,
        for cul_dir, _, fnames in sorted(os.walk(root)):
            for fname in sorted(fnames):
                path = os.path.join(cul_dir, fname)  # .replace('\\', '/')
                if func is not None and not func(path):
                    continue
                if use_absPath:

                    files.append(path)
                else:
                    path=os.path.relpath(path, root)
                    path = './image/' + path
                    # files.append(os.path.relpath(path, root))
                    files.append(path)
        print("    find {} file under {}".format(len(files), root))
        print(files)
        return files

    def read_csv(self,csv_path):
        data_list=[]
        label_data=pandas.read_csv(csv_path)
        for i in range(len(label_data)):
            data=[]
            data.append('./data/input/WaterMeterNumberRecognition'+label_data['image_path'][i].replace('.','').split('j')[0]+'.jpg')
            data.append(label_data['label'][i])
            data_list.append(data)
        # print(data_list)
        # print('数据集长度为{}'.format(len(data_list)))


        return data_list

    def func(self, data_list):

        # data_list = data_list[:3]
        data_list_new = []
        # easy_data_dict = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[]}
        # hard_data_dict = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[]}
        for data in data_list:
            image_name, label_str = data
            labels = label_str.split(",")
            for i, label in enumerate(labels):
                data_list_new.append((image_name, i, int(label)))
        return data_list_new

    def Cut2cat(self, img_path, num_label, rand_cat=False, data_aug=False):
        # img = cv2.imread(img_path, 0)
        if (random.random() > 0.5) | (self.phase != "train"):
            img = cv2.imread(img_path, 0)
        else:
            img, num_label = self.gen_img_from_online_dataset1(self.online_dataset_dict)
        img_list = []
        crop_w = 200
        crop_h = 200
        img = cv2.resize(img, dsize=(200, 64))
        img_arr = np.array(img)
        if not data_aug:
            data_aug = transforms.Compose([transforms.Resize(size=(224, 224)),
                                           transforms.RandomCrop(size=(200, 200))])
        else:
            data_aug = self.transform

        for i in range(5):
            img_pil = Image.fromarray(img_arr[0:64, (40 * i):(40 * (i + 1))])
            img_pil = data_aug(img_pil)
            img_s = np.array(img_pil)
            img_list.append(img_s)
        img_list = np.stack(img_list, axis=0)
        rand_index = [0, 1, 2, 3, 4]
        cut_img_list = np.zeros(shape=(crop_h, crop_w*5))
        if rand_cat:
            np.random.shuffle(rand_index)
        label_list = num_label.split(" ")
        label = ""
        for cnt, index in enumerate(rand_index):
            mask = self.Rand_mask(mask_size=(crop_h, crop_w), gen_mask=self.rand_mask)
            cut_img_list[0:crop_h, (crop_w*cnt):(crop_w*(cnt+1))] =\
                img_list[index]*mask+(1-mask)*255
            label = label + ' ' + label_list[index]
        label = label[1:]
        label = [int(l) for l in label.split(" ")]
        # img_pil = Image.fromarray(np.array(cut_img_list))
        # img_pil.show()
        # exit()
        return cut_img_list, label

    @staticmethod
    def Rand_mask(mask_size=(54, 30), n=50, gen_mask=True):
        n = 5 + int(random.random()*n)
        mask = np.ones(shape=mask_size)
        if (random.random() > 0.5) and gen_mask:
            x = int(random.random() * (mask_size[0]-n))
            y = int(random.random() * (mask_size[1]-n))
            mask[x:x+n, y:y+n] = 0
        return mask

    @ staticmethod
    def collate_fn(data):
        x_list = []
        y_list = []
        for (x, y) in data:
            x = torch.chunk(x.unsqueeze(dim=0), 5, dim=3)

            x = torch.cat(x, dim=0)
            x_list.append(x)
            y_list.append(torch.tensor(y))

        x_out = torch.cat(x_list, dim=0)
        y_out = torch.cat(y_list, dim=0).unsqueeze(dim=1)

        return x_out, y_out











class Appearance_scores_div_into(Appearance_scores):
    def __init__(self,model,tra_val_rate,File_path,train_csv_path,transform=None):
        super(Appearance_scores_div_into, self).__init__(File_path,train_csv_path, transform=None)
        self.model=model
        self.tra_val_rate=tra_val_rate
        # shuffle(self.data_list)
        self.data_input=self.dev_into()

    def __len__(self):
        return (len(self.data_list))


    def dev_into(self):
        if not isinstance(self.model,str):
            raise ('model trouble!')
        if not self.tra_val_rate>0 and self.tra_val_rate<1:
            raise ('rate must belong to [0-1]')
        if self.model=='train':
            self.input_list=self.data_list[:,(len(self.data_list))*self.tra_val_rate]
        if self.model=='val':
            self.input_list = self.data_list[ (len(self.data_list)) * self.tra_val_rate,:]

    def __getitem__(self, item):
        image_path=self.data_input[item][0]
        label=self.data_input[item][1]
        image=Image.open(image_path).convert('RGB')
        # plt.imshow(image)
        # plt.show()
        if self.transform is not None:
            image = self.transform(image)
        return image,label




def main():
    path='./data/input/WaterMeterNumberRecognition/image'
    train_vsc_path='./data/input/WaterMeterNumberRecognition/train.csv'
    mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
    train_trans = transforms.Compose([
        # transforms.ToPILImage(),
        # transforms.RandomCrop(224),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomCrop(200),
        # transforms.RandomRotation(15),
        # transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])

    data_loder=Appearance_scores(path,train_vsc_path, train_trans)
    train_data_loader, val_data_loader = data_loder.get_data()

    data_demo = train_data_loader[1]
    print(data_demo[1], data_demo[2], data_demo[3])
    cv2.imshow("1", data_demo[0])
    cv2.waitKey()
    # print(len(train_data_loader))
    # print(len(val_data_loader))

if __name__=="__main__":
    main()
