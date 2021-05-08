import torch
import cv2
import os
from PIL import Image
from torchvision import transforms as T
import numpy as np
import random


class OCR_dataset(object):
    def __init__(self, data_root=None, phase="train", rand_mask=True):
        self.root = data_root
        self.paths, self.labels = self.get_list()

        self.phase = phase
        self.rand_mask = rand_mask
        if self.phase == "train":
            self.paths = self.paths[:int(0.8*len(self.paths))]
            self.labels = self.labels[:int(0.8*len(self.labels))]
        else:
            self.paths = self.paths[int(0.8*len(self.paths)):]
            self.labels = self.labels[int(0.8*len(self.labels)):]
        self.online_dataset_dict = self.Gen_online_dataset(self.paths, self.labels)
        self.trans = T.Compose([T.ToTensor(), T.Normalize(mean=[0.5], std=[0.5])])

    def __getitem__(self, item):
        img_arr, label = self.Cut2cat(os.path.join(self.root, self.paths[item]), self.labels[item])

        img_pil = Image.fromarray(np.uint8(img_arr))
        img_tensor = self.trans(img_pil)
        return img_tensor, label

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
        index_p = [0.9, 0.9, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]
        # 生成预选的标签
        rand_labels = np.random.choice(rand_label, 5, index_p)
        img_slice_list = []
        gen_label_list = []
        for cnt, rand_label in enumerate(rand_labels):
            # 0.2的概率生成拼接图
            if random.random() > 0.7:
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

        # cv2.imshow("a", gen_img)
        # print(gen_label)
        # cv2.waitKey()
        # exit()

        return gen_img, gen_label

    def __len__(self):
        return len(self.paths)

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
        return paths, label

    def Cut2cat1(self, img_path, num_label, rand_cat=False):
        img = cv2.imread(img_path, 0)
        img = cv2.resize(img, dsize=(200, 64))
        img_arr = np.array(img)
        img_list = []
        for i in range(5):
            img_list.append(img_arr[0:64, (40*i):(40*(i+1))])
        img_list = np.stack(img_list, axis=0)
        rand_index = [0, 1, 2, 3, 4]
        cut_img_list = np.zeros(shape=(64, 200))
        if rand_cat:
            np.random.shuffle(rand_index)
        label_list = num_label.split(" ")
        label = ""
        for cnt, index in enumerate(rand_index):
            cut_img_list[0:64, (40*cnt):(40*(cnt+1))] = img_list[index]*self.Rand_mask(gen_mask=self.rand_mask)
            label = label + ' ' + label_list[index]
        label = label[1:]
        label = [int(l) for l in label.split(" ")]

        return cut_img_list, label

    def Cut2cat(self, img_path, num_label, rand_cat=False):
        if (random.random() > 0.5) | (self.phase != "train"):
            img = cv2.imread(img_path, 0)

        else:
            img, num_label = self.gen_img_from_online_dataset(self.online_dataset_dict)

        # print(num_label)
        # cv2.imshow("a", img)
        # cv2.waitKey()
        # exit()
        img_list = []
        crop_w = 200
        crop_h = 200
        img = cv2.resize(img, dsize=(200, 64))
        img_arr = np.array(img)
        if self.phase == "train":
            data_aug = T.Compose([T.Resize(size=(224, 224)), T.RandomCrop(size=(crop_h, crop_w))])
        else:
            data_aug = T.Compose([T.Resize(size=(224, 224))])
            crop_w = 224
            crop_h = 224
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


# pre:10分类的预测概率[b,10]
def reshape_pre_label(pre):
    pre2, index2 = torch.topk(pre, k=2, dim=1)
    label = []
    for pre, index in zip(pre2, index2):
        if abs(index[0]-index[1]) != 1:
            label.append(int(index[0]))
        else:
            if abs(pre[0]-pre[1]) < 8:
                label.append(int(index[0])+10)
            else:
                label.append(int(index[0]))
    return label


def reshape_label(labels, num_classes=10):
    label_list = []
    for label in labels:
        label_hot = torch.zeros(size=(1, 1))
        if int(label) < 10:
            label_hot[:, 0] = int(label)
        else:
            label_hot[:, 0] = int(label) - 10
        label_list.append(label_hot)
    label_one_hot = torch.cat(label_list, dim=1).squeeze(dim=0)
    return label_one_hot


if __name__ == "__main__":
    root = "/media/root/软件/wqr/data/flyai/data/input/WaterMeterNumberRecognition"
    data1 = OCR_dataset(data_root=root)
    a, b = data1[0]




