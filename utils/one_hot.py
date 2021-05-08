# 定义one-hot
import torch


def one_hot(x, class_num):
    return torch.eye(class_num)[x, :]


def get_one_hot(label, class_nums):
    size = label.size()
    oneHot_size = (size[0], class_nums, size[2], size[3])
    input_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
    input_label = input_label.scatter_(1, label.data.long(), 1.0)
    return input_label

