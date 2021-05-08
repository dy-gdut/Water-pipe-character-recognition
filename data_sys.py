from torchvision import transforms
from data_loder1 import Appearance_scores
import matplotlib.pyplot as plt
path = './data/input/WaterMeterNumberRecognition/image'
train_vsc_path = './data/input/WaterMeterNumberRecognition/train.csv'
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

data_loder = Appearance_scores(path, train_vsc_path, train_trans)
class_dict={}
sort_dict={}

local_data_train={'0': 655, '1': 319,
                  '2': 110, '3': 119,
                  '4': 50, '5': 95, '6': 70,
                  '7': 122, '8': 143, '9': 84, '10': 10,
                  '11': 17, '12': 24, '13': 32, '14': 27,
                  '15': 13, '16': 25, '17': 27, '18': 30, '19': 28}

online_train_data={'0': 6616, '1': 3243,
                   '2': 1115, '3': 1189,
                   '4': 514, '5': 929,
                   '6': 765, '7': 1094, '8': 1521,
                   '9': 908, '10': 158, '11': 204,
                   '12': 216, '13': 322, '14': 183,
                   '15': 180, '16': 188, '17': 200,
                   '18': 245, '19': 210}
for i in data_loder.read_csv(train_vsc_path):
    labels=i[1].split(',')
    for label_single in labels:
        if label_single not  in class_dict.keys():
            class_dict[label_single]=1
        else:
            class_dict[label_single]+=1
for indax in range(20):
    sort_dict[str(indax)]=class_dict[str(indax)]
print(class_dict)
print(sort_dict)
# exit()
x1=online_train_data.keys()
y1=online_train_data.values()
print(sum(y1))
# print(x1)
# print(y1)
plt.bar(x1,y1)
for x_,y_ in enumerate(y1):
    plt.text(x_,y_+250,'%s'%y_,ha='center',va='top')
plt.savefig('./Data_online_distribute.jpg')
plt.show()
