import os
import cv2
import glob
import random
import numpy as np


def gen_mask(root, mask_type, img_size, mask_scale=1.0, print_inf=False):
    file_path = os.path.join(root+'/mask/', mask_type)
    mask_path = glob.glob(file_path+'/*.png')
    assert len(mask_path) > 0
    # random select mask
    random_index = int(random.random()*len(mask_path))
    mask = cv2.imread(mask_path[random_index], 0)

    # random rotate [-90,90]
    height = mask.shape[0]
    width = mask.shape[1]
    rotation_angle = int(180 * (-0.5 + random.random()))

    # mat rotate 1 center 2 angle 3 缩放系数
    rotate_mat = cv2.getRotationMatrix2D((height * 0.5, width * 0.5), rotation_angle, 1)
    rotate_mask = cv2.warpAffine(mask, rotate_mat, (width, height))

    # random resize---scale[0.5,1.5)*mask_scale
    width_scale = 0.5*(1+random.random()*2)*mask_scale
    height_scale = 0.5*(1 + random.random() * 2) * mask_scale

    resize_width = min(int(width*width_scale), img_size[1])
    resize_height = min(int(height*height_scale), img_size[0])
    resize_mask = cv2.resize(rotate_mask, dsize=(resize_width, resize_height))

    # threshold
    binary_mask = cv2.threshold(resize_mask, 128, 1, cv2.THRESH_BINARY)[1]

    # gen random position(x,y)  avoid the mask of out-of-bound
    x = int(img_size[0]*random.random()*(1-resize_height/img_size[0]))
    y = int(img_size[1]*random.random()*(1-resize_width/img_size[1]))

    # add raw image
    container = np.zeros(shape=img_size, dtype=np.uint8)
    container[x:x+resize_height, y:y+resize_width] = binary_mask*255

    if print_inf:
        print(" init_size:({},{})\n".format(height, width),
              "random_size:({},{})\n".format(resize_height, resize_width),
              "random_angle:{}\n".format(rotation_angle),
              "random_index:{}\n".format(random_index),
              "random_position:({},{})\n".format(x, y))
    return container


def gen_defection(input_img, mask, fill_mode="Adapt"):
    mask_inv = cv2.bitwise_not(mask)
    img1 = cv2.bitwise_and(input_img, input_img, mask=mask_inv)
    if fill_mode == "Adapt":
        fill_value = 0
    else:
        fill_value = 0
    mask_fill = np.uint8(np.multiply(np.array(mask)/255, fill_value))
    output = cv2.add(img1, mask_fill)
    return output


if __name__ == "__main__":
    root1 = '/media/root/文档/wqr/image_data/'
    img_path = '/media/root/文档/wqr/image_data/Class2/train/0583.PNG'
    img = cv2.imread(img_path, 0)
    mask1 = gen_mask(root=root1, mask_type="ellipse", img_size=(512,512))
    img_defection = gen_defection(img, mask1)
    cv2.imshow('img', img_defection)
    cv2.waitKey()

