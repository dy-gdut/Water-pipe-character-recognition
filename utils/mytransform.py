import numbers
import random
from torchvision.transforms import functional as F
from torchvision import transforms
from PIL import Image


class GroupResize(transforms.Resize):
    def __call__(self, img):
        def fun(i):
            return F.resize(i, self.size, self.interpolation)
        if isinstance(img, list):
            img = [fun(i) for i in img]
            return img
        else:
            return fun(img)


# 在图像内随机crop固定尺寸的image
class GroupRandomCrop(object):
    def __init__(self, crop_size):
        if isinstance(crop_size, numbers.Number):
            self.size = (int(crop_size), int(crop_size))
        else:
            self.size = crop_size

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.
        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        if isinstance(img,list):
            w, h = img[0].size
        else:
            w, h = img.size
        tw, th = output_size
        if w == tw and h == th:
            return 0, 0, h, w
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)

        return i, j, th, tw

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """
        i, j, h, w = self.get_params(img, self.size)

        def fun(o):
            return F.crop(o, i, j, h, w)

        if isinstance(img,list):
            img = [fun(img_) for img_ in img]
            return img
        else:
            return fun(img)


# 随机旋转，输入一个num-->[-num.num]随机旋转，也可以指定[num1,num2]
class GroupRandomRotate(transforms.RandomRotation):
    def __call__(self, img):
        angle = self.get_params(self.degrees)

        def fun(o):
            return F.rotate(o, angle, self.resample, self.expand, self.center, self.fill)
        if isinstance(img, list):
            img = [fun(img_) for img_ in img]
            return img
        else:
            return fun(img)


class GroupToTensor(object):

    def __call__(self, img):
        """
        Args:
            img (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        def fun(o):
            return F.to_tensor(o)
        if isinstance(img,list):
            img = [fun(img_) for img_ in img]
            return img
        else:
            return fun(img)


class GroupCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        def fun(o):
            for t in self.transforms:
                o = t(o)
            return o

        if isinstance(img, list):
            img = fun(img)
            return img
        else:
            return fun(img)


# 设置image亮度在[1-range,1+range]的范围变化,参数change_index可以指定要变换的image
class GroupRandomBrightness(object):
    def __init__(self, brightness_range =0.2,change_index=-1):
        self.BrightnessRange=brightness_range
        self.change_index = change_index

    def __call__(self, img):
        brightness = self.get_params(self.BrightnessRange)

        def fun(o):
            return F.adjust_brightness(o, brightness)

        if isinstance(img, list):
            if self.change_index == -1:
                img = [fun(img_) for img_ in img]
                return img
            else:
                assert isinstance(self.change_index, int)
                if self.change_index == 0:
                    return [fun(img[0]), img[1]]
                elif self.change_index == 1:
                    return [img[0], fun(img[1])]
                else:
                    raise Exception('change_index must be 0 or 1 !')

        else:
            return fun(img)

    @staticmethod
    def get_params(brightness_range):
        if brightness_range < 0:
            brightness_range = abs(brightness_range)
        brightness_range = 1+(2*random.random()-1)*brightness_range
        return brightness_range


class GroupRandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """

        def fun(o, r_):
            if r_ < self.p:
                return F.hflip(o)
            else:
                return o

        r = random.random()
        if isinstance(img, list):
            img = [fun(img_, r) for img_ in img]
            return img
        else:
            return fun(img, r)

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class GroupRandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """

        def fun(o, r_):
            if r_ < self.p:
                return F.vflip(o)
            else:
                return o

        r = random.random()
        if isinstance(img, list):
            img = [fun(img_, r) for img_ in img]
            return img
        else:
            return fun(img, r)

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


def HorizontalMove(img, p):
    W, H,  = img.size
    mode = "RGB" if img.mode == "RGB" else "L"
    color = (0, 0, 0) if img.mode == "RGB" else 0
    img_loader = Image.new(mode, img.size, color)
    img1 = F.crop(img, i=0, j=0, h=H, w=int(W*p))
    img2 = F.crop(img, i=0, j=int(W*p), h=H, w=W-int(W*p))
    img_loader.paste(img2, (0, 0))
    img_loader.paste(img1, (W-int(W*p), 0))
    return img_loader


class GroupRandomHorizontalMove(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        def fun(o, r_):
            return HorizontalMove(o, r_)
        r = random.random()
        if isinstance(img, list):
            img = [fun(img_, r) for img_ in img]
            return img
        else:
            return fun(img, r)


if __name__ == "__main__":
    from PIL import Image
    import os
    import cv2
    import numpy as np
    path = os.path.dirname(os.getcwd())+"/up_facet/image/2679-3_2020-12-03-10-16_1_3.bmp"

    image = cv2.imread(path)
    image_array = np.array(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    img_pil = Image.fromarray(image_array)
    img_pil.show()
    img = HorizontalMove(img_pil, 0.1)
    img.show()




