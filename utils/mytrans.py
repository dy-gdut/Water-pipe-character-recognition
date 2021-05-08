import numbers
import random
from torchvision.transforms import functional as F
from torchvision import transforms


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
        def fun(o, ra):
            if ra < self.p:
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

        def fun(o, ra):
            if ra < self.p:
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


class GroupResize(transforms.Resize):
    def __call__(self, img):
        def fun(i):
            return F.resize(i, self.size, self.interpolation)
        if isinstance(img, list):
            img = [fun(i) for i in img]
            return img
        else:
            return fun(img)




