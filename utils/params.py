

class Param(object):
    def __init__(self, _dict=None):
        if _dict is not None:
            self.regist_from_dict(_dict)

    def regist_from_parser(self, parser):
        for key, val in parser.__dict__.items():
            self.__setitem__(key, val)

    def regist_from_dict(self,_dict):
        assert isinstance(_dict,dict)
        for key,val in _dict.items():
            self.__setitem__(key, val)

    def regist(self, key, val):
        self.__setitem__(key, val)

    # 功能 A["a"]
    def __setitem__(self, key, value):
        super(Param,self).__setattr__( key, value)
        #self.__dict__[key] = value

    def __getitem__(self, attr):
        return super(Param, self).__getattribute__(attr)

    def __delitem__(self, key):
        try:
            del self.__dict__[key]
        except KeyError as k:
            return None

    # 功能  A.a
    def __setattr__(self, key, value):
        super(Param,self).__setattr__( key, value)

    def __getattribute__(self, attr):
        return super(Param, self).__getattribute__(attr)

    def __getattr__(self, attr):
        """
        重载此函数防止属性不存在时__getattribute__报错，而是返回None
        那“_ getattribute_”与“_ getattr_”的最大差异在于：
        1. 无论调用对象的什么属性，包括不存在的属性，都会首先调用“_ getattribute_”方法；
        2. 只有找不到对象的属性时，才会调用“_ getattr_”方法；
        :param attr:
        :return:
        """
        return None

    def __delattr__(self, key):
        try:
            del self.__dict__[key]
        except KeyError as k:
            return None

    def __str__(self):
        for key,val in self.__dict__.items():
            print("{}:{}".format(key,val))

    def __len__(self):
        return len(self.__dict__)

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def items(self):
        return self.__dict__.items()

def main():
    DefaultParam = {
        "mode": "training",  # 模式  {"training","testing" }
        "train_mode": "decision",  # 训练模式，{"segment","decision","total"}
        "image_size": (1280, 512),
        "epochs_num": 20,
        "batch_size": 1,
        "threshold": 0.5,
        "optimizer": "SGD",}
    params=Param(DefaultParam)
    params["name"] = "aaa"
    params.age = 15
    print(params["optimizer"])
    print(params.age)
    size=[60,60]



if __name__ == '__main__':
    main()

