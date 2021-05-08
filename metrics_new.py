import numpy as np
from sklearn.metrics import confusion_matrix


class Metrics(object):

    def __init__(self,):
        # self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.y = []
        self.y_pred = []
        self.confusion_matrix = None

    def add_batch(self, y_pred, y):
        self.y.append(y.flatten())
        self.y_pred.append(y_pred.flatten())
        # print(self.y)
        # print(self.y_pred)

    def cul_acc(self):
        return np.sum(np.diag(self.confusion_matrix)) / np.sum(self.confusion_matrix)
    #
    # def cul_recall(self, index):
    #     """
    #     recall = tp / (tp + fn)
    #     :param index: index of cls
    #     :return:
    #     """
    #     return self.confusion_matrix[index][index] / np.sum(self.confusion_matrix[index, :])
    #
    # def cul_precision(self, index):
    #     """
    #     precision = tp / (tp + fp)
    #     :param index:
    #     :return:
    #     """
    #     return self.confusion_matrix[index][index] / np.sum(self.confusion_matrix[:, index])
    #
    # def cul_miou(self):
    #     """
    #     iou = tp / (tp + fn + fp)
    #     :return:
    #     """
    #     tp = np.diag(self.confusion_matrix)
    #     ious = tp / (np.sum(self.confusion_matrix, axis=0) + np.sum(self.confusion_matrix, axis=0) - tp)
    #     return np.nanmean(ious)

    def cul_mseacc(self):
        y=np.concatenate(self.y)
        y_pred=np.concatenate(self.y_pred)
        y_pred=np.clip(y_pred,0,5)
        # return  (5-(np.mean((y-y_pred)**2))**0.5)/5.0
        return (1 - (np.mean((y - y_pred) ** 2)) ** 0.5)

    def apply(self):
        # print( self.y_pred)
        # print(self.y)
        self.confusion_matrix = confusion_matrix(np.concatenate(self.y), np.concatenate(self.y_pred))
        # if self.confusion_matrix.shape[0] != self.num_classes:
        #     raise
        reslut = {
            "ACC": self.cul_acc(),
        }
        # loss=self.cul_mseacc()*100
        self.reset()
        return reslut


if __name__ == "__main__":
    a = np.array([0, 1, 4, 1, 1])
    b = np.array([0, 1, 1, 1, 1])
    c = np.array([0, 0, 0, 0, 0])
    d = np.where(a == 1, True, False)
    e = np.where(b == 1, True, False)
    # print(d)
    # print(e)
    #
    # tp = np.sum(np.bitwise_and(d, e))
    # fn = np.sum(np.bitwise_and(d, ~e))
    # print(tp, fn)
    me=Metrics()
    me.add_batch(a,b)
    print(me.apply())

