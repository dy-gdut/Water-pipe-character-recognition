import numpy as np

# 语义分割其实是像素的2分类问题
# TP：正样本预测为正样本
# TN：负样本预测为负样本
# FP：负样本预测为正样本
# FN：正样本预测为负样本

# confusionMatrix:   0  1
#                 0[[1. 1.]
#                 1 [1. 3.]]--->TP=1,FP=3,TN=3,FN=1
# acc=(1+3)/(1+1+1+3)=0.666
"""
confusionMetric

P/L      P    N

P      TP    FP

N      FN    TN

"""


class Seg_metrics(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.confusionMatrix = np.zeros((num_classes, num_classes))

    # 正样本被分为正样本和错误副负样本的比例
    # 错检率
    def classRecall(self):
        # assert np.sum(self.confusionMatrix, axis=0)[0] != 0
        # recall = self.confusionMatrix[0][0]/np.sum(self.confusionMatrix, axis=0)[0]
        recall = 1
        return 1 - recall

    # 误检率
    def classFalse(self):
        assert np.sum(self.confusionMatrix, axis=0)[1] != 0
        FalseAlarm = self.confusionMatrix[0][1]/np.sum(self.confusionMatrix, axis=0)[1]
        return 1 - FalseAlarm

    def pixelAccuracy(self):
        # return all class overall pixel accuracy
        # acc = (TP + TN) / (TP + TN + FP + FN)
        acc = np.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return acc

    def classPixelAccuracy(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = TP/ (TP + FP)
        # 预测为正样本中正确的个数
        classAcc = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        # 返回2类的类别精度
        return classAcc

    def meanPixelAccuracy(self):
        classAcc = self.classPixelAccuracy()
        # ignore nan
        meanAcc = np.nanmean(classAcc)
        return meanAcc

    def meanIntersectionOverUnion(self):
        # Intersection = TP   Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix)
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(
            self.confusionMatrix)
        IoU = intersection / union
        mIoU = np.nanmean(IoU)
        return mIoU

    def clsIntersectionOverUnion(self, cls):
        assert cls < self.num_classes
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix)
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(
            self.confusionMatrix)
        IoU = intersection / union
        return IoU[cls]

    def Frequency_Weighted_Intersection_over_Union(self):
        # FWIoU = [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        freq = np.sum(self.confusionMatrix, axis=1) / np.sum(self.confusionMatrix)
        iu = np.diag(self.confusionMatrix) / (
                np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) -
                np.diag(self.confusionMatrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def genConfusionMatrix(self, imgPredict, imgLabel):
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.num_classes)
        # 预测标签与真实标签相同，相加后数值在混淆矩阵的对角线上
        label = self.num_classes * imgLabel[mask] + imgPredict[mask]
        # 统计0 -（num_classes**2-1）索引出现的次数
        count = np.bincount(label, minlength=self.num_classes**2)
        confusionMatrix = count.reshape(self.num_classes, self.num_classes)
        return confusionMatrix

    def add_batch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape, print("image_predict.shape:{}!=image_label.shape{}"
                                                         .format(imgPredict.shape, imgLabel.shape))
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)
        return self.confusionMatrix

    def reset(self):
        self.confusionMatrix = np.zeros((self.num_classes, self.num_classes))


if __name__ == '__main__':
    import cv2
    seg_metrics = Seg_metrics(num_classes=2)
    img = cv2.imread('/media/root/文档/wqr/image_data/mask/crack/crack_17.png', 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(3, 3))
    img_map = cv2.erode(img, kernel=kernel)

    img = np.where(np.array(img) > 128, 1, 0)
    img_mask = np.where(np.array(img_map) > 128, 1, 0)
    # reshape为一维numpy数据
    b = seg_metrics.add_batch(img_mask.reshape(1, -1), img.reshape(1, -1))
    Iou = seg_metrics.meanIntersectionOverUnion()
    acc = seg_metrics.pixelAccuracy()
    FwIou = seg_metrics.Frequency_Weighted_Intersection_over_Union()
    print(b)
    print(Iou)
    print(acc)
    print(FwIou)



