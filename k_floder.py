import numpy as np
import os

def get_k_fold_data(k,i,X):
    assert k>1
    fold_size=len(X)//k
    x_train=None
    for j in range(k):
        idx=slice(j*fold_size,(j+1)*fold_size)
        x_part=X[idx]
        if j==i:
            x_valid=x_part
        elif x_train is None:
            x_train=x_part
        else:
            x_train=np.concatenate((x_train,x_part),axis=0)

    return x_train,x_valid


if __name__=="__main__":
    k=5 #五折叠
    x=[['1.jpg',1],['2.jpg',2],['3.jpg',3],['4.jpg',4],['5.jpg',5],['6.jpg',6],['7.jpg',7],['8.jpg',8],['9.jpg',9],['10.jpg',10]]
    #训练五次
    for i in range(5):
        x_train,x_valid=get_k_fold_data(k=k,i=i,X=x)
        print(x_train)
        print(x_valid)