import torch
#import numpy as np
from torch.utils.data import Dataset
import cv2


#自定义投影空间的数据类，需要输入：
# 1、ttdm（size为样本数*ttdm长*ttdm宽）2、label(样本数*图像长*图像宽)2、TTDS位置矩阵x 3、TTDS位置矩阵y
#三个均为numpy矩阵，在初始化中会转化为tenosr张量
#输出：
# class TTDS(Dataset): #Travel time difference space
#     def __init__(self,ttdm,label,ttds_x,ttds_y):
#         self.ttds_x=ttds_x
#         self.ttds_y = ttds_y
#         #ttdm = torch.from_numpy(ttdm)
#         ttdm[:,0,0]=0
#         #label = torch.from_numpy(label)
#         self.ttdm=ttdm.cuda()#放入GPU中，才能加速从TTDM到TTDS的转换，不然的话数据读取的速度都跟不上
#         self.label=label.reshape(label.shape[0],1,label.shape[1],label.shape[2]).cuda()
#     def __len__(self):
#         return self.ttdm.shape[0]
#     def __getitem__(self, idx):
#         ttds=self.ttdm[idx][self.ttds_x,self.ttds_y]
#         label=self.label[idx]#label应该得旋转才对，等和matlab里面得数据对比下，然后再换
#         return ttds,label#ttds的维度好像不对，得像label一样换一下
#
# class TTDM(Dataset): #Travel time difference space
#     def __init__(self,ttdm,label):
#         #ttdm = torch.from_numpy(ttdm)
#         self.ttdm=ttdm
#         ttdm[:,0,0]=0
#         #label = torch.from_numpy(label)
#         self.label=label.reshape(label.shape[0],1,label.shape[1],label.shape[2])
#     def __len__(self):
#         return self.ttdm.shape[0]
#     def __getitem__(self, idx):
#         ttdm=self.ttdm[idx]
#         label=self.label[idx]
#         return ttdm,label#ttds的维度好像不对，得像label一样换一下

def mandu(array):
    index=len(array)
    for i in range(index):
        resolution=0.11/array.shape[1]
        water_speed=1540
        array[i]=resolution/(array[i])-resolution/water_speed
        #array[i] = array[i] * 1e7
        array[i]=array[i]*2e7#不归一化的话1e7还是太小,20000的数据集的分布和400的差异有些大
        # array[i]=array[i]-array[i].min()
        # array[i]=array[i]/array[i].max()
    return array

class TTDM(Dataset): #Travel time difference space
    def __init__(self,ttdm,label):
        #ttdm = torch.from_numpy(ttdm)
        self.ttdm=ttdm
        ttdm[:,0,0]=0
        #label = torch.from_numpy(label)
        self.label=label.reshape(label.shape[0],1,label.shape[1],label.shape[2])
    def __len__(self):
        return self.ttdm.shape[0]
    def __getitem__(self, idx):
        ttdm=cv2.resize(self.ttdm[idx], (256, 256), interpolation=cv2.INTER_LINEAR)
        label=self.label[idx]
        return ttdm,label

#定义一个图像后处理的输入类,输入为吉洪诺夫重构出的图片以及标签，大小均为：样本数*图像长*图像宽
class RECON_POST(Dataset): #Travel time difference space
    def __init__(self,recon_imgs,labels):
        self.recon_imgs=recon_imgs.reshape(labels.shape[0],1,labels.shape[1],labels.shape[2])
        self.labels=labels.reshape(labels.shape[0],1,labels.shape[1],labels.shape[2])
    def __len__(self):
        return self.recon_imgs.shape[0]
    def __getitem__(self, idx):
        recon_img = self.recon_imgs[idx]
        label=self.labels[idx]
        return recon_img,label

class TTDMs_datasate(RECON_POST):
    pass

class Pre_Direct_Datasate(Dataset):
    def __init__(self,recon_imgs,labels1,labels2):
        self.recon_imgs=recon_imgs.reshape(recon_imgs.shape[0],1,recon_imgs.shape[1],recon_imgs.shape[2])
        self.labels1=labels1.reshape(labels1.shape[0],1,labels1.shape[1],labels1.shape[2])
        self.labels2=labels2.reshape(labels2.shape[0],1,labels2.shape[1],labels2.shape[2])
    def __len__(self):
        return self.recon_imgs.shape[0]
    def __getitem__(self, idx):
        recon_img = self.recon_imgs[idx]
        label1=self.labels1[idx]
        recon_img = self.recon_imgs[idx]
        label2=self.labels2[idx]
        return recon_img, label1, label2