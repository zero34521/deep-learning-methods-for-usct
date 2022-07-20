import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from data_prepare import TTDMs_datasate
from sklearn.model_selection import train_test_split
from My_models.test_models import Postprocessnet, UNet_origin, DirectNet
import scipy.io as io

# X = np.load(r'D:\students\yanguo\reconstruction_USCT\data\simu_simple_breast_pre_process\四种噪声/TTDMs_simple_breast_noise_free_train_val_bysalstm.npy')#维度：（样本数*图像大小*图像大小）
#Y = np.load(r'D:\students\yanguo\reconstruction_USCT\data\simu_simple_breast_post_process\四种噪声/slowness_labels_simp_bre_train_val.npy')
#下面的直接拿标签进行尝试
X_test = np.load(r'D:\students\yanguo\rescontruction_USCT\data\simu_simple_breast_pre_process\四种噪声/PTTDMs_simple_breast_free_free_test.npy')
Y_test = np.load(r'D:\students\yanguo\rescontruction_USCT\data\simu_simple_breast_post_process\四种噪声/slowness_labels_simp_bre_test.npy')

'''先用更小的图像进行尝试'''
# X_test=X_test[:,::2,::2]
# Y_test=Y_test[:,::2,::2]
'''加载吉洪诺夫矩阵作为初值'''
T_inv = np.load(r'D:\students\yanguo\rescontruction_USCT\data\FC_matrix\T_inv_gama5.npy')
'''量级变换'''
X_test=X_test*1e6#量级太小
Y_test=Y_test*1e8#如果最后一层是relu激活函数对应1e8会比较好，最后一层是tanh得对应1e7，因为tanh上下限就是+-1，不能超过这个范围，不过要是最后一层没有激活，那建议还是1e8比较好
'''模型声明'''
recon_net=DirectNet(128)
'''初始化全连接层权重'''
T_inv=torch.from_numpy(T_inv)
T_inv=T_inv.type(torch.FloatTensor)
recon_net.fc1.weight=torch.nn.Parameter(T_inv*100)#64的话这里乘以50，128这里乘以100
# recon_net.fc1.requires_grad_=False
'''测试数据'''
X_test_tensor=torch.from_numpy(X_test)
X_test_tensor=X_test_tensor.type(torch.FloatTensor)
X_test_tensor=X_test_tensor.reshape(X_test_tensor.shape[0],1,X_test_tensor.shape[1],X_test_tensor.shape[2])
'''放入模型'''
Y_test_hat=recon_net(X_test_tensor)
Y_test_hat=Y_test_hat.detach().numpy()
Y_test_hat=Y_test_hat.reshape(Y_test_hat.shape[0],Y_test_hat.shape[2],Y_test_hat.shape[3])
'''画图'''
i=100
plt.figure()
plt.imshow(Y_test_hat[i,:,:])
plt.show()
plt.figure()
plt.imshow(Y_test[i,:,:])
plt.show()

'''结果存为mat文件'''
PTTDMs_hat=np.reshape(Y_test_hat,(X_test.shape[0],X_test.shape[1],X_test.shape[2]))
PTTDMs_hat=np.transpose(PTTDMs_hat)
X_test=np.transpose(X_test)
Y_test=np.transpose(Y_test)
io.savemat(r'D:\students\yanguo/reconstruction_USCT/results/direct_recon_results/sim/recon_slowness_hat.mat',
           {'recon_slowness_hat':PTTDMs_hat})
io.savemat(r'D:\students\yanguo/reconstruction_USCT/results/direct_recon_results/sim/recon_slowness_label.mat',
           {'recon_slowness_label':Y_test})
io.savemat(r'D:\students\yanguo/reconstruction_USCT/results/direct_recon_results/sim/TTDMs.mat', {'TTDMs':X_test})