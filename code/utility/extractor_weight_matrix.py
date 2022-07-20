import torch
import numpy as np
import scipy.io as io
import sys
sys.path.append("..")#导入隔壁文件夹下的文件
from My_models.test_models import Postprocessnet, UNet_origin, DirectNet

'''加载模型'''
# save_path=r'D:/研究生/超声CT重构/reconstruction_USCT/results/model_results/'
load_path='../../results/Results_saved/pre_direct/2020.10.20.1/model_results/'
load_model_name='Direct_of_model_best' #
load_path_and_model_name=load_path+load_model_name
recon_net=torch.load(load_path_and_model_name)
'''提取矩阵'''
T_inv_128_train=recon_net.fc1.weight
T_inv_128_train=T_inv_128_train.cpu().detach().numpy()
'''分别以mat格式和npy格式保存起来'''
# load_path=r'E:\研究生\项目\项目_超声CT重构\results\full_connection'
path=load_path
io.savemat(load_path+'T_inv_128_train.mat',{'T_inv_128_train':T_inv_128_train})
np.save(load_path+'T_inv_128_train.npy', T_inv_128_train)