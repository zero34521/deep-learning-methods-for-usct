import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../..")#导入隔壁文件夹下的文件
from data_prepare import TTDMs_datasate
from My_models.test_models import Postprocessnet,UNet_origin
import scipy.io as io
from sklearn.model_selection import train_test_split

device=torch.device("cuda:1")#gpu准备
assert(torch.cuda.is_available())#判断GPU是否可用

for fold in range(1,6):
    '''----------------------------------数据读取与处理----------------------------'''
    X_test_exp = np.load('/home/user/yanguo/pycharm_connection/reconstruction_USCT/data/exp_data/TTDM_exp.npy')  # 维度：（样本数*图像大小*图像大小）

    '''量级变换'''
    X_test = X_test_exp * 1e6  # 慢度量级太小
    
    #转换成自己的数据类并加载
    test_data = X_test.reshape(X_test.shape[0], 1, X_test.shape[1], X_test.shape[2])
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)  # num_workers应该设为多少合适

    '''--------------------------模型实例化并放入GPU中，设定损失函数和优化器--------------------------------'''
    folder_path = '/home/hdd/yanguo/results_cv3'
    # folder_path = '/home/hdd/yanguo/Results_saved_cv/direct/2021.06.01.1'
    save_path=folder_path+'/fold'+str(fold)+'/model_results/'
    save_model_name='DeepPET_best' #
    save_path_and_model_name=save_path+save_model_name
    recon_net=torch.load(save_path_and_model_name)
    recon_net.to(device)
    
    Y_test_hat=np.zeros((X_test.shape[0],1,X_test.shape[1],X_test.shape[2]))
    '''---------------------------------------开始迭代循环训练以及测试--------------------------------'''
    for i,data in enumerate(test_loader):
        recon_net.eval()  # 验证模式
        x_test = data
        x_test = x_test.type(torch.FloatTensor)
        x_test = x_test.to(device)
        y_test_hat_i = recon_net(x_test)

        x_test = x_test.cpu().detach().numpy()
        y_test_hat_i = y_test_hat_i.cpu().detach().numpy()

        compare_result_test = np.concatenate((x_test[0, 0, :, :], y_test_hat_i[0, 0, :, :]), axis=1)
        # compare_result_test = np.concatenate((y_test_hat_i[0, 0, :, :], y_test[0, 0, :, :]), axis=1)
        plt.figure()
        plt.imshow(compare_result_test, vmin=-1, vmax=1)
        # plt.imshow(compare_result_test, vmin=-0.2, vmax=0.2)
        plt.colorbar()
        savename2 = folder_path+'/fold'+str(fold)+'/test_results/exp/'+'test_' + str(i)
        plt.savefig(savename2)
        plt.close()
        Y_test_hat[i,0,:,:]=y_test_hat_i
    
    '''结果存为mat文件'''
    Y_test_hat=np.reshape(Y_test_hat,(X_test.shape[0],X_test.shape[1],X_test.shape[2]))
    Y_test_hat=np.transpose(Y_test_hat)
    X_test=np.transpose(X_test)
    io.savemat(folder_path+'/fold'+str(fold)+'/direct_recon_results/exp/recon_slowness_hat.mat',
               {'recon_slowness_hat':Y_test_hat})
    io.savemat(folder_path+'/fold'+str(fold)+'/direct_recon_results/exp/TTDMs.mat', {'TTDMs':X_test})
