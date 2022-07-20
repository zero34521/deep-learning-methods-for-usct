import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../..")#导入隔壁文件夹下的文件
from data_prepare import Pre_Direct_Datasate

# from my_code.My_models.test_models import Postprocessnet,UNet_origin
import scipy.io as io

#test_pre_direct_cv2后面的2表示结果存在末尾为2的文件夹中，比如results_cv2

device = torch.device("cuda:1")  # gpu准备
assert (torch.cuda.is_available())  # 判断GPU是否可用

for fold in range(1,6):
    '''---------------------------数据读取与处理----------------------'''
    X_test_exp = np.load('/home/user/yanguo/pycharm_connection/reconstruction_USCT/data/exp_data/TTDM_exp.npy')  # 维度：（样本数*图像大小*图像大小）

    '''量级变换'''
    X_test = X_test_exp * 1e6  # 慢度量级太小

    # 转换成自己的数据类并加载
    test_data = X_test.reshape(X_test.shape[0], 1, X_test.shape[1], X_test.shape[2])
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)  # num_workers应该设为多少合适

    # 模型实例化并放入GPU中，设定损失函数和优化器!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    folder_path = '/home/hdd/yanguo/results_cv2'
    # folder_path = '/home/hdd/yanguo/Results_saved_cv/pre_direct/2021.03.06.1'
    '''加载前处理模型'''
    save_path = folder_path + '/fold' + str(fold) + '/model_results/'
    save_model_pre_name = 'Pre_of_model_when_direct_best'# 'Pre_of_model_best'/Pre_of_model_when_direct_best
    save_path_and_model_pre_name = save_path + save_model_pre_name
    pre_net = torch.load(save_path_and_model_pre_name)
    pre_net.to(device)
    '''加载直接重构模型'''
    # save_path = r'/home/hdd/yanguo/results_cv2/fold'+str(fold)+'/model_results/'
    save_path = folder_path + '/fold' + str(fold) + '/model_results/'
    save_model_direct_name = 'Direct_of_model_best'#Direct_of_model_best
    save_path_and_model_direct_name = save_path + save_model_direct_name
    direct_net = torch.load(save_path_and_model_direct_name)
    
    
    pre_net.to(device)
    direct_net.to(device)
    
    # summary(recon_net, input_size=(128,128))
    
    '''验证部分'''
    PTTDMs_hat = np.zeros((X_test.shape[0], 1, X_test.shape[1], X_test.shape[2]))
    RECONs_hat = np.zeros((X_test.shape[0], 1, X_test.shape[1], X_test.shape[2]))
    
    for i, data in enumerate(test_loader):
        pre_net.eval()  # 验证模式
        direct_net.eval()  # 验证模式
        x_test = data
        x_test = x_test.type(torch.FloatTensor)
        x_test = x_test.to(device)
        y1_hat_test = pre_net(x_test)
        y2_hat_test = direct_net(y1_hat_test)

        x_test = x_test.cpu().detach().numpy()
        y1_hat_test = y1_hat_test.cpu().detach().numpy()
        y2_hat_test = y2_hat_test.cpu().detach().numpy()

        compare_result_test_pre = np.concatenate((x_test[0, 0, :, :], y1_hat_test[0, 0, :, :]), axis=1)
        compare_result_test_direct = y2_hat_test[0, 0, :, :]
        plt.figure()
        plt.imshow(compare_result_test_pre, vmin=-2, vmax=2)
        plt.colorbar()
        savename2 = folder_path+'/fold' + str(fold) + '/test_results/exp/test_pre_' + str(i)
        plt.savefig(savename2)
        plt.close()
        plt.figure()
        plt.imshow(compare_result_test_direct, vmin=-2, vmax=2)
        plt.colorbar()
        savename2 = folder_path+'/fold' + str(fold) + '/test_results/exp/test_direct_' + str(i)
        plt.savefig(savename2)
        plt.close()
        '''把预测结果保存在模型'''
        PTTDMs_hat[i, 0, :, :] = y1_hat_test
        RECONs_hat[i, 0, :, :] = y2_hat_test
    
    '''结果存为mat文件'''
    PTTDMs_hat = np.reshape(PTTDMs_hat, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))
    PTTDMs_hat = np.transpose(PTTDMs_hat)
    RECONs_hat = np.reshape(RECONs_hat, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))
    RECONs_hat = np.transpose(RECONs_hat)
    X_test = np.transpose(X_test)
    io.savemat(folder_path+'/fold' + str(fold) + '/PTTDMs_RECONs_hat_results/exp/TTDMs.mat', {'TTDMs': X_test})
    io.savemat(folder_path+'/fold' + str(fold) + '/PTTDMs_RECONs_hat_results/exp/recon_slowness_hat.mat',
               {'recon_slowness_hat': RECONs_hat})
    io.savemat(folder_path+'/fold' + str(fold) + '/PTTDMs_RECONs_hat_results/exp/PTTDMs_hat.mat',
               {'PTTDMs_hat': PTTDMs_hat})
    print('第{0}折交叉验证结束'.format(fold))