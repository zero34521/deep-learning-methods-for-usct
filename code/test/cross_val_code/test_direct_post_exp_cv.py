import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../..')
from data_prepare import Pre_Direct_Datasate
from My_models.test_models import Postprocessnet, UNet_origin
import scipy.signal as signal
import scipy.io as io

device = torch.device("cuda:2")  # gpu准备
# device = 'cpu'  # gpu准备
assert (torch.cuda.is_available())  # 判断GPU是否可用

for fold in range(1,6):
    '''---------------------------数据读取与处理----------------------'''
    X_test_exp = np.load('/home/user/yanguo/pycharm_connection/reconstruction_USCT/data/exp_data/TTDM_exp.npy')  # 维度：（样本数*图像大小*图像大小）
    for object_num in range(4):
        X_test_exp[object_num,:,:] = signal.medfilt(X_test_exp[object_num,:,:],(3,3))#中值滤波
    '''量级变换'''
    X_test = X_test_exp * 1e6  # 慢度量级太小

    # 转换成自己的数据类并加载
    test_data = X_test.reshape(X_test.shape[0], 1, X_test.shape[1], X_test.shape[2])
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)  # num_workers应该设为多少合适

    # 模型实例化并放入GPU中，设定损失函数和优化器!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    '''输入模型以及保存数据的文件夹'''
    folder_path = '/home/hdd/yanguo/results_cv1'
    # folder_path_model = folder_path
    folder_path_model = '/home/hdd/yanguo/Results_saved_cv/direct_post/2021.03.12.1'
    '''加载直接重构模型'''
    save_path = folder_path_model+'/fold'+str(fold)+'/model_results/'
    save_model_direct_name = 'Direct_of_model_when_post_best'#Direct_of_model，Direct_of_model_when_post_best，Direct_of_model_epoch200
    save_path_and_model_direct_name = save_path + save_model_direct_name
    direct_net = torch.load(save_path_and_model_direct_name)
    # direct_net = torch.load(save_path_and_model_direct_name, map_location={'cuda:2': 'cuda:3'})

    '''加载后处理模型'''
    save_path = folder_path_model+'/fold'+str(fold)+'/model_results/'
    save_model_pre_name = 'Post_of_model_best'#Post_of_model，Post_of_model_best，post_of_model_epoch200
    save_path_and_model_pre_name = save_path + save_model_pre_name
    post_net = torch.load(save_path_and_model_pre_name)
    # post_net = torch.load(save_path_and_model_pre_name,map_location={'cuda:2':'cuda:3'})
    post_net.to(device)
    
    direct_net.to(device)
    post_net.to(device)
    
    
    # summary(recon_net, input_size=(128,128))
    
    '''验证部分'''
    direct_recon_slowness_hats = np.zeros((X_test.shape[0], 1, X_test.shape[1], X_test.shape[2]))
    post_recon_slowness_hats = np.zeros((X_test.shape[0], 1, X_test.shape[1], X_test.shape[2]))
    
    for i, data in enumerate(test_loader):
        direct_net.eval()  # 验证模式
        post_net.eval()  # 验证模式
        x_test = data
        x_test = x_test.type(torch.FloatTensor)
        x_test = x_test.to(device)
        y1_hat_test = direct_net(x_test)
        y2_hat_test = post_net(y1_hat_test)

        x_test = x_test.cpu().detach().numpy()
        y1_hat_test = y1_hat_test.cpu().detach().numpy()
        y2_hat_test = y2_hat_test.cpu().detach().numpy()

        compare_result_test = np.concatenate((x_test[0, 0, :, :], y1_hat_test[0, 0, :, :], y2_hat_test[0, 0, :, :]),
                                             axis=1)
        plt.figure()
        plt.imshow(compare_result_test, vmin=-1, vmax=1)
        plt.colorbar()
        savename2 = folder_path+'/fold'+str(fold)+'/test_results/exp/' + 'test_direct_' + str(i)
        plt.savefig(savename2)
        plt.close()
        '''把预测结果保存在模型'''
        direct_recon_slowness_hats[i, 0, :, :] = y1_hat_test
        post_recon_slowness_hats[i, 0, :, :] = y2_hat_test
    
    '''结果存为mat文件'''
    direct_recon_slowness_hats = np.reshape(direct_recon_slowness_hats, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))
    direct_recon_slowness_hats = np.transpose(direct_recon_slowness_hats)
    post_recon_slowness_hats = np.reshape(post_recon_slowness_hats, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))
    post_recon_slowness_hats = np.transpose(post_recon_slowness_hats)
    X_test = np.transpose(X_test)
    io.savemat(folder_path+'/fold'+str(fold)+'/direct_post_hats_results/exp/TTDMs.mat', {'TTDMs': X_test})
    io.savemat(folder_path+'/fold'+str(fold)+'/direct_post_hats_results/exp/post_recon_slowness_hats.mat',
               {'post_recon_slowness_hats': post_recon_slowness_hats})
    io.savemat(folder_path+'/fold'+str(fold)+'/direct_post_hats_results/exp/direct_recon_slowness_hats.mat',
               {'direct_recon_slowness_hats': direct_recon_slowness_hats})
    print('第{0}折交叉验证结束'.format(fold))