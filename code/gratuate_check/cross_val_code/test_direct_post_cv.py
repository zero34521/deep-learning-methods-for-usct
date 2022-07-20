import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../..')
from data_prepare import Pre_Direct_Datasate
from My_models.test_models import Postprocessnet, UNet_origin
import scipy.io as io
import time

device = torch.device("cuda:1")  # gpu准备
# device = 'cpu'  # gpu准备
assert (torch.cuda.is_available())  # 判断GPU是否可用

time_all=[]
for fold in range(1,6):
    '''---------------------------数据读取与处理----------------------'''
    '''无噪声'''
    load_data_path1 = '../../../data/cross_val_data/free/fold'+str(fold)
    X_test = np.load(load_data_path1+'/TTDMs_mix_fold'+str(fold)+'_test.npy')
    '''1dB噪声'''
    # load_data_path1 = '../../../data/cross_val_data/noise_1dB/fold'+str(fold)
    # X_test = np.load(load_data_path1+'/TTDMs_mix_1dB_fold'+str(fold)+'_test.npy')
    '''4dB噪声'''
    # load_data_path1 = '../../../data/cross_val_data/noise_4dB/fold'+str(fold)
    # X_test = np.load(load_data_path1+'/TTDMs_mix_4dB_fold'+str(fold)+'_test.npy')
    '''7dB噪声'''
    # load_data_path1 = '../../../data/cross_val_data/noise_7dB/fold'+str(fold)
    # X_test = np.load(load_data_path1+'/TTDMs_mix_7dB_fold'+str(fold)+'_test.npy')
    '''10dB噪声'''
    # load_data_path1 = '../../../data/cross_val_data/noise_10dB/fold'+str(fold)
    # X_test = np.load(load_data_path1+'/TTDMs_mix_10dB_fold'+str(fold)+'_test.npy')

    load_data_path2 = '../../../data/cross_val_data/free/fold' + str(fold)
    Y1_test = np.load(load_data_path2+'/slowness_mix_label_fold'+str(fold)+'_test.npy')
    Y2_test = np.load(load_data_path2+'/slowness_mix_label_fold'+str(fold)+'_test.npy')

    '''----------------------有噪声--------------------'''
    '''直接添加，对应7dB噪声(实际6.7)'''

    '''量级变换'''
    X_test, Y1_test = X_test * 1e6, Y1_test * 1e8  # 量级太小
    Y2_test = Y2_test * 1e8
    
    # 转换成自己的数据类并加载
    
    test_data = Pre_Direct_Datasate(X_test, Y1_test, Y2_test)  # 将输入标签放入自定义的TTDM模型类
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)  # num_workers应该设为多少合适


    # 模型实例化并放入GPU中，设定损失函数和优化器!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    '''输入模型以及保存数据的文件夹'''
    folder_path = '/home/hdd/yanguo/results_cv1'
    # folder_path = '/home/hdd/yanguo/Results_saved_cv/direct_post/2021.03.06.1'



    '''加载直接重构模型'''
    save_path = folder_path+'/fold'+str(fold)+'/model_results/'
    save_model_direct_name = 'Direct_of_model_when_post_best'#Direct_of_model，Direct_of_model_when_post_best，Direct_of_model_epoch200
    save_path_and_model_direct_name = save_path + save_model_direct_name
    direct_net = torch.load(save_path_and_model_direct_name)
    # direct_net = torch.load(save_path_and_model_direct_name, map_location={'cuda:2': 'cuda:3'})
    # direct_net = torch.load(save_path_and_model_direct_name, map_location={'cuda:2': 'cpu'})

    '''加载后处理模型'''
    save_path = folder_path+'/fold'+str(fold)+'/model_results/'
    save_model_pre_name = 'Post_of_model_best'#Post_of_model，Post_of_model_best，post_of_model_epoch200
    save_path_and_model_pre_name = save_path + save_model_pre_name
    post_net = torch.load(save_path_and_model_pre_name)
    # post_net = torch.load(save_path_and_model_pre_name,map_location={'cuda:2':'cuda:3'})
    # post_net = torch.load(save_path_and_model_pre_name, map_location={'cuda:2': 'cpu'})
    post_net.to(device)
    
    direct_net.to(device)
    post_net.to(device)

    # summary(recon_net, input_size=(128,128))
    
    '''验证部分'''
    direct_recon_slowness_hats = np.zeros((X_test.shape[0], 1, X_test.shape[1], X_test.shape[2]))
    post_recon_slowness_hats = np.zeros((X_test.shape[0], 1, X_test.shape[1], X_test.shape[2]))

    '''计时开始'''
    start = time.time()
    for i, data in enumerate(test_loader):
        direct_net.eval()  # 验证模式
        post_net.eval()  # 验证模式
        x_test, y1_test, y2_test = data
        x_test = x_test.type(torch.FloatTensor)
        y1_test = y1_test.type(torch.FloatTensor)
        y2_test = y2_test.type(torch.FloatTensor)
        x_test = x_test.to(device)
        y1_test = y1_test.to(device)
        y2_test = y2_test.to(device)
        y1_hat_test = direct_net(x_test)
        y2_hat_test = post_net(y1_hat_test)
    
        x_test = x_test.cpu().detach().numpy()
        y1_hat_test = y1_hat_test.cpu().detach().numpy()
        y2_hat_test = y2_hat_test.cpu().detach().numpy()
        y1_test = y1_test.cpu().detach().numpy()
        y2_test = y2_test.cpu().detach().numpy()
    
        compare_result_test1 = np.concatenate((x_test[0, 0, :, :], y1_hat_test[0, 0, :, :], y1_test[0, 0, :, :]), axis=1)
        compare_result_test2 = np.concatenate((y1_hat_test[0, 0, :, :], y2_hat_test[0, 0, :, :], y2_test[0, 0, :, :]), axis=1)
        plt.figure()
        plt.imshow(compare_result_test1, vmin=-1, vmax=1)
        plt.colorbar()
        savename2 = folder_path+'/fold'+str(fold)+'/test_results/sim/' + 'test_direct_' + str(i)
        plt.savefig(savename2)
        plt.close()
        plt.figure()
        plt.imshow(compare_result_test2, vmin=-1, vmax=1)
        plt.colorbar()
        savename2 = folder_path+'/fold'+str(fold)+'/test_results/sim/' + 'test_post_' + str(i)
        plt.savefig(savename2)
        plt.close()
        '''把预测结果保存在模型'''
        direct_recon_slowness_hats[i, 0, :, :] = y1_hat_test
        post_recon_slowness_hats[i, 0, :, :] = y2_hat_test

    end = time.time()
    time_all.append(end-start)
    print(end-start)
    '''结果存为mat文件'''
    direct_recon_slowness_hats = np.reshape(direct_recon_slowness_hats, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))
    direct_recon_slowness_hats = np.transpose(direct_recon_slowness_hats)
    post_recon_slowness_hats = np.reshape(post_recon_slowness_hats, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))
    post_recon_slowness_hats = np.transpose(post_recon_slowness_hats)
    X_test = np.transpose(X_test)
    Y1_test = np.transpose(Y1_test)
    Y2_test = np.transpose(Y2_test)
    io.savemat(folder_path+'/fold'+str(fold)+'/direct_post_hats_results/sim/TTDMs.mat', {'TTDMs': X_test})
    io.savemat(folder_path+'/fold'+str(fold)+'/direct_post_hats_results/sim/post_recon_slowness_hats.mat',
               {'post_recon_slowness_hats': post_recon_slowness_hats})
    io.savemat(folder_path+'/fold'+str(fold)+'/direct_post_hats_results/sim/direct_recon_slowness_hats.mat',
               {'direct_recon_slowness_hats': direct_recon_slowness_hats})
    io.savemat(folder_path+'/fold'+str(fold)+'/direct_post_hats_results/sim/recon_slowness_label.mat', {'recon_slowness_label': Y1_test})
    io.savemat(folder_path+'/fold'+str(fold)+'/direct_post_hats_results/sim/recon_slowness_label.mat', {'recon_slowness_label': Y2_test})
    print('第{0}折交叉验证结束'.format(fold))
print('sum of time_all:',sum(time_all))
print('aver of time_all:',sum(time_all)/855)