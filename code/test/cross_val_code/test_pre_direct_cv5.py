import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../..")#导入隔壁文件夹下的文件
from data_prepare import Pre_Direct_Datasate
import time
# from my_code.My_models.test_models import Postprocessnet,UNet_origin
import scipy.io as io

#test_pre_direct_cv2后面的2表示结果存在末尾为2的文件夹中，比如results_cv5

device = torch.device("cuda:2")  # gpu准备
# device = torch.device("cpu")  # gpu准备
assert (torch.cuda.is_available())  # 判断GPU是否可用

time_all = []
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
    Y1_test = np.load(load_data_path2+'/PTTDMs_mix_fold'+str(fold)+'_test.npy')
    Y2_test = np.load(load_data_path2+'/slowness_mix_label_fold'+str(fold)+'_test.npy')
    
    '''----------------------有噪声--------------------'''
    '''直接添加，对应7dB噪声(实际6.7)'''
    # X_test = np.load('../../data/simu_simp_bre_pre_direct/noise_7dB/20210301/TTDMs_simp_bre_noise_test_7dB.npy')
    # Y1_test = np.load('../../data/simu_simp_bre_pre_direct/free/PTTDMs_simp_bre_free_test.npy')
    # Y2_test = np.load('../../data/simu_simp_bre_pre_direct/free/slowness_labels_simp_bre_test.npy')
    
    '''量级变换'''
    X_test, Y1_test = X_test * 1e6, Y1_test * 1e6  # 量级太小
    Y2_test = Y2_test * 1e8


    # 转换成自己的数据类并加载
    test_data = Pre_Direct_Datasate(X_test, Y1_test, Y2_test)  # 将输入标签放入自定义的TTDM模型类
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)  # num_workers应该设为多少合适


    # 模型实例化并放入GPU中，设定损失函数和优化器!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    '''加载前处理模型'''
    save_path = r'/home/hdd/yanguo/results_cv5/fold'+str(fold)+'/model_results/'
    save_model_pre_name = 'Pre_of_model_when_direct_best'# 'Pre_of_model_best'/Pre_of_model_when_direct_best
    save_path_and_model_pre_name = save_path + save_model_pre_name
    pre_net = torch.load(save_path_and_model_pre_name)
    # pre_net = torch.load(save_path_and_model_pre_name,map_location={'cuda:1':'cuda:3'})
    # pre_net = torch.load(save_path_and_model_pre_name, map_location={'cuda:1': 'cpu'})


    '''加载直接重构模型'''
    save_path = r'/home/hdd/yanguo/results_cv5/fold'+str(fold)+'/model_results/'
    save_model_direct_name = 'Direct_of_model_best'#Direct_of_model_best
    save_path_and_model_direct_name = save_path + save_model_direct_name
    direct_net = torch.load(save_path_and_model_direct_name)
    # direct_net = torch.load(save_path_and_model_direct_name, map_location={'cuda:1': 'cuda:3'})
    # direct_net = torch.load(save_path_and_model_direct_name, map_location={'cuda:1': 'cpu'})


    pre_net.to(device)
    direct_net.to(device)

    # summary(recon_net, input_size=(128,128))
    
    '''验证部分'''
    PTTDMs_hat = np.zeros((X_test.shape[0], 1, X_test.shape[1], X_test.shape[2]))
    RECONs_hat = np.zeros((X_test.shape[0], 1, X_test.shape[1], X_test.shape[2]))

    '''计时开始'''
    start = time.time()
    for i, data in enumerate(test_loader):
        pre_net.eval()  # 验证模式
        direct_net.eval()  # 验证模式
        x_test, y1_test, y2_test = data
        x_test = x_test.type(torch.FloatTensor)
        y1_test = y1_test.type(torch.FloatTensor)
        y2_test = y2_test.type(torch.FloatTensor)
        x_test = x_test.to(device)
        y1_test = y1_test.to(device)
        y2_test = y2_test.to(device)
        y1_hat_test = pre_net(x_test)
        y2_hat_test = direct_net(y1_hat_test)
    
        x_test = x_test.cpu().detach().numpy()
        y1_hat_test = y1_hat_test.cpu().detach().numpy()
        y2_hat_test = y2_hat_test.cpu().detach().numpy()
        y1_test = y1_test.cpu().detach().numpy()
        y2_test = y2_test.cpu().detach().numpy()
    
        compare_result_test_pre = np.concatenate((x_test[0, 0, :, :], y1_hat_test[0, 0, :, :], y1_test[0, 0, :, :]), axis=1)
        compare_result_test_direct = np.concatenate((y2_hat_test[0, 0, :, :], y2_test[0, 0, :, :]), axis=1)
        plt.figure()
        plt.imshow(compare_result_test_pre, vmin=-2, vmax=2)
        plt.colorbar()
        savename2 = r'/home/hdd/yanguo/results_cv5/fold' + str(fold) + '/test_results/sim/test_pre_' + str(i)
        plt.savefig(savename2)
        plt.close()
        plt.figure()
        plt.imshow(compare_result_test_direct, vmin=-2, vmax=2)
        plt.colorbar()
        savename2 = r'/home/hdd/yanguo/results_cv5/fold' + str(fold) + '/test_results/sim/test_direct_' + str(i)
        plt.savefig(savename2)
        plt.close()
        '''保存预测结果'''
        PTTDMs_hat[i, 0, :, :] = y1_hat_test
        RECONs_hat[i, 0, :, :] = y2_hat_test

    '''计时结束'''
    end = time.time()
    time_all.append(end-start)
    print(end-start)
    '''结果存为mat文件'''
    PTTDMs_hat = np.reshape(PTTDMs_hat, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))
    PTTDMs_hat = np.transpose(PTTDMs_hat)
    RECONs_hat = np.reshape(RECONs_hat, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))
    RECONs_hat = np.transpose(RECONs_hat)
    X_test = np.transpose(X_test)
    Y1_test = np.transpose(Y1_test)
    Y2_test = np.transpose(Y2_test)
    io.savemat(r'/home/hdd/yanguo/results_cv5/fold' + str(fold) + '/PTTDMs_RECONs_hat_results/sim/TTDMs.mat', {'TTDMs': X_test})
    io.savemat(r'/home/hdd/yanguo/results_cv5/fold' + str(fold) + '/PTTDMs_RECONs_hat_results/sim/recon_slowness_hat.mat',
               {'recon_slowness_hat': RECONs_hat})
    io.savemat(r'/home/hdd/yanguo/results_cv5/fold' + str(fold) + '/PTTDMs_RECONs_hat_results/sim/PTTDMs_hat.mat',
               {'PTTDMs_hat': PTTDMs_hat})
    io.savemat(r'/home/hdd/yanguo/results_cv5/fold' + str(fold) + '/PTTDMs_RECONs_hat_results/sim/PTTDMs.mat', {'PTTDMs': Y1_test})
    io.savemat(r'/home/hdd/yanguo/results_cv5/fold' + str(fold) + '/PTTDMs_RECONs_hat_results/sim/recon_slowness_label.mat', {'recon_slowness_label': Y2_test})

    print('第{0}折交叉验证结束'.format(fold))
print('sum of time_all:',sum(time_all))
print('aver of time_all:',sum(time_all)/855)