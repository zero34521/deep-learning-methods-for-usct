import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import sys
# sys.path.append("../..")#导入隔壁文件夹下的文件
sys.path.append("/home/user/yanguo/pycharm_connection/reconstruction_USCT/my_code")#导入隔壁文件夹下的文件
from data_prepare import TTDMs_datasate
from My_models.test_models import Postprocessnet,UNet_origin
import scipy.io as io
from sklearn.model_selection import train_test_split
import time
#%%
device=torch.device("cuda", 1)#gpu准备
# device=torch.device("cpu")#gpu准备
assert(torch.cuda.is_available())#判断GPU是否可用
upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
# upsample = upsample.to(device)
time_all = []
for fold in range(1,6):
    '''----------------------------------数据读取与处理----------------------------'''
    # 无噪声
    load_data_path1 = '../../../data/cross_val_data/free/fold'+str(fold)
    X_test = np.load(load_data_path1+'/TTDMs_mix_fold'+str(fold)+'_test.npy')
    # 7dB噪声
    # load_data_path1 = '../../../data/cross_val_data/noise_7dB/fold'+str(fold)
    # X_test = np.load(load_data_path1+'/TTDMs_mix_7dB_fold'+str(fold)+'_test.npy')

    load_data_path2 = '../../../data/cross_val_data/free/fold' + str(fold)
    Y_test = np.load(load_data_path2 + '/slowness_mix_label_fold' + str(fold) + '_test.npy')

    '''先用更小的图像进行尝试,将128*128的输入改成32*32'''
    # X_test=X_test[:,::2,::2]
    # Y_test=Y_test[:,::2,::2]
    
    X_test=X_test*1e6#慢度量级太小
    Y_test=Y_test*1e8#relu激活函数对应1e8会比较好，tanh得对应1e7，因为tanh上下限就是+-1，不能超过这个范围
    
    samples_num=X_test.shape[0]#样本数
    
    
    #转换成自己的数据类并加载
    test_data=TTDMs_datasate(X_test,Y_test)
    test_loader=DataLoader(test_data,batch_size=1,shuffle=False,num_workers=0)#num_workers应该设为多少合适

    '''--------------------------模型实例化并放入GPU中，设定损失函数和优化器--------------------------------'''
    # folder_path = '/home/user/yanguo/pycharm_connection/reconstruction_USCT/Results_saved/direct/2021.01.20.1/model_results/'
    # save_path = folder_path#计算重构时间
    # model_name = 'AUTOMAP'  # option:direct\unet

    folder_path = '/home/hdd/yanguo/Results_saved_cv/direct/2021.06.01.1'
    # folder_path = '/home/hdd/yanguo/Results_saved_cv/direct/2021.03.17.1'
    save_path=folder_path+'/fold'+str(fold)+'/model_results/'
    model_name = 'DeepPET' #option:direct\unet\AUTOMAP\DeepPET

    save_model_name = model_name  #采用最后一个epoch模型
    save_model_name=model_name+'_best' #采用验证集最优模型
    save_path_and_model_name=save_path+save_model_name
    # save_path_and_model_name = '/home/user/yanguo/pycharm_connection/reconstruction_USCT/Results_saved/direct/2021.01.22.2/model_results/AUTOMAP_best'
    recon_net=torch.load(save_path_and_model_name, map_location='cuda:1')
    # recon_net = torch.load(save_path_and_model_name, map_location={'cuda:1': 'cuda:3'})
    # recon_net = torch.load(save_path_and_model_name, map_location={'cuda:1': 'cpu'})
    recon_net.to(device)
    
    Y_test_hat=np.zeros((X_test.shape[0],1,X_test.shape[1],X_test.shape[2]))
    '''计时开始'''
    start = time.time()
    '''---------------------------------------开始迭代循环训练以及测试--------------------------------'''
    for i,data in enumerate(test_loader):
        recon_net.eval()#验证模式
        x_test,y_test=data
        x_test=x_test.type(torch.FloatTensor)
        y_test=y_test.type(torch.FloatTensor)
        x_test=x_test.to(device)
        y_test=y_test.to(device)
        y_test_hat_i=recon_net(x_test)
        y_test_hat_i2=upsample(y_test_hat_i)
        x_test = x_test.cpu().detach().numpy()
        y_test_hat_i=y_test_hat_i.cpu().detach().numpy()
        y_test_hat_i2 = y_test_hat_i2.cpu().detach().numpy()
        y_test=y_test.cpu().detach().numpy()

        # compare_result_test=np.concatenate((x_test[0,0,:,:],y_test_hat_i[0, 0, :, :],y_test[0, 0, :, :]),axis=1)
        # compare_result_test = np.concatenate((y_test_hat_i[0, 0, :, :], y_test[0, 0, :, :]), axis=1)
        plt.figure()
        plt.imshow(y_test_hat_i[0,0,:,:],vmin=-1,vmax=1)
        # plt.imshow(compare_result_test, vmin=-0.2, vmax=0.2)
        plt.colorbar()
        # savename2 = r'/home/hdd/yanguo/results_cv3/fold'+str(fold)+'/test_results/sim/'+'test_' + str(i)
        savename2 = '/home/user/yanguo/pycharm_connection/reconstruction_USCT/results_pic_ren/results_deeppet_128/test'+str(i)
        plt.savefig(savename2)
        plt.close()

        plt.figure()
        plt.imshow(y_test_hat_i2[0, 0, :, :], vmin=-1, vmax=1)
        # plt.imshow(compare_result_test, vmin=-0.2, vmax=0.2)
        plt.colorbar()
        # savename2 = r'/home/hdd/yanguo/results_cv3/fold'+str(fold)+'/test_results/sim/'+'test_' + str(i)
        savename3 = '/home/user/yanguo/pycharm_connection/reconstruction_USCT/results_pic_ren/results_deeppet_256/test' + str(i)
        plt.savefig(savename3)
        plt.close()
        '''保存预测结果'''
        # Y_test_hat[i,0,:,:]=y_test_hat_i

    '''计时结束'''
    end = time.time()
    time_all.append(end-start)
    print(end-start)
    '''结果存为mat文件'''
    Y_test_hat=np.reshape(Y_test_hat,(X_test.shape[0],X_test.shape[1],X_test.shape[2]))
    Y_test_hat=np.transpose(Y_test_hat)
    X_test=np.transpose(X_test)
    Y_test=np.transpose(Y_test)
    # io.savemat(r'/home/hdd/yanguo/results_cv3/fold'+str(fold)+'/direct_recon_results/sim/recon_slowness_hat.mat',
    #            {'recon_slowness_hat':Y_test_hat})
    # io.savemat(r'/home/hdd/yanguo/results_cv3/fold'+str(fold)+'/direct_recon_results/sim/recon_slowness_label.mat',
    #            {'recon_slowness_label':Y_test})
    # io.savemat(r'/home/hdd/yanguo/results_cv3/fold'+str(fold)+'/direct_recon_results/sim/TTDMs.mat', {'TTDMs':X_test})
    print('第{0}折交叉验证结束'.format(fold))
print('sum of time_all:',sum(time_all))
print('aver of time_all:',sum(time_all)/855)