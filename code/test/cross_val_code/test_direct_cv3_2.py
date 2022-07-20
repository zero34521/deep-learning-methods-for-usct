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


'''代码名后面的_2代表模型拆分到了两个GPU'''

device=torch.device("cuda:1")#gpu准备
assert(torch.cuda.is_available())#判断GPU是否可用

for fold in range(1,6):
    '''----------------------------------数据读取与处理----------------------------'''
    # 无噪声
    load_data_path1 = '../../../data/cross_val_data/free/fold'+str(fold)
    # X_test = np.load(load_data_path1+'/TTDMs_mix_fold'+str(fold)+'_test.npy')
    X_test = np.load(load_data_path1 + '/PTTDMs_mix_fold' + str(fold) + '_test.npy')
    '''7dB'''
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
    '''方式一'''
    # recon_net=UNet_origin(1,1)
    # # recon_net=Postprocessnet()
    # # print(recon_net)
    # recon_net.to(device)
    # # summary(recon_net, input_size=(128,128))
    # save_path=r'/home/user/yanguo/pycharm_connection/reconstruction_USCT/results2/model_results/'
    # save_model_name='model_best' #option:u-net twolayersnet resnet
    # save_path_and_model_name=save_path+save_model_name
    # recon_net.load_state_dict(torch.load(save_path_and_model_name))
    # #check_model.state_dict()#查看模型参数
    '''方式二'''
    save_path=r'/home/hdd/yanguo/results_cv3/fold'+str(fold)+'/model_results/'
    # save_path=r'D:/students/yanguo/项目_超声CT重构/结果记录/direct_results/2020.6.15.2/model_results/'
    save_model_name='AUTOMAP_best' #
    save_path_and_model_name=save_path+save_model_name
    recon_net=torch.load(save_path_and_model_name)
    # recon_net.to(device)
    
    Y_test_hat=np.zeros((X_test.shape[0],1,X_test.shape[1],X_test.shape[2]))
    '''---------------------------------------开始迭代循环训练以及测试--------------------------------'''
    for i,data in enumerate(test_loader):
        recon_net.eval()#验证模式
        x_test,y_test=data
        x_test=x_test.type(torch.FloatTensor)
        y_test=y_test.type(torch.FloatTensor)
        x_test=x_test
        y_test=y_test.to('cuda:1')
        y_test_hat_i=recon_net(x_test)
    
        x_test = x_test.cpu().detach().numpy()
        y_test_hat_i=y_test_hat_i.cpu().detach().numpy()
        y_test=y_test.cpu().detach().numpy()
        compare_result_test=np.concatenate((x_test[0,0,:,:],y_test_hat_i[0, 0, :, :],y_test[0, 0, :, :]),axis=1)
        # compare_result_test = np.concatenate((y_test_hat_i[0, 0, :, :], y_test[0, 0, :, :]), axis=1)
        plt.figure()
        plt.imshow(compare_result_test,vmin=-1,vmax=1)
        # plt.imshow(compare_result_test, vmin=-0.2, vmax=0.2)
        plt.colorbar()
        savename2 = r'/home/hdd/yanguo/results_cv3/fold'+str(fold)+'/test_results/sim/'+'test_' + str(i)
        plt.savefig(savename2)
        plt.close()
        Y_test_hat[i,0,:,:]=y_test_hat_i
    
    '''结果存为mat文件'''
    Y_test_hat=np.reshape(Y_test_hat,(X_test.shape[0],X_test.shape[1],X_test.shape[2]))
    Y_test_hat=np.transpose(Y_test_hat)
    X_test=np.transpose(X_test)
    Y_test=np.transpose(Y_test)
    io.savemat(r'/home/hdd/yanguo/results_cv3/fold'+str(fold)+'/direct_recon_results/sim/recon_slowness_hat.mat',
               {'recon_slowness_hat':Y_test_hat})
    io.savemat(r'/home/hdd/yanguo/results_cv3/fold'+str(fold)+'/direct_recon_results/sim/recon_slowness_label.mat',
               {'recon_slowness_label':Y_test})
    io.savemat(r'/home/hdd/yanguo/results_cv3/fold'+str(fold)+'/direct_recon_results/sim/TTDMs.mat', {'TTDMs':X_test})
