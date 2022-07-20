import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
sys.path.append('../..')
from data_prepare import TTDMs_datasate
from My_models.test_models import Postprocessnet, UNet_origin, DirectNet, AUTOMAP, AUTOMAP_unet,DirectPostNet
from My_models.DeepPET import DeepPET
from torchsummary import summary

device=torch.device("cuda:3")#gpu准备
device_message = {device.type:device.index}
assert (torch.cuda.is_available())  # 判断GPU是否可用

for fold in range(1,6):
    '''---------------------------数据读取与处理----------------------'''
    print('第{0}折交叉验证开始'.format(fold))
    '''无噪声(无扩充)'''
    # load_data_path = '../../../data/cross_val_data/free/fold'+str(fold)
    # X_train = np.load(load_data_path+'/TTDMs_mix_fold'+str(fold)+'_train.npy')
    # X_val = np.load(load_data_path+'/TTDMs_mix_fold'+str(fold)+'_test.npy')
    #
    # Y_train = np.load(load_data_path+'/slowness_mix_label_fold'+str(fold)+'_train.npy')
    # Y_val = np.load(load_data_path+'/slowness_mix_label_fold'+str(fold)+'_test.npy')

    '''无噪声(16倍扩充)(训练集有扩增，测试集没有扩增)'''
    load_data_path1 = '/home/hdd/yanguo/cross_val_data_aug_16times/free/fold' + str(fold)
    load_data_path2 = '../../../data/cross_val_data/free/fold' + str(fold)
    X_train = np.load(load_data_path1 + '/TTDMs_mix_train_aug.npy')
    X_val = np.load(load_data_path2 + '/TTDMs_mix_fold' + str(fold) + '_test.npy')
    # X_train = np.load(load_data_path1 + '/PTTDMs_mix_train_aug.npy')
    # X_val = np.load(load_data_path2 + '/PTTDMs_mix_fold' + str(fold) + '_test.npy')
    Y_train = np.load(load_data_path1 + '/slowness_mix_label_train_aug.npy')
    Y_val = np.load(load_data_path2 + '/slowness_mix_label_fold' + str(fold) + '_test.npy')
    X_train = np.transpose(X_train)  # 用hdf5导入的文件，还得转置一下才行
    Y_train = np.transpose(Y_train)  # 用hdf5导入的文件，还得转置一下才行

    '''7dB噪声(16倍扩充)(训练集有扩增，测试集没有扩增)'''
    # load_data_path1 = '/home/hdd/yanguo/cross_val_data_aug_16times/noise_7dB/fold' + str(fold)
    # load_data_path2 = '../../../data/cross_val_data/noise_7dB/fold' + str(fold)
    # load_data_path3 = '/home/hdd/yanguo/cross_val_data_aug_16times/free/fold' + str(fold)
    # load_data_path4 = '../../../data/cross_val_data/free/fold' + str(fold)
    # X_train = np.load(load_data_path1 + '/TTDMs_mix_noise_train_aug.npy')
    # X_val = np.load(load_data_path2 + '/TTDMs_mix_7dB_fold' + str(fold) + '_test.npy')
    # Y_train = np.load(load_data_path3 + '/slowness_mix_label_train_aug.npy')
    # Y_val = np.load(load_data_path4 + '/slowness_mix_label_fold' + str(fold) + '_test.npy')
    # X_train = np.transpose(X_train)  # 用hdf5导入的文件，还得转置一下才行
    # Y_train = np.transpose(Y_train)  # 用hdf5导入的文件，还得转置一下才行

    '''量级变换'''
    X_train, X_val=X_train*1e6, X_val*1e6              #量级太小
    Y_train, Y_val=Y_train*1e8, Y_val *1e8
    # Y_train, Y_val=Y_train*1e7, Y_val *1e7
    '''
    1、如果最后一层是relu激活函数对应1e8会比较好，最后一层是tanh得对应1e7，
    因为tanh上下限就是+-1，不能超过这个范围，不过要是最后一层没有激活，
    那建议还是1e8比较好
    2、如果这里变了，那后面的权重矩阵的比值也得变
    '''
    
    '''先用更少的图像进行尝试'''
    # X_train = X_train[::100, :, :]
    # Y_train= Y_train[::100, :, :]
    
    # 转换成自己的数据类并加载
    batch_size = 10
    train_data = TTDMs_datasate(X_train, Y_train)  # 将输入标签放入自定义的TTDM模型类
    val_data = TTDMs_datasate(X_val, Y_val)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)  # num_workers应该设为多少合适
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=0)  # num_workers应该设为多少合适
    
    # 模型实例化并放入GPU中，设定损失函数和优化器!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    '''模型选择，吉洪诺夫为初值的全连接'''
    # recon_net = DirectNet(128)
    # model_name = 'direct'
    # '''---初始化直接重构全连接层权重---'''
    # T_inv = np.load(r'../../../data/FC_matrix/T_inv_gama5.npy')#加载吉洪诺夫矩阵
    # T_inv = torch.from_numpy(T_inv)
    # T_inv = T_inv.type(torch.FloatTensor)
    # recon_net.fc1.weight = torch.nn.Parameter(T_inv * 100)#64大小的图像对应倍数为50，128对应100

    '''模型选择：AUTOMAP'''
    # recon_net = AUTOMAP(128)#记得该下面对应的优化器
    # model_name = 'AUTOMAP'
    '''模型选择:unet'''
    recon_net = UNet_origin(1, 1)
    model_name = 'unet'
    '''模型选择:DeepPET'''
    # recon_net = DeepPET()
    # model_name = 'DeepPET'

    # print(recon_net)

    '''用GPU加载模型'''
    recon_net.to(device)
    # summary(recon_net.cuda(), input_size=(1,128, 128))
    
    criterion = nn.MSELoss()
    lr = 1e-3

    # optimizer = optim.RMSprop(recon_net.parameters(), lr=lr)#AUTOMAP专用
    optimizer = optim.Adam(recon_net.parameters(), lr=lr)  # DeepPET里采用了weight_decay=1e-5
    # optimizer = optim.Adam(recon_net.parameters(), lr=lr, weight_decay=1e-5)#DeepPET里采用了weight_decay=1e-5

    message_batch_size = '(batch_size:%d)' % (batch_size)
    message = '-------------------------------------------------------------------------------------------------------'
    with open('/home/hdd/yanguo/results_cv4/fold'+str(fold)+'/loss_results/log.txt', 'a') as log_f:
        log_f.write('%s\n' % message)
        log_f.write('%s\n' % device_message)
        log_f.write('%s\n' % message_batch_size)
    # 开始迭代循环训练以及测试
    train_loss = []  # 学习率曲线保存
    val_loss_save = []  # 学习率曲线保存
    val_loss_best = 10  # 随便取一个大的值
    epoch_best = 0
    for epoch in range(1, 100):
        # 训练
        train_loss_all = 0
        for i, data in enumerate(tqdm(train_loader)):
            recon_net.train()
            x_train, y_train = data
            x_train = x_train.type(torch.FloatTensor)
            y_train = y_train.type(torch.FloatTensor)
            x_train = x_train.to(device)
            y_train = y_train.to(device)
            optimizer.zero_grad()
            y_hat = recon_net(x_train)
            loss = criterion(y_hat, y_train)
            # print('当前训练的loss为:%f' % loss)
            loss.backward()
            optimizer.step()
            train_loss_all += loss.detach().cpu().numpy()
        train_loss_aver = train_loss_all / i
        train_loss.append(train_loss_aver)  # 学习率曲线
        x_train = x_train.cpu().detach().numpy()
        y_hat = y_hat.cpu().detach().numpy()
        y_train = y_train.cpu().detach().numpy()
        compare_result_train = np.concatenate((x_train[0, 0, :, :], y_hat[0, 0, :, :], y_train[0, 0, :, :]), axis=1)
        # compare_result_train = np.concatenate((y_hat[0, 0, :, :], y_train[0, 0, :, :]), axis=1)
        '''画图'''
        if epoch % 2 == 0:
            plt.figure()
            plt.imshow(compare_result_train, vmin=-1, vmax=1)
            # plt.imshow(compare_result_train, vmin=-0.2, vmax=0.2)
            # plt.colorbar()
            savename1 = '/home/hdd/yanguo/results_cv4/fold'+str(fold)+'/train_results/' + 'train_epoch' + str(epoch)
            plt.savefig(savename1)
            plt.close()
        print('当前epoch次数为:%d' % epoch)
        print('当前训练的loss为:%f' % train_loss_aver)
        message_train = '(epoch:%d,loss_train=%f)' % (epoch, train_loss_aver)
        with open('/home/hdd/yanguo/results_cv4/fold'+str(fold)+'/loss_results/log.txt', 'a') as log_f:
            log_f.write('%s\n' % message_train)
    
        # print(x_train.shape, y_hat.shape, y_train.shape)
        '''验证部分'''
        val_loss_all = 0
    
        for i, data in enumerate(val_loader):
            recon_net.eval()  # 验证模式
            x_val, y_val = data
            x_val = x_val.type(torch.FloatTensor)
            y_val = y_val.type(torch.FloatTensor)
            x_val = x_val.to(device)
            y_val = y_val.to(device)
            optimizer.zero_grad()
            y_hat_val = recon_net(x_val)
            val_loss = criterion(y_hat_val, y_val)
            # print('当前验证的loss为:%f' % val_loss)
            val_loss_all += val_loss.detach().cpu().numpy()
        val_loss_aver = val_loss_all / i
        val_loss_save.append(val_loss_aver)  # 学习率曲线，方便画图
        # scheduler.step()
        # scheduler.step(val_loss_aver)  # 学习率动态衰减，观察验证集的loss，判断要不要更新学习率
        lr = optimizer.param_groups[0]['lr']  # 学习率动态衰减,更新学习率
        print(epoch, lr)  # 检查学习率
        x_val = x_val.cpu().detach().numpy()
        y_hat_val = y_hat_val.cpu().detach().numpy()
        y_val = y_val.cpu().detach().numpy()
        # print(x_val.shape,y_hat_val.shape,y_val.shape)
        compare_result_val = np.concatenate((x_val[0, 0, :, :], y_hat_val[0, 0, :, :], y_val[0, 0, :, :]), axis=1)
        '''画图'''
        if epoch % 2 == 0:
            plt.figure()
            # plt.imshow(compare_result_val, vmin=-0.2, vmax=0.2)
            plt.imshow(compare_result_val, vmin=-1, vmax=1)
            # plt.colorbar()
            savename2 = '/home/hdd/yanguo/results_cv4/fold'+str(fold)+'/train_results/' + 'val_epoch' + str(epoch)
            plt.savefig(savename2)
            plt.close()
        print('当前验证的loss为:%f' % val_loss_aver)
        message_val = '(epoch:%d,lr=%f,loss_val=%f)' % (epoch, lr, val_loss_aver)
        with open('/home/hdd/yanguo/results_cv4/fold'+str(fold)+'/loss_results/log.txt', 'a') as log_f:
            log_f.write('%s\n' % message_val)
        # 保存val_loss最低时的模型
        if val_loss_best > val_loss_aver:
            best_epoch = epoch
            message_update = '(epoch:%d:model_best update！！！！！！！！！！！！！！！！)' % (epoch)
            with open(r'/home/hdd/yanguo/results_cv4/fold'+str(fold)+'/loss_results/log.txt', 'a') as log_f:
                log_f.write('%s\n' % message_update)
            val_loss_best = val_loss_aver
            save_path = '/home/hdd/yanguo/results_cv4/fold'+str(fold)+'/model_results/'
            save_model_name = model_name+'_best'  # option:u-net twolayersnet resnet
            save_path_and_model_name = save_path + save_model_name
            '''方式一'''
            # torch.save(recon_net.state_dict(), save_path_and_model_name)
            '''方式二'''
            torch.save(recon_net, save_path_and_model_name)
        # if epoch==200:
        #     save_path = r'../..//model_results/'
        #     save_model_name = 'test_epoch200'  # option:u-net twolayersnet resnet
        #     save_path_and_model_name = save_path + save_model_name
        #     '''方式一'''
        #     # torch.save(recon_net.state_dict(), save_path_and_model_name)
        #     '''方式二'''
        #     torch.save(recon_net, save_path_and_model_name)
    message_update = '(epoch_best:%d:val_loss_best=%f)' % (epoch_best,val_loss_best)
    with open(r'/home/hdd/yanguo/results_cv4/fold' + str(fold) + '/loss_results/log.txt', 'a') as log_f:
        log_f.write('%s\n' % message_update)
    '''保存学习率曲线数据'''
    np.save('/home/hdd/yanguo/results_cv4/fold'+str(fold)+'/loss_results/train_loss.npy', train_loss)
    np.save('/home/hdd/yanguo/results_cv4/fold'+str(fold)+'/loss_results/val_loss_save.npy', val_loss_save)
    #############最终模型的保存############
    '''方式一'''
    # save_path=r'/home/hdd/yanguo/results_cv4/fold'+str(fold)+'/model_results/'
    # save_model_name='model'
    # save_path_and_model_name=save_path+save_model_name
    # torch.save(recon_net.state_dict(),save_path_and_model_name)
    '''方式二'''
    save_path = '/home/hdd/yanguo/results_cv4/fold'+str(fold)+'/model_results/'
    # save_model_name='direct_model_fc_mix_simple_breast_noisebysalstm'
    save_model_name = model_name
    save_path_and_model_name = save_path + save_model_name
    torch.save(recon_net, save_path_and_model_name)
    #################模型的读取################
    # check_model=twolayernet(ttds_x,ttds_y)
    # check_model.load_state_dict(torch.load(save_path_and_model_name))
    # check_model.state_dict()#查看模型参数
    '''学习率曲线绘制'''
    plt.figure()
    epoch_times = np.arange(epoch)
    plt.plot(epoch_times, train_loss)
    plt.plot(epoch_times, val_loss_save)
    plt.legend(['train_loss', 'val_loss'])
    savename2 = '/home/hdd/yanguo/results_cv4/fold'+str(fold)+'/loss_results/train_val_loss.png'
    plt.savefig(savename2)
    plt.show()
    plt.close()

    print('第{0}折交叉验证结束'.format(fold))