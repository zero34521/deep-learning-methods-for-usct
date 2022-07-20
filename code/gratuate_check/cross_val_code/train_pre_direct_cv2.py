import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from torchsummary import summary
import matplotlib.pyplot as plt
import sys
sys.path.append("../..")#导入隔壁文件夹下的文件
from data_prepare import Pre_Direct_Datasate
from sklearn.model_selection import train_test_split
from My_models.test_models import Postprocessnet, UNet_origin, DirectNet

device=torch.device("cuda:3")#gpu准备
device_message = {device.type:device.index}
assert (torch.cuda.is_available())  # 判断GPU是否可用

for fold in range(1,6):
    print('第{0}折交叉验证开始'.format(fold))
    '''---------------------------数据读取与处理----------------------'''
    '''无噪声无扩充'''
    # load_data_path1 = '../../../data/cross_val_data/free/fold'+str(fold)
    # X_train = np.load(load_data_path1+'/TTDMs_mix_fold'+str(fold)+'_train.npy')
    # X_val = np.load(load_data_path1+'/TTDMs_mix_fold'+str(fold)+'_test.npy')
    '''1dB噪声无扩充'''
    # load_data_path1 = '../../../data/cross_val_data/noise_1dB/fold'+str(fold)
    # X_train = np.load(load_data_path1+'/TTDMs_mix_1dB_fold'+str(fold)+'_train.npy')
    # X_val = np.load(load_data_path1+'/TTDMs_mix_1dB_fold'+str(fold)+'_test.npy')
    '''4dB噪声无扩充'''
    # load_data_path1 = '../../../data/cross_val_data/noise_4dB/fold'+str(fold)
    # X_train = np.load(load_data_path1+'/TTDMs_mix_4dB_fold'+str(fold)+'_train.npy')
    # X_val = np.load(load_data_path1+'/TTDMs_mix_4dB_fold'+str(fold)+'_test.npy')
    '''7dB噪声无扩充'''
    # load_data_path1 = '../../../data/cross_val_data/noise_7dB/fold'+str(fold)
    # X_train = np.load(load_data_path1+'/TTDMs_mix_7dB_fold'+str(fold)+'_train.npy')
    # X_val = np.load(load_data_path1+'/TTDMs_mix_7dB_fold'+str(fold)+'_test.npy')
    '''10dB噪声无扩充'''
    # load_data_path1 = '../../../data/cross_val_data/noise_10dB/fold'+str(fold)
    # X_train = np.load(load_data_path1+'/TTDMs_mix_10dB_fold'+str(fold)+'_train.npy')
    # X_val = np.load(load_data_path1+'/TTDMs_mix_10dB_fold'+str(fold)+'_test.npy')
    '''无扩充标签'''
    # load_data_path2 = '../../../data/cross_val_data/free/fold' + str(fold)
    # Y1_train = np.load(load_data_path2+'/PTTDMs_mix_fold'+str(fold)+'_train.npy')
    # Y1_val = np.load(load_data_path2+'/PTTDMs_mix_fold'+str(fold)+'_test.npy')
    # Y2_train = np.load(load_data_path2+'/slowness_mix_label_fold'+str(fold)+'_train.npy')
    # Y2_val = np.load(load_data_path2+'/slowness_mix_label_fold'+str(fold)+'_test.npy')

    '''无噪声(16倍扩充)(训练集有扩增，测试集没有扩增)'''
    # load_data_path1 = '/home/hdd/yanguo/cross_val_data_aug_16times/free/fold' + str(fold)
    # load_data_path2 = '../../../data/cross_val_data/free/fold' + str(fold)
    # X_train = np.load(load_data_path1 + '/TTDMs_mix_train_aug.npy')
    # X_val = np.load(load_data_path2 + '/TTDMs_mix_fold' + str(fold) + '_test.npy')
    # Y1_train = np.load(load_data_path1 + '/PTTDMs_mix_train_aug.npy')
    # Y1_val = np.load(load_data_path2 + '/PTTDMs_mix_fold' + str(fold) + '_test.npy')
    # Y2_train = np.load(load_data_path1 + '/slowness_mix_label_train_aug.npy')
    # Y2_val = np.load(load_data_path2 + '/slowness_mix_label_fold' + str(fold) + '_test.npy')
    # X_train = np.transpose(X_train)  # 用hdf5导入的文件，还得转置一下才行
    # Y1_train = np.transpose(Y1_train)  # 用hdf5导入的文件，还得转置一下才行
    # Y2_train = np.transpose(Y2_train)  # 用hdf5导入的文件，还得转置一下才行

    '''7dB噪声(16倍扩充)(训练集有扩增，测试集没有扩增)'''
    load_data_path1 = '/home/hdd/yanguo/cross_val_data_aug_16times/noise_7dB/fold' + str(fold)
    load_data_path2 = '../../../data/cross_val_data/noise_7dB/fold' + str(fold)
    load_data_path3 = '/home/hdd/yanguo/cross_val_data_aug_16times/free/fold' + str(fold)
    load_data_path4 = '../../../data/cross_val_data/free/fold' + str(fold)
    X_train = np.load(load_data_path1 + '/TTDMs_mix_noise_train_aug.npy')
    Y1_train = np.load(load_data_path3 + '/PTTDMs_mix_train_aug.npy')
    Y2_train = np.load(load_data_path3 + '/slowness_mix_label_train_aug.npy')

    X_val = np.load(load_data_path2 + '/TTDMs_mix_7dB_fold' + str(fold) + '_test.npy')
    Y1_val = np.load(load_data_path4 + '/PTTDMs_mix_fold' + str(fold) + '_test.npy')
    Y2_val = np.load(load_data_path4 + '/slowness_mix_label_fold' + str(fold) + '_test.npy')
    X_train = np.transpose(X_train)  # 用hdf5导入的文件，还得转置一下才行
    Y1_train = np.transpose(Y1_train)  # 用hdf5导入的文件，还得转置一下才行
    Y2_train = np.transpose(Y2_train)  # 用hdf5导入的文件，还得转置一下才行


    '''加载吉洪诺夫矩阵作为初值'''
    T_inv = np.load(r'../../../data/FC_matrix/T_inv_gama5.npy')

    '''量级变换'''
    X_train, X_val, Y1_train, Y1_val = X_train * 1e6, X_val * 1e6, Y1_train * 1e6, Y1_val * 1e6  # 量级太小
    Y2_train, Y2_val = Y2_train * 1e8, Y2_val * 1e8

    samples_num = X_train.shape[0]  # 样本数
    # 转换成自己的数据类并加载
    train_data = Pre_Direct_Datasate(X_train, Y1_train, Y2_train)  # 将输入标签放入自定义的TTDM模型类
    val_data = Pre_Direct_Datasate(X_val, Y1_val, Y2_val)
    batch_size = 10
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=0)
    '''模型实例化并放入GPU中，设定损失函数和优化器!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'''
    pre_net=UNet_origin(1,1)
    direct_net=DirectNet(128)
    # print(recon_net)
    #---初始化直接重构全连接层权重---
    T_inv = torch.from_numpy(T_inv)
    T_inv = T_inv.type(torch.FloatTensor)
    direct_net.fc1.weight = torch.nn.Parameter(T_inv * 100)#64大小的图像对应倍数为50，128对应100

    #---初始化前处理权重---
    # save_path = r'/home/hdd/yanguo/results_cv2/fold'+str(fold)+'/results_cv2_saved/pre_direct/2020.10.16.1/model_results/'
    # save_path = r'/home/hdd/yanguo/results_cv2/fold'+str(fold)+'/results_cv2_saved/pre/2020.10.15.1/model_results/'
    # save_model_pre_name = 'Pre_of_model_best'
    # save_path_and_model_pre_name = save_path + save_model_pre_name
    # pre_net_pretrained = torch.load(save_path_and_model_pre_name)
    # pre_model_dict_pretrained=pre_net_pretrained.state_dict()#获取预训练权重
    # pre_net.load_state_dict(pre_model_dict_pretrained)#预训练权重赋值

    '''放入GPU中'''
    pre_net.to(device)
    direct_net.to(device)
    # summary(pre_net.cuda(), input_size=(1,128,128))
    # summary(direct_net.cuda(), input_size=(1,128, 128))

    criterion=nn.MSELoss()
    lr1=1e-3#如果设置了初值，可以直接用1e-4，没有的话建议1e-3
    optimizer1=optim.Adam(pre_net.parameters(),lr=lr1)
    lr2=1e-6
    optimizer2=optim.Adam(direct_net.parameters(),lr=lr2)

    '''训练日志记录'''
    message = '-------------------------------------------------------------------------------------------------------'
    message_batch_size = '(batch_size:%d)' % (batch_size)
    with open(r'/home/hdd/yanguo/results_cv2/fold'+str(fold)+'/loss_results/log.txt', 'a') as log_f:
       log_f.write('%s\n' % message)
       log_f.write('%s\n' % device_message)
       log_f.write('%s\n' % message_batch_size)
    #学习率初始值设定
    train_loss_pre=[]#前处理非线性映射网络部分学习率保存
    train_loss_direct=[]#直接重构线性映射部分学习率保存
    val_loss_save_pre=[]
    val_loss_save_direct=[]
    val_loss_best_pre=10#随便取一个大的初值
    val_loss_best_direct=10#随便取一个大的初值
    epoch_best = 0
    #开始迭代循环训练以及测试
    for epoch in range(1,100):
       train_loss_all_pre = 0
       train_loss_all_direct = 0
       for i,data in enumerate(tqdm(train_loader)):
          pre_net.train()
          direct_net.train()
          x_train, y1_train, y2_train=data
          x_train = x_train.type(torch.FloatTensor)#转换为FloatTensor类型
          y1_train = y1_train.type(torch.FloatTensor)#转换为FloatTensor类型
          y2_train = y2_train.type(torch.FloatTensor)#转换为FloatTensor类型
          x_train = x_train.to(device)#数据置入GPU
          y1_train = y1_train.to(device)#数据置入GPU
          y2_train = y2_train.to(device)#数据置入GPU
          optimizer1.zero_grad()#梯度清零
          optimizer2.zero_grad()#梯度清零
          y1_hat=pre_net(x_train)#得到前处理网络输出结果
          loss1=criterion(y1_hat,y1_train)#
          # loss1.backward()
          loss1.backward(retain_graph=True)
          # y1_hat = y1_hat.detach()#避免梯度传入到loss2中
          y2_hat = direct_net(y1_hat)
          loss2 = criterion(y2_hat, y2_train)
          loss2.backward()
          optimizer1.step()
          optimizer2.step()
          train_loss_all_pre += loss1.detach().cpu().numpy()
          train_loss_all_direct += loss2.detach().cpu().numpy()
       train_loss_aver_pre = train_loss_all_pre / i
       train_loss_pre.append(train_loss_aver_pre)#前处理学习率曲线
       train_loss_aver_direct = train_loss_all_direct / i
       train_loss_direct.append(train_loss_aver_direct)#直接重构学习率曲线
       '''画图'''
       if epoch % 2 == 0:
          x_train=x_train.cpu().detach().numpy()
          y1_hat=y1_hat.cpu().detach().numpy()
          y1_train=y1_train.cpu().detach().numpy()
          y2_hat=y2_hat.cpu().detach().numpy()
          y2_train=y2_train.cpu().detach().numpy()
          compare_result_train1=np.concatenate((x_train[0, 0, :, :],y1_hat[0, 0, :, :],y1_train[0, 0, :, :]),axis=1)
          compare_result_train2 = np.concatenate((y2_hat[0, 0, :, :], y2_train[0, 0, :, :]), axis=1)
          plt.figure()
          plt.imshow(compare_result_train1,vmin=-2,vmax=2)
          plt.colorbar()
          savename1 = r'/home/hdd/yanguo/results_cv2/fold'+str(fold)+'/train_results/'+'train_pre_epoch' + str(epoch)
          plt.savefig(savename1)
          plt.close()
          plt.figure()
          plt.imshow(compare_result_train2,vmin=-2,vmax=2)
          plt.colorbar()
          savename2 = r'/home/hdd/yanguo/results_cv2/fold'+str(fold)+'/train_results/'+'train_direct_epoch' + str(epoch)
          plt.savefig(savename2)
          plt.close()
       print('当前epoch次数为:%d' % epoch)
       print('当前前处理训练的loss为:%f'% train_loss_aver_pre)
       print('当前直接重构训练的loss为:%f' % train_loss_aver_direct)
       message_train='(epoch:%d,loss_train_pre=%f,loss_train_direct=%f)'%(epoch,train_loss_aver_pre,train_loss_aver_direct)
       with open(r'/home/hdd/yanguo/results_cv2/fold'+str(fold)+'/loss_results/log.txt','a') as log_f:
          log_f.write('%s\n'%message_train)

       '''验证部分'''
       val_loss_all_pre=0
       val_loss_all_direct=0
       for i,data in enumerate(val_loader):
          pre_net.eval()#验证模式
          direct_net.eval()  # 验证模式
          x_val, y1_val, y2_val=data
          x_val = x_val.type(torch.FloatTensor)
          y1_val = y1_val.type(torch.FloatTensor)
          y2_val = y2_val.type(torch.FloatTensor)
          x_val = x_val.to(device)
          y1_val = y1_val.to(device)
          y2_val = y2_val.to(device)
          y1_hat_val=pre_net(x_val)
          y2_hat_val = direct_net(y1_hat_val)
          val_loss1 = criterion(y1_hat_val, y1_val)
          val_loss2 = criterion(y2_hat_val, y2_val)
          #print('当前验证的loss为:%f' % val_loss)
          val_loss_all_pre += val_loss1.detach().cpu().numpy()
          # check.append(val_loss1.detach())
          val_loss_all_direct += val_loss2.detach().cpu().numpy()
       val_loss_aver_pre=val_loss_all_pre/i
       val_loss_save_pre.append(val_loss_aver_pre)#损失函数曲线，方便画图
       val_loss_aver_direct=val_loss_all_direct/i
       val_loss_save_direct.append(val_loss_aver_direct)#损失函数曲线，方便画图
       # lr1 = optimizer1.param_groups[0]['lr']#学习率动态衰减,更新学习率
       # lr2 = optimizer2.param_groups[0]['lr']  # 学习率动态衰减,更新学习率
       print(epoch, lr1)#检查学习率
       # print(epoch, lr2)  # 检查学习率
       '''画图'''
       if epoch % 2 == 0:
          x_val=x_val.cpu().detach().numpy()
          y1_hat_val=y1_hat_val.cpu().detach().numpy()
          y2_hat_val = y2_hat_val.cpu().detach().numpy()
          y1_val=y1_val.cpu().detach().numpy()
          y2_val = y2_val.cpu().detach().numpy()
          compare_result_val_pre=np.concatenate((x_val[0, 0, :, :],y1_hat_val[0, 0, :, :],y1_val[0, 0, :, :]),axis=1)
          compare_result_val_direct = np.concatenate((y2_hat_val[0, 0, :, :], y2_val[0, 0, :, :]), axis=1)
          plt.figure()
          plt.imshow(compare_result_val_pre,vmin=-2,vmax=2)
          plt.colorbar()
          savename2 = r'/home/hdd/yanguo/results_cv2/fold'+str(fold)+'/train_results/'+'val_pre_epoch' + str(epoch)
          plt.savefig(savename2)
          plt.close()
          plt.figure()
          plt.imshow(compare_result_val_direct,vmin=-2,vmax=2)
          plt.colorbar()
          savename2 = r'/home/hdd/yanguo/results_cv2/fold'+str(fold)+'/train_results/'+'val_direct_epoch' + str(epoch)
          plt.savefig(savename2)
          plt.close()
       print('当前前处理验证的loss为:%f'%val_loss_aver_pre)
       print('当前直接重构验证的loss为:%f' % val_loss_aver_direct)
       # message_val='(epoch:%d,lr1=%f,lr2=%f,loss_val_pre=%f,loss_val_direct=%f)'%(epoch, lr1, lr2, val_loss_aver_pre, val_loss_aver_direct)
       message_val='(epoch:%d,lr1=%f,loss_val_pre=%f,loss_val_direct=%f)'%(epoch, lr1, val_loss_aver_pre, val_loss_aver_direct)

       with open(r'/home/hdd/yanguo/results_cv2/fold'+str(fold)+'/loss_results/log.txt','a') as log_f:
          log_f.write('%s\n'%message_val)
       '''保存前处理val_loss最低时的模型'''
       # if val_loss_best_pre>val_loss_aver_pre:
       #    best_epoch = epoch
       #    message_update = '(epoch:%d:Pre_of_model_best update!Direct_of_model_when_Pre_best update!)' % (epoch)
       #    with open(r'/home/hdd/yanguo/results_cv2/fold'+str(fold)+'/loss_results/log.txt', 'a') as log_f:
       #       log_f.write('%s\n' % message_update)
       #    val_loss_best_pre=val_loss_aver_pre
       #    save_path = r'/home/hdd/yanguo/results_cv2/fold'+str(fold)+'/model_results/'
       #    save_model_name = 'Pre_of_model_best' # option:u-net twolayersnet resnet
       #    save_path_and_model_name = save_path + save_model_name
       #    torch.save(pre_net,save_path_and_model_name)
       #
       #    save_model_name = 'Direct_of_model_when_pre_best'  # option:u-net twolayersnet resnet
       #    save_path_and_model_name = save_path + save_model_name
       #    '''方式二'''
       #    torch.save(direct_net, save_path_and_model_name)

       '''保存直接重构val_loss最低时的模型'''
       if val_loss_best_direct > val_loss_aver_direct:
          message_update = '(epoch:%d:Direct_of_model_best update!Pre_of_model_when_direct_best update!)' % (epoch)
          with open(r'/home/hdd/yanguo/results_cv2/fold'+str(fold)+'/loss_results/log.txt', 'a') as log_f:
             log_f.write('%s\n' % message_update)
          val_loss_best_direct = val_loss_aver_direct
          save_path = r'/home/hdd/yanguo/results_cv2/fold'+str(fold)+'/model_results/'
          save_model_name = 'Direct_of_model_best'  # option:u-net twolayersnet resnet
          save_path_and_model_name = save_path + save_model_name
          '''方式一'''
          # torch.save(recon_net.state_dict(), save_path_and_model_name)
          '''方式二'''
          torch.save(direct_net, save_path_and_model_name)
          #
          save_model_name = 'Pre_of_model_when_direct_best' # option:u-net twolayersnet resnet
          save_path_and_model_name = save_path + save_model_name
          '''方式一'''
          # torch.save(recon_net.state_dict(), save_path_and_model_name)
          '''方式二'''
          torch.save(pre_net,save_path_and_model_name)
       '''设置保存节点'''
       # if epoch == 100:
       #    #前处理模型保存
       #    save_path = r'/home/hdd/yanguo/results_cv2/fold'+str(fold)+'/model_results/'
       #    save_model_name = 'Pre_of_model_epoch'+str(epoch)  # option:
       #    save_path_and_model_name = save_path + save_model_name
       #    torch.save(pre_net, save_path_and_model_name)
       #    #直接重构模型保存
       #    save_model_name = 'Direct_of_model_epoch'+str(epoch)  # option:
       #    save_path_and_model_name = save_path + save_model_name
       #    torch.save(direct_net, save_path_and_model_name)
    message_update = '(epoch_best:%d:direct_of_model_best=%f)' % (epoch_best,val_loss_best_direct)
    with open(r'/home/hdd/yanguo/results_cv2/fold' + str(fold) + '/loss_results/log.txt', 'a') as log_f:
        log_f.write('%s\n' % message_update)
    '''保存学习率曲线数据'''
    np.save(r'/home/hdd/yanguo/results_cv2/fold'+str(fold)+'/loss_results/train_loss_pre.npy', train_loss_pre)
    np.save(r'/home/hdd/yanguo/results_cv2/fold'+str(fold)+'/loss_results/val_loss_save_pre.npy',val_loss_save_pre)
    np.save(r'/home/hdd/yanguo/results_cv2/fold'+str(fold)+'/loss_results/train_loss_direct.npy', train_loss_direct)
    np.save(r'/home/hdd/yanguo/results_cv2/fold'+str(fold)+'/loss_results/val_loss_save_direct.npy',val_loss_save_direct)
    #############最终模型的保存############
    # '''-----------前处理模型保存-----------'''
    # '''方式一'''
    # # save_path=r'/home/hdd/yanguo/results_cv2/fold'+str(fold)+'/model_results/'
    # # save_model_name='model'
    # # save_path_and_model_name=save_path+save_model_name
    # # torch.save(recon_net.state_dict(),save_path_and_model_name)
    # '''方式二'''
    # save_path=r'/home/hdd/yanguo/results_cv2/fold'+str(fold)+'/model_results/'
    # save_model_name='Pre_of_model'
    # save_path_and_model_name=save_path+save_model_name
    # torch.save(pre_net, save_path_and_model_name)
    # '''-----------直接重构模型保存-----------'''
    # '''方式一'''
    # # save_path=r'/home/hdd/yanguo/results_cv2/fold'+str(fold)+'/model_results/'
    # # save_model_name='model'
    # # save_path_and_model_name=save_path+save_model_name
    # # torch.save(recon_net.state_dict(),save_path_and_model_name)
    # '''方式二'''
    # save_path=r'/home/hdd/yanguo/results_cv2/fold'+str(fold)+'/model_results/'
    # save_model_name='Direct_of_model'
    # save_path_and_model_name=save_path+save_model_name
    # torch.save(direct_net, save_path_and_model_name)
    #################模型的读取################
    # check_model=twolayernet(ttds_x,ttds_y)
    # check_model.load_state_dict(torch.load(save_path_and_model_name))
    #check_model.state_dict()#查看模型参数
    '''学习率曲线绘制'''
    plt.figure()
    plt.plot(train_loss_pre)
    plt.plot(val_loss_save_pre)
    plt.legend(['train_loss_pre', 'val_loss_save_pre'])
    pic_savename1 = '/home/hdd/yanguo/results_cv2/fold'+str(fold)+'/loss_results/train_val_loss_pre.png'
    plt.savefig(pic_savename1)
    plt.show()
    plt.close()

    plt.figure()
    plt.plot(train_loss_direct)
    plt.plot(val_loss_save_direct)
    plt.legend(['train_loss_direct', 'val_loss_save_direct'])
    pic_savename2 = '/home/hdd/yanguo/results_cv2/fold'+str(fold)+'/loss_results/train_val_loss_direct.png'
    plt.savefig(pic_savename2)
    plt.show()
    plt.close()

    print('第{0}折交叉验证结束'.format(fold))