import scipy.io as io
import numpy as np
# import torch

# mat1=np.load('E:/研究生/项目/项目_超声CT重构/深度学习后处理/simulated simple data/PTTDMs_hat.npy')
# io.savemat('E:/研究生/项目/项目_超声CT重构/深度学习后处理/simulated simple data/PTTDMs_hat.mat',{'PTTDMs_hat':mat1})

# mat2=np.load('E:/研究生/项目/项目_超声CT重构/深度学习后处理/simulated simple data/PTTDMs_hat.npy')
# io.savemat('E:/研究生/项目/项目_超声CT重构/深度学习后处理/simulated simple data/PTTDMs_hat.mat',{'TTDMs_hat':mat2})

'''损失曲线转换'''
'''UTiknet loss转换'''
load_path_free = 'E:/code/remote_document/reconstruction_USCT/hdd_yanguo/Results_saved_cv/pre_direct/2021.06.09.1/fold1/loss_results'
train_loss_free = np.load(load_path_free + '/train_loss_direct.npy')
val_loss_free = np.load(load_path_free + '/val_loss_save_direct.npy')

# load_path_noise = 'E:/code/remote_document/reconstruction_USCT/hdd_yanguo/Results_saved_cv/pre_direct/2021.03.06.1/fold4/loss_results'
# train_loss_exp_noise = np.load(load_path_noise + '/train_loss_direct.npy')
# val_loss_exp_noise = np.load(load_path_noise + '/val_loss_save_direct.npy')

# train_loss_free = train_loss_free.astype(float)#转之前读取到的是带有device信息的tensor
# val_loss_free = val_loss_free.astype(float)
# train_loss_exp_noise = train_loss_exp_noise.astype(float)
# val_loss_exp_noise = val_loss_exp_noise.astype(float)
save_path = 'E:/研究生/项目/项目_超声CT重构/results/loss_results/UTiknet'
io.savemat(save_path + '/train_loss_free.mat',{'train_loss_free':train_loss_free})
io.savemat(save_path + '/val_loss_free.mat',{'val_loss_free':val_loss_free})
# io.savemat(save_path + '/train_loss_exp_noise.mat',{'train_loss_exp_noise':train_loss_exp_noise})
# io.savemat(save_path + '/val_loss_exp_noise.mat',{'val_loss_exp_noise':val_loss_exp_noise})

'''TikUnet loss转换'''
load_path_free = 'E:/code/remote_document/reconstruction_USCT/hdd_yanguo/Results_saved_cv/direct_post/2021.06.06.1/fold1/loss_results'
train_loss_free = np.load(load_path_free +r'/train_loss_post.npy')
val_loss_free = np.load(load_path_free +r'/val_loss_save_post.npy')
# load_path_noise = 'E:/code/remote_document/reconstruction_USCT/hdd_yanguo/Results_saved_cv/direct_post/2021.03.06.1/fold4/loss_results'
# train_loss_exp_noise = np.load(load_path_noise +'/train_loss_post.npy')
# val_loss_exp_noise = np.load(load_path_noise +'/val_loss_save_post.npy')

# train_loss_free = train_loss_free.astype(float)#转之前读取到的是带有device信息的tensor
# val_loss_free = val_loss_free.astype(float)
# train_loss_exp_noise = train_loss_exp_noise.astype(float)
# val_loss_exp_noise = val_loss_exp_noise.astype(float)
save_path = 'E:/研究生/项目/项目_超声CT重构/results/loss_results/TikUnet'
io.savemat(save_path +'/train_loss_free.mat',{'train_loss_free':train_loss_free})
io.savemat(save_path +'/val_loss_free.mat',{'val_loss_free':val_loss_free})
# io.savemat(save_path +'/train_loss_exp_noise.mat',{'train_loss_exp_noise':train_loss_exp_noise})
# io.savemat(save_path +'/val_loss_exp_noise.mat',{'val_loss_exp_noise':val_loss_exp_noise})

'''AUTOMAP loss转换(旧)'''
# load_path_free = 'E:/code/remote_document/reconstruction_USCT/hdd_yanguo/Results_saved_cv/direct/2021.03.28.1/fold1/loss_results'
# train_loss_free = np.load(load_path_free +r'/train_loss.npy')
# val_loss_free = np.load(load_path_free +r'/val_loss_save.npy')
# # load_path_noise = 'E:/code/remote_document/reconstruction_USCT/hdd_yanguo/Results_saved_cv/direct/2021.03.28.1/fold1/loss_results'
# # train_loss_exp_noise = np.load(load_path_noise +'/train_loss.npy')
# # val_loss_exp_noise = np.load(load_path_noise +'/val_loss_save.npy')
#
# # train_loss_free = train_loss_free.astype(float)#转之前读取到的是带有device信息的tensor
# # val_loss_free = val_loss_free.astype(float)
# # train_loss_exp_noise = train_loss_exp_noise.astype(float)
# # val_loss_exp_noise = val_loss_exp_noise.astype(float)
# save_path = 'E:/研究生/项目/项目_超声CT重构/results/loss_results/AUTOMAP'
# io.savemat(save_path +'/train_loss_free.mat',{'train_loss_free':train_loss_free})
# io.savemat(save_path +'/val_loss_free.mat',{'val_loss_free':val_loss_free})
# # io.savemat('/train_loss_exp_noise.mat',{'train_loss_exp_noise':train_loss_exp_noise})
# # io.savemat('/val_loss_exp_noise.mat',{'val_loss_exp_noise':val_loss_exp_noise})


'''DeepPET loss转换'''
# load_path_free = 'E:/code/remote_document/reconstruction_USCT/hdd_yanguo/Results_saved_cv/direct/2021.05.31.1/fold1/loss_results'
# train_loss_free = np.load(load_path_free +r'/train_loss.npy')
# val_loss_free = np.load(load_path_free +r'/val_loss_save.npy')
#
# save_path = 'E:/研究生/项目/项目_超声CT重构/results/loss_results/DeepPET'
# io.savemat(save_path +'/train_loss_free.mat',{'train_loss_free':train_loss_free})
# io.savemat(save_path +'/val_loss_free.mat',{'val_loss_free':val_loss_free})

