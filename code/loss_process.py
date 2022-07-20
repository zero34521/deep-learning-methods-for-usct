import numpy as np
import matplotlib.pyplot as plt
#前处理
train_loss_pre_free = np.load('/home/user/yanguo/pycharm_connection/reconstruction_USCT/results/Results_saved/pre_direct/2020.11.03.1/loss_results/train_loss_pre.npy',allow_pickle=True)
train_loss_pre_noise = np.load('/home/user/yanguo/pycharm_connection/reconstruction_USCT/results/Results_saved/pre_direct/2020.10.29.4/loss_results/train_loss_pre.npy',allow_pickle=True)
val_loss_pre_free = np.load('/home/user/yanguo/pycharm_connection/reconstruction_USCT/results/Results_saved/pre_direct/2020.11.03.1/loss_results/val_loss_save_pre.npy',allow_pickle=True)
val_loss_pre_noise = np.load('/home/user/yanguo/pycharm_connection/reconstruction_USCT/results/Results_saved/pre_direct/2020.10.29.4/loss_results/val_loss_save_pre.npy',allow_pickle=True)

plt.figure()
plt.plot(train_loss_pre_free)
plt.plot(train_loss_pre_noise)
plt.plot(val_loss_pre_free)
plt.plot(val_loss_pre_noise)
plt.legend(['train_loss_pre_free', 'train_loss_pre_noise','val_loss_pre_free', 'val_loss_pre_noise'])
pic_savename1 = '../results/Results_saved/20201126/train_val_loss_pre_free_noise.png'
plt.savefig(pic_savename1)
plt.show()
plt.close()

#后处理
train_loss_post_free = np.load('/home/user/yanguo/pycharm_connection/reconstruction_USCT/results/Results_saved/direct_post/2020.11.02.3/loss_results/train_loss_post.npy',allow_pickle=True)
train_loss_post_noise = np.load('/home/user/yanguo/pycharm_connection/reconstruction_USCT/results/Results_saved/direct_post/2020.11.02.1/loss_results/train_loss_post.npy',allow_pickle=True)
val_loss_post_free = np.load('/home/user/yanguo/pycharm_connection/reconstruction_USCT/results/Results_saved/direct_post/2020.11.02.3/loss_results/val_loss_save_post.npy',allow_pickle=True)
val_loss_post_noise = np.load('/home/user/yanguo/pycharm_connection/reconstruction_USCT/results/Results_saved/direct_post/2020.11.02.1/loss_results/val_loss_save_post.npy',allow_pickle=True)

plt.figure()
plt.plot(train_loss_post_free)
plt.plot(train_loss_post_noise)
plt.plot(val_loss_post_free)
plt.plot(val_loss_post_noise)
plt.legend(['train_loss_post_free', 'train_loss_post_noise','val_loss_post_free', 'val_loss_post_noise'])
pic_savename2 = '../results/Results_saved/20201126/train_val_loss_post_free_noise.png'
plt.savefig(pic_savename2)
plt.show()
plt.close()