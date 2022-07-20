import os
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
#%%
x = np.linspace(1, 128, num=128, endpoint=True)
y = np.linspace(1, 128, num=128, endpoint=True)
x_new = np.linspace(1, 128, num=256, endpoint=True)
y_new = np.linspace(1, 128, num=256, endpoint=True)
for fold in range(1,6):
    print('第{0}折开始'.format(fold))
    load_data_path3 = '/home/hdd/yanguo/cross_val_data_aug_16times/free/fold' + str(fold)
    load_data_path4 = '/home/user/yanguo/pycharm_connection/reconstruction_USCT/data/cross_val_data/free/fold' + str(fold)
    slowness_train = np.load(load_data_path3 + '/slowness_mix_label_train_aug.npy')
    slowness_train = np.transpose(slowness_train)
    slowness_val = np.load(load_data_path4 + '/slowness_mix_label_fold' + str(fold) + '_test.npy')
    slowness_train_256 = np.zeros((slowness_train.shape[0],256,256), dtype=np.float32)
    slowness_val_256 = np.zeros((slowness_val.shape[0],256,256), dtype=np.float32)

    for i in range(slowness_train.shape[0]):
        slowness = slowness_train[i,:,:]
        interpfunc = interpolate.interp2d(x,y,slowness,kind='linear')
        slowness_train_256[i,:,:] = interpfunc(x_new, y_new)
        # print('done')
    slowness_train_256 = np.transpose(slowness_train_256)

    for i in range(slowness_val.shape[0]):
        slowness = slowness_val[i,:,:]
        interpfunc = interpolate.interp2d(x,y,slowness,kind='linear')
        slowness_val_256[i,:,:] = interpfunc(x_new, y_new)

    np.save(load_data_path3 + '/slowness_mix_label_train_aug_256.npy', slowness_train_256)
    np.save(load_data_path4 + '/slowness_mix_label_fold' + str(fold) + '_test_256.npy', slowness_val_256)

