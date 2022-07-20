import scipy.io as io
import numpy as np
import h5py
from matplotlib import pylab as plt

'''真麻烦，写一个函数算了'''
def mat2npy(data_name, load_mat_path, save_npy_path):
    # 函数功能：将mat数据转换为npy数据并保存
    # 输入：变量名，输入的mat路径 ，保存的npy路径
    # 格式要求：mat的文件名应该与变量名相同，路径的结尾不需要加/，变量名后面不需要加后缀
    mat_data = io.loadmat(load_mat_path + '/' + data_name + '.mat')
    data = mat_data[data_name]
    npy_data = np.transpose(data)
    np.save(save_npy_path+ '/' + data_name + '.npy', npy_data)
    print('convert done')


def mat2npy_h5(data_name, load_mat_path, save_npy_path):
    # 函数功能：将mat数据转换为npy数据并保存,针对v7.3的mat数据，利用h5py来读取
    # 输入：变量名，输入的mat路径 ，保存的npy路径
    # 格式要求：mat的文件名应该与变量名相同，路径的结尾不需要加/，变量名后面不需要加后缀
    mat_data = h5py.File(load_mat_path + '/' + data_name + '.mat')
    data = mat_data[data_name]
    npy_data = np.transpose(data)
    np.save(save_npy_path+ '/' + data_name + '.npy', npy_data)
    print('convert done')

# load_path=r'D:/研究生/超声CT重构/项目_超声CT重构/深度学习后处理/exp_data'
# matr1 = io.loadmat(load_path+r'/slowness_sart_s_aic_exp_128.mat')
# data1 = matr1['slowness_sart_s_aic_exp_128']
# slowness_sart_s_aic_exp_128 = np.transpose(data1)
# save_path=r'D:/研究生/超声CT重构/reconstruction_USCT/data/exp_data'
# np.save(save_path+r'/slowness_sart_s_aic_exp_128.npy',slowness_sart_s_aic_exp_128)



'''四种噪声_简单仿真混合复杂乳腺_噪声是由salstm作为真值得到的！！！！！！！！！！！'''
# load_path=r'D:/研究生/超声CT重构/项目_超声CT重构/深度学习前处理/simu_simple_breast_data/四种噪声'
# matr1 = io.loadmat(load_path+r'/PTTDMs_simple_breast_free_free_train_val.mat')
# matr2 = io.loadmat(load_path+r'/PTTDMs_simple_breast_free_free_test.mat')
# matr3 = io.loadmat(load_path+r'/TTDMs_simple_breast_noise_free_train_val_bysalstm.mat')
# matr4 = io.loadmat(load_path+r'/TTDMs_simple_breast_noise_free_test_bysalstm.mat')
# # matr5 = io.loadmat(load_path+r'/TTDMs_free_train_val.mat')
# # matr6 = io.loadmat(load_path+r'/TTDMs_free_test.mat')
# #
# data1 = matr1['PTTDMs_simple_breast_free_free_train_val']
# data2 = matr2['PTTDMs_simple_breast_free_free_test']
# data3 = matr3['TTDMs_simple_breast_noise_free_train_val_bysalstm']
# data4 = matr4['TTDMs_simple_breast_noise_free_test_bysalstm']
# # data5 = matr5['TTDMs_free_train_val']
# # data6 = matr6['TTDMs_free_test']
# #
# PTTDMs_simple_breast_free_free_train_val = np.transpose(data1)
# PTTDMs_simple_breast_free_free_test = np.transpose(data2)
# TTDMs_simple_breast_noise_free_train_val_bysalstm = np.transpose(data3)
# TTDMs_simple_breast_noise_free_test_bysalstm = np.transpose(data4)
# # TTDMs_free_train_val = np.transpose(data5)
# # TTDMs_free_test = np.transpose(data6)
# save_path=r'D:/研究生/超声CT重构/reconstruction_USCT/data/simu_simple_breast_pre_process/四种噪声'
# np.save(save_path+r'/PTTDMs_simple_breast_free_free_train_val.npy',PTTDMs_simple_breast_free_free_train_val)
# np.save(save_path+r'/PTTDMs_simple_breast_free_free_test.npy',PTTDMs_simple_breast_free_free_test)
# np.save(save_path+r'/TTDMs_simple_breast_noise_free_train_val_bysalstm.npy',TTDMs_simple_breast_noise_free_train_val_bysalstm)
# np.save(save_path+r'/TTDMs_simple_breast_noise_free_test_bysalstm.npy',TTDMs_simple_breast_noise_free_test_bysalstm)
# # np.save(save_path+r'/TTDMs_free_train_val.npy',TTDMs_free_train_val)
# # np.save(save_path+r'/TTDMs_free_test.npy',TTDMs_free_test)

'''慢度_四种噪声（实际用了三种）_简单仿真混合复杂乳腺_噪声是由salstm作为真值得到的！！！！！！！！！！！'''
# load_path=r'D:/研究生/超声CT重构/项目_超声CT重构/深度学习后处理/simu_simple_breast_data/四种噪声'
# matr1 = io.loadmat(load_path+r'/slowness_labels_simp_bre_train_val.mat')
# matr2 = io.loadmat(load_path+r'/slowness_labels_simp_bre_test.mat')
# matr3 = io.loadmat(load_path+r'/slowness_sart_s_simp_bre_noise_free_train_val_bysalstm.mat')
# matr4 = io.loadmat(load_path+r'/slowness_sart_s_simp_bre_noise_free_test_bysalstm.mat')
#
#
# data1 = matr1['slowness_labels_simp_bre_train_val']
# data2 = matr2['slowness_labels_simp_bre_test']
# data3 = matr3['slowness_sart_s_simp_bre_noise_free_train_val_bysalstm']
# data4 = matr4['slowness_sart_s_simp_bre_noise_free_test_bysalstm']
#
#
# slowness_labels_simp_bre_train_val = np.transpose(data1)
# slowness_labels_simp_bre_test = np.transpose(data2)
# slowness_sart_s_simp_bre_noise_free_train_val_bysalstm = np.transpose(data3)
# slowness_sart_s_simp_bre_noise_free_test_bysalstm = np.transpose(data4)
#
# save_path=r'D:/研究生/超声CT重构/reconstruction_USCT/data/simu_simple_breast_post_process/四种噪声'
# np.save(save_path+r'/slowness_labels_simp_bre_train_val.npy',slowness_labels_simp_bre_train_val)
# np.save(save_path+r'/slowness_labels_simp_bre_test.npy',slowness_labels_simp_bre_test)
# np.save(save_path+r'/slowness_sart_s_simp_bre_noise_free_train_val_bysalstm.npy',slowness_sart_s_simp_bre_noise_free_train_val_bysalstm)
# np.save(save_path+r'/slowness_sart_s_simp_bre_noise_free_test_bysalstm.npy',slowness_sart_s_simp_bre_noise_free_test_bysalstm)


'''慢度_实验数据_由SART-S重建的AIC结果（128*128）！！！！！！！！！！！'''
# load_path=r'D:/研究生/超声CT重构/项目_超声CT重构/深度学习后处理/exp_data'
# matr1 = io.loadmat(load_path+r'/slowness_sart_s_aic_exp_128.mat')
# data1 = matr1['slowness_sart_s_aic_exp_128']
# slowness_sart_s_aic_exp_128 = np.transpose(data1)
# save_path=r'D:/研究生/超声CT重构/reconstruction_USCT/data/exp_data'
# np.save(save_path+r'/slowness_sart_s_aic_exp_128.npy',slowness_sart_s_aic_exp_128)

'''前处理的输出，用于后处理（仿真_训练部分）：慢度_四种噪声（实际用了三种）_简单仿真混合复杂乳腺_噪声是由salstm作为真值得到的！！！！！！！！！！！'''
# load_mat_path=r'D:/研究生/超声CT重构/项目_超声CT重构/前处理_后处理/post_input'
# save_npy_path=r'D:/研究生/超声CT重构/reconstruction_USCT/data/pre_post'
# mat2npy('slowness_pttdm_sart_s_simp_bre_noise_free_train_val_bysalstm',load_mat_path,save_npy_path)

'''前处理的输出，用于后处理（仿真_测试部分）！！！！！！！！！！！'''
# load_mat_path=r'D:/研究生/超声CT重构/项目_超声CT重构/前处理_后处理/post_input'
# save_npy_path=r'D:/研究生/超声CT重构/reconstruction_USCT/data/pre_post'
# mat2npy('slowness_pttdm_jihong_simp_bre_noise_free_test_bysalstm',load_mat_path,save_npy_path)

'''前处理的输出，用于后处理（实验）！！！！！！！！！！！'''
# load_mat_path=r'D:/研究生/超声CT重构/项目_超声CT重构/前处理_后处理/post_input'
# save_npy_path=r'D:/研究生/超声CT重构/reconstruction_USCT/data/pre_post'
# mat2npy('recon_slowness_pttdm_jihong_exp',load_mat_path,save_npy_path)

'''全连接变换的矩阵初值！！！！！！！！！！！'''
# load_mat_path=r'D:/students/yanguo/超声CT重构/深度学习后处理'
# save_npy_path=r'D:/students/yanguo/reconstruction_USCT/data/FC_matrix'
# mat2npy_h5('T_inv_gama5',load_mat_path,save_npy_path)

'''前处理输出的PTTDM，也就是前处理的结果（训练验证集部分）'''
# load_mat_path=r'D:/研究生/超声CT重构/项目_超声CT重构/结果记录/pre_post_results/2020.5.8/pre_output/sim/train'
# save_npy_path=r'D:/研究生/超声CT重构/项目_超声CT重构/结果记录/pre_post_results/2020.5.8/pre_output/sim/train'
# mat2npy('PTTDMs_hat_train_val_for_post',load_mat_path,save_npy_path)

'''前处理输出的PTTDM，也就是前处理的结果（测试集部分）'''
# load_mat_path=r'D:/研究生/超声CT重构/项目_超声CT重构/结果记录/pre_process_results/2020.4.22/PTTDMs_hat_mat_results/sim'
# save_npy_path=r'D:/研究生/超声CT重构/项目_超声CT重构/结果记录/pre_process_results/2020.4.22/PTTDMs_hat_mat_results/sim'
# mat2npy('PTTDMs_hat',load_mat_path,save_npy_path)

'''删除了重复病人后的数据集，简单+乳腺，无噪声'''
# load_mat_path=r'D:/students/yanguo/超声CT重构/前处理_直接重构/simu_simple_breast_data/free'
# save_npy_path=r'D:/students/yanguo/reconstruction_USCT/data/simu_simp_breast_pre_direct/free'
# mat2npy('PTTDMs_simp_bre_free_train',load_mat_path,save_npy_path)
# mat2npy('PTTDMs_simp_bre_free_val',load_mat_path,save_npy_path)
# mat2npy('PTTDMs_simp_bre_free_test',load_mat_path,save_npy_path)
#
# mat2npy('slowness_labels_simp_bre_train',load_mat_path,save_npy_path)
# mat2npy('slowness_labels_simp_bre_val',load_mat_path,save_npy_path)
# mat2npy('slowness_labels_simp_bre_test',load_mat_path,save_npy_path)
#
# mat2npy('TTDMs_simp_bre_free_train',load_mat_path,save_npy_path)
# mat2npy('TTDMs_simp_bre_free_val',load_mat_path,save_npy_path)
# mat2npy('TTDMs_simp_bre_free_test',load_mat_path,save_npy_path)

'''删除了重复病人后的数据集，简单+乳腺，无噪声'''
# load_mat_path=r'E:/研究生/项目/项目_超声CT重构/data/simu_simple_breast_data/noise'
# save_npy_path=r'../../data/simu_simp_bre_pre_direct/noise'
#
# mat2npy('TTDMs_simp_bre_noise_train',load_mat_path,save_npy_path)
# mat2npy('TTDMs_simp_bre_noise_val',load_mat_path,save_npy_path)
# mat2npy('TTDMs_simp_bre_noise_test',load_mat_path,save_npy_path)

'''删除了重复病人后的数据集，简单+乳腺，有噪声（satblstm）'''
# load_mat_path=r'D:/students/yanguo/超声CT重构/前处理_直接重构/simu_simple_breast_data/noise'
# save_npy_path=r'D:/students/yanguo/reconstruction_USCT/data/simu_simp_bre_pre_direct/noise'
# mat2npy('PTTDMs_simp_bre_train_free_free',load_mat_path,save_npy_path)
# mat2npy('PTTDMs_simp_bre_val_free_free',load_mat_path,save_npy_path)
# mat2npy('PTTDMs_simp_bre_test_free_free',load_mat_path,save_npy_path)
#
# mat2npy('slowness_labels_simp_bre_train_free_free',load_mat_path,save_npy_path)
# mat2npy('slowness_labels_simp_bre_val_free_free',load_mat_path,save_npy_path)
# mat2npy('slowness_labels_simp_bre_test_free_free',load_mat_path,save_npy_path)
#
# mat2npy('TTDMs_simp_bre_train_free_noise',load_mat_path,save_npy_path)
# mat2npy('TTDMs_simp_bre_val_free_noise',load_mat_path,save_npy_path)
# mat2npy('TTDMs_simp_bre_test_free_noise',load_mat_path,save_npy_path)

'''数据扩充16倍（无噪声）'''
# load_mat_path=r'E:/研究生/项目/项目_超声CT重构/data/simu_simple_breast_data_aug_16times/free'
# save_npy_path=r'E:/code/remote_document/reconstruction_USCT/data/simu_simp_bre_aug_16times/free'
# mat2npy_h5('PTTDMs_simp_bre_free_train_aug',load_mat_path,save_npy_path)
# mat2npy_h5('PTTDMs_simp_bre_free_val_aug',load_mat_path,save_npy_path)
#
# mat2npy_h5('slowness_labels_simp_bre_train_aug',load_mat_path,save_npy_path)
# mat2npy_h5('slowness_labels_simp_bre_val_aug',load_mat_path,save_npy_path)
#
# mat2npy_h5('TTDMs_simp_bre_free_train_aug',load_mat_path,save_npy_path)
# mat2npy_h5('TTDMs_simp_bre_free_val_aug',load_mat_path,save_npy_path)

'''数据扩充16倍（有噪声）'''
# load_mat_path=r'E:/研究生/项目/项目_超声CT重构/data/simu_simple_breast_data_aug_16times/exp_noise'
# save_npy_path=r'E:/code/remote_document/reconstruction_USCT/data/simu_simp_bre_aug_16times/exp_noise'
#
# mat2npy_h5('TTDMs_simp_bre_exp_noise_train_aug',load_mat_path,save_npy_path)
# mat2npy_h5('TTDMs_simp_bre_exp_noise_val_aug',load_mat_path,save_npy_path)


'''数据扩充64倍（无噪声）'''
# load_mat_path=r'E:/研究生/项目/项目_超声CT重构/data/simu_simple_breast_data_aug/free'
# save_npy_path=r'E:/code/remote_document/reconstruction_USCT/data/simu_simp_bre_aug_64times/free'
# mat2npy_h5('PTTDMs_simp_bre_free_train_aug',load_mat_path,save_npy_path)
# mat2npy_h5('PTTDMs_simp_bre_free_val_aug',load_mat_path,save_npy_path)
#
# mat2npy_h5('slowness_labels_simp_bre_train_aug',load_mat_path,save_npy_path)
# mat2npy_h5('slowness_labels_simp_bre_val_aug',load_mat_path,save_npy_path)
#
# mat2npy_h5('TTDMs_simp_bre_free_train_aug',load_mat_path,save_npy_path)
# mat2npy_h5('TTDMs_simp_bre_free_val_aug',load_mat_path,save_npy_path)

'''数据扩充64倍（有噪声）'''
# load_mat_path=r'E:/研究生/项目/项目_超声CT重构/data/simu_simple_breast_data_aug/exp_noise'
# save_npy_path=r'E:/code/remote_document/reconstruction_USCT/data/simu_simp_bre_aug_64times/exp_noise'
# mat2npy_h5('TTDMs_simp_bre_exp_noise_train_aug',load_mat_path,save_npy_path)
# mat2npy_h5('TTDMs_simp_bre_exp_noise_val_aug',load_mat_path,save_npy_path)


'''数据扩充128倍（无噪声）'''
# load_mat_path=r'E:/研究生/项目/项目_超声CT重构/data/simu_simple_breast_data_aug_128times/free'
# save_npy_path='E:/code/remote_document/reconstruction_USCT/data/simu_simp_bre_aug_128times/free'
# mat2npy_h5('TTDMs_simp_bre_free_train_aug',load_mat_path,save_npy_path)
# mat2npy_h5('TTDMs_simp_bre_free_val_aug',load_mat_path,save_npy_path)
#
# mat2npy_h5('PTTDMs_simp_bre_free_train_aug',load_mat_path,save_npy_path)
# mat2npy_h5('PTTDMs_simp_bre_free_val_aug',load_mat_path,save_npy_path)
#
# mat2npy_h5('slowness_labels_simp_bre_train_aug',load_mat_path,save_npy_path)
# mat2npy_h5('slowness_labels_simp_bre_val_aug',load_mat_path,save_npy_path)

'''数据扩充128倍（有噪声）'''
# load_mat_path=r'E:/研究生/项目/项目_超声CT重构/data/simu_simple_breast_data_aug_128times/exp_noise'
# save_npy_path='E:/code/remote_document/reconstruction_USCT/data/simu_simp_bre_aug_128times/exp_noise'
# mat2npy_h5('TTDMs_simp_bre_exp_noise_train_aug',load_mat_path,save_npy_path)
# mat2npy_h5('TTDMs_simp_bre_exp_noise_val_aug',load_mat_path,save_npy_path)


'''仿真添加1dB噪声'''
# load_mat_path=r'E:/研究生/项目/项目_超声CT重构/data/simu_simple_breast_data/noise_1dB'
# save_npy_path=r'E:/code/remote_document/reconstruction_USCT/data/simu_simp_bre_pre_direct/noise_1dB'
# mat2npy('TTDMs_simp_bre_noise_train_1dB',load_mat_path,save_npy_path)
# mat2npy('TTDMs_simp_bre_noise_val_1dB',load_mat_path,save_npy_path)
# mat2npy('TTDMs_simp_bre_noise_test_1dB',load_mat_path,save_npy_path)

'''仿真添加4dB噪声'''
# load_mat_path=r'E:/研究生/项目/项目_超声CT重构/data/simu_simple_breast_data/noise_4dB'
# save_npy_path=r'E:/code/remote_document/reconstruction_USCT/data/simu_simp_bre_pre_direct/noise_4dB'
# mat2npy('TTDMs_simp_bre_noise_train_4dB',load_mat_path,save_npy_path)
# mat2npy('TTDMs_simp_bre_noise_val_4dB',load_mat_path,save_npy_path)
# mat2npy('TTDMs_simp_bre_noise_test_4dB',load_mat_path,save_npy_path)

'''仿真添加7dB噪声'''
# load_mat_path=r'E:/研究生/项目/项目_超声CT重构/data/simu_simple_breast_data/noise_7dB'
# save_npy_path=r'E:/code/remote_document/reconstruction_USCT/data/simu_simp_bre_pre_direct/noise_7dB'
# mat2npy('TTDMs_simp_bre_noise_train_7dB',load_mat_path,save_npy_path)
# mat2npy('TTDMs_simp_bre_noise_val_7dB',load_mat_path,save_npy_path)
# mat2npy('TTDMs_simp_bre_noise_test_7dB',load_mat_path,save_npy_path)

'''仿真添加10dB噪声'''
# load_mat_path=r'E:/研究生/项目/项目_超声CT重构/data/simu_simple_breast_data/noise_10dB'
# save_npy_path=r'E:/code/remote_document/reconstruction_USCT/data/simu_simp_bre_pre_direct/noise_10dB'
# mat2npy('TTDMs_simp_bre_noise_train_10dB',load_mat_path,save_npy_path)
# mat2npy('TTDMs_simp_bre_noise_val_10dB',load_mat_path,save_npy_path)
# mat2npy('TTDMs_simp_bre_noise_test_10dB',load_mat_path,save_npy_path)

'''仿真添加1dB噪声(16倍扩充)'''
# load_mat_path=r'E:/研究生/项目/项目_超声CT重构/data/simu_simple_breast_data_aug_16times/noise_1dB'
# save_npy_path=r'E:/code/remote_document/reconstruction_USCT/data/simu_simp_bre_aug_16times/noise_1dB'
# mat2npy_h5('TTDMs_simp_bre_exp_noise_train_aug',load_mat_path,save_npy_path)
# mat2npy_h5('TTDMs_simp_bre_exp_noise_val_aug',load_mat_path,save_npy_path)

'''仿真添加4dB噪声(16倍扩充)'''
# load_mat_path=r'E:/研究生/项目/项目_超声CT重构/data/simu_simple_breast_data_aug_16times/noise_4dB'
# save_npy_path=r'E:/code/remote_document/reconstruction_USCT/data/simu_simp_bre_aug_16times/noise_4dB'
# mat2npy_h5('TTDMs_simp_bre_exp_noise_train_aug',load_mat_path,save_npy_path)
# mat2npy_h5('TTDMs_simp_bre_exp_noise_val_aug',load_mat_path,save_npy_path)


'''仿真添加10dB噪声(16倍扩充)'''
# load_mat_path=r'E:/研究生/项目/项目_超声CT重构/data/simu_simple_breast_data_aug_16times/noise_10dB'
# save_npy_path=r'E:/code/remote_document/reconstruction_USCT/data/simu_simp_bre_aug_16times/noise_10dB'
# mat2npy_h5('TTDMs_simp_bre_exp_noise_train_aug',load_mat_path,save_npy_path)
# mat2npy_h5('TTDMs_simp_bre_exp_noise_val_aug',load_mat_path,save_npy_path)

'''手动标注的实验标签，并且经过了数据扩充'''
# load_mat_path=r'E:/研究生/项目/项目_超声CT重构/data/exp_data_mannual_label_32times'
# save_npy_path=r'E:/code/remote_document/reconstruction_USCT/data/exp_data_mannual_label_32times'
# mat2npy('TTDM_phantom_simple_aug',load_mat_path,save_npy_path)
# mat2npy('PTTDM_phantom_simple_aug',load_mat_path,save_npy_path)
# mat2npy('slowness_label_phantom_simple_aug',load_mat_path,save_npy_path)

'''仿真添加7dB噪声（这里的噪声除去了坏道）'''
# load_mat_path=r'E:/研究生/项目/项目_超声CT重构/data/simu_simple_breast_data/noise_7dB/20210301'
# save_npy_path=r'E:/code/remote_document/reconstruction_USCT/data/simu_simp_bre_pre_direct/noise_7dB/20210301'
# mat2npy('TTDMs_simp_bre_noise_train_7dB',load_mat_path,save_npy_path)
# mat2npy('TTDMs_simp_bre_noise_val_7dB',load_mat_path,save_npy_path)
# mat2npy('TTDMs_simp_bre_noise_test_7dB',load_mat_path,save_npy_path)

'''仿真添加7dB噪声(16倍扩充)（这里的噪声除去了坏道）'''
# load_mat_path=r'E:/研究生/项目/项目_超声CT重构/data/simu_simple_breast_data_aug_16times/noise_7dB/20210301'
# save_npy_path=r'E:/code/remote_document/reconstruction_USCT/data/simu_simp_bre_aug_16times/noise_7dB/20210301'
# mat2npy_h5('TTDMs_simp_bre_exp_noise_train_aug',load_mat_path,save_npy_path)
# mat2npy_h5('TTDMs_simp_bre_exp_noise_val_aug',load_mat_path,save_npy_path)


'''交叉验证数据转换'''
'''无噪声'''
# for i in range(1,6):
#     load_mat_path=r'E:/研究生/项目/项目_超声CT重构/data/cross_val_data/free/fold'+str(i)
#     save_npy_path=r'E:/code/remote_document/reconstruction_USCT/data/cross_val_data/free/fold'+str(i)
#     mat2npy('TTDMs_mix_fold'+str(i)+'_train',load_mat_path,save_npy_path)
#     mat2npy('TTDMs_mix_fold'+str(i)+'_test',load_mat_path,save_npy_path)
#     mat2npy('PTTDMs_mix_fold'+str(i)+'_train',load_mat_path,save_npy_path)
#     mat2npy('PTTDMs_mix_fold'+str(i)+'_test',load_mat_path,save_npy_path)
#     mat2npy('slowness_mix_label_fold'+str(i)+'_train',load_mat_path,save_npy_path)
#     mat2npy('slowness_mix_label_fold'+str(i)+'_test',load_mat_path,save_npy_path)
'''有噪声'''
# snrs = [1,4,7,10]
# for snr in snrs:
#     for i in range(1, 6):
#         load_mat_path = 'E:/研究生/项目/项目_超声CT重构/data/cross_val_data/noise_'+str(snr)+ 'dB/fold' + str(i)
#         save_npy_path = 'E:/code/remote_document/reconstruction_USCT/data/cross_val_data/noise_'+str(snr)+ 'dB/fold' + str(i)
#         mat2npy('TTDMs_mix_'+str(snr)+'dB_fold' + str(i) + '_train', load_mat_path, save_npy_path)
#         mat2npy('TTDMs_mix_'+str(snr)+'dB_fold' + str(i) + '_test', load_mat_path, save_npy_path)

'''交叉验证数据转换(16倍扩充版)'''
#变量名里取消了fold和dB
'''无噪声'''
# aug_times = 16
# for i in range(1,6):
#     load_mat_path='E:/研究生/项目/项目_超声CT重构/data/cross_val_data_aug_'+str(aug_times)+'times/free/fold'+str(i)
#     save_npy_path=r'E:/code/remote_document/reconstruction_USCT/data/cross_val_data_aug_'+str(aug_times)+'times/free/fold'+str(i)
#     mat2npy_h5('TTDMs_mix_train_aug',load_mat_path,save_npy_path)
#     mat2npy_h5('PTTDMs_mix_train_aug',load_mat_path,save_npy_path)
#     mat2npy_h5('slowness_mix_label_train_aug',load_mat_path,save_npy_path)
'''有噪声'''
aug_times = 16
snrs = [1,4,7,10]
for snr in snrs:
    for i in range(1, 6):
        load_mat_path = 'E:/研究生/项目/项目_超声CT重构/data/cross_val_data_aug_'+str(aug_times)+'times/noise_'+str(snr)+ 'dB/fold' + str(i)
        save_npy_path = 'E:/code/remote_document/reconstruction_USCT/data/cross_val_data_aug_'+str(aug_times)+'times/noise_'+str(snr)+ 'dB/fold' + str(i)
        mat2npy_h5('TTDMs_mix_noise_train_aug', load_mat_path, save_npy_path)