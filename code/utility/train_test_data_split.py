from sklearn.model_selection import train_test_split
import numpy as np

# '''上面代码和下面代码是一样的，就是改名称，上面的输入是TTDM,下面的是吉洪诺夫的重构结果'''
# ttdms=np.load('/home/user/yanguo/pycharm_connection/reconstruction_USCT/data/simulated_data_breast/ttdms.npy')
# simulation_patterns=np.load('/home/user/yanguo/pycharm_connection/reconstruction_USCT/data/simulated_data_breast/simulation_patterns.npy')
# ttdms_train, ttdms_test, labels_train, labels_test = train_test_split(ttdms, simulation_patterns, test_size=0.1, shuffle=False)
# np.save('/home/user/yanguo/pycharm_connection/reconstruction_USCT/data/simulated_data_breast/ttdms_train.npy',ttdms_train)
# np.save('/home/user/yanguo/pycharm_connection/reconstruction_USCT/data/simulated_data_breast/ttdms_test.npy',ttdms_test)
# np.save('/home/user/yanguo/pycharm_connection/reconstruction_USCT/data/simulated_data_breast/labels_train.npy',labels_train)
# np.save('/home/user/yanguo/pycharm_connection/reconstruction_USCT/data/simulated_data_breast/labels_test.npy',labels_test)


# '''吉洪诺夫输入重构结果'''
# recon_jihong_slowness=np.load('/home/user/yanguo/pycharm_connection/reconstruction_USCT/data/simulated_breast_data_post_process/recon_jihong_slowness.npy')
# simulation_patterns_slowness=np.load('/home/user/yanguo/pycharm_connection/reconstruction_USCT/data/simulated_breast_data_post_process/simulation_patterns_slowness.npy')
# recons_train, recons_test, labels_train, labels_test = train_test_split(recon_jihong_slowness, simulation_patterns_slowness, test_size=0.1, shuffle=True)
# np.save('/home/user/yanguo/pycharm_connection/reconstruction_USCT/data/simulated_breast_data_post_process/recons_train.npy',recons_train)
# np.save('/home/user/yanguo/pycharm_connection/reconstruction_USCT/data/simulated_breast_data_post_process/recons_test.npy',recons_test)
# np.save('/home/user/yanguo/pycharm_connection/reconstruction_USCT/data/simulated_breast_data_post_process/labels_train.npy',labels_train)
# np.save('/home/user/yanguo/pycharm_connection/reconstruction_USCT/data/simulated_breast_data_post_process/labels_test.npy',labels_test)

'''PTTDMs simple输入重构结果'''
# PTTDMs=np.load('/home/user/yanguo/pycharm_connection/reconstruction_USCT/data/simulated_simple_data_pre_process/PTTDMs.npy')
# TTDMs=np.load('/home/user/yanguo/pycharm_connection/reconstruction_USCT/data/simulated_simple_data_pre_process/TTDMs.npy')
# TTDMs_train, TTDMs_test, labels_train, labels_test = train_test_split(TTDMs, PTTDMs, test_size=0.1, shuffle=True)
# np.save('/home/user/yanguo/pycharm_connection/reconstruction_USCT/data/simulated_simple_data_pre_process/TTDMs_train.npy',TTDMs_train)
# np.save('/home/user/yanguo/pycharm_connection/reconstruction_USCT/data/simulated_simple_data_pre_process/TTDMs_test.npy',TTDMs_test)
# np.save('/home/user/yanguo/pycharm_connection/reconstruction_USCT/data/simulated_simple_data_pre_process/labels_train.npy',labels_train)
# np.save('/home/user/yanguo/pycharm_connection/reconstruction_USCT/data/simulated_simple_data_pre_process/labels_test.npy',labels_test)
'''PTTDMs breast输入重构结果'''
PTTDMs=np.load('/home/user/yanguo/pycharm_connection/reconstruction_USCT/data/simulated_breast_data_pre_process/PTTDMs.npy')
TTDMs=np.load('/home/user/yanguo/pycharm_connection/reconstruction_USCT/data/simulated_breast_data_pre_process/TTDMs.npy')
TTDMs_train, TTDMs_test, labels_train, labels_test = train_test_split(TTDMs, PTTDMs, test_size=0.1, shuffle=True)
np.save('/home/user/yanguo/pycharm_connection/reconstruction_USCT/data/simulated_breast_data_pre_process/TTDMs_train.npy',TTDMs_train)
np.save('/home/user/yanguo/pycharm_connection/reconstruction_USCT/data/simulated_breast_data_pre_process/TTDMs_test.npy',TTDMs_test)
np.save('/home/user/yanguo/pycharm_connection/reconstruction_USCT/data/simulated_breast_data_pre_process/labels_train.npy',labels_train)
np.save('/home/user/yanguo/pycharm_connection/reconstruction_USCT/data/simulated_breast_data_pre_process/labels_test.npy',labels_test)