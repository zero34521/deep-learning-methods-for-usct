import numpy as np
'''UTiknet部分'''
# path = '../../Results_saved/pre_direct/2020.12.16.1/loss_results'
# train_loss_direct = np.load(path+'/train_loss_direct.npy',allow_pickle = True)
# train_loss_pre = np.load(path+'/train_loss_pre.npy',allow_pickle = True)
# val_loss_save_direct = np.load(path+'/val_loss_save_direct.npy',allow_pickle = True)
# val_loss_save_pre = np.load(path+'/val_loss_save_pre.npy',allow_pickle = True)
# train_loss_direct = train_loss_direct.astype(float)
# train_loss_pre = train_loss_pre.astype(float)
# val_loss_save_direct = val_loss_save_direct.astype(float)
# val_loss_save_pre = val_loss_save_pre.astype(float)
# np.save(path+'/train_loss_direct.npy',train_loss_direct)
# np.save(path+'/train_loss_pre.npy',train_loss_pre)
# np.save(path+'/val_loss_save_direct.npy',val_loss_save_direct)
# np.save(path+'/val_loss_save_pre.npy',val_loss_save_pre)

'''TikUnet部分'''
#10.22.1以后的都转换完成了
# path = '../../Results_saved/direct_post/2020.10.22.1/loss_results'
# train_loss_direct = np.load(path+'/train_loss_direct.npy',allow_pickle = True)
# train_loss_post = np.load(path+'/train_loss_post.npy',allow_pickle = True)
# val_loss_save_direct = np.load(path+'/val_loss_save_direct.npy',allow_pickle = True)
# val_loss_save_post = np.load(path+'/val_loss_save_post.npy',allow_pickle = True)
# train_loss_direct = train_loss_direct.astype(float)
# train_loss_post = train_loss_post.astype(float)
# val_loss_save_direct = val_loss_save_direct.astype(float)
# val_loss_save_post = val_loss_save_post.astype(float)
# np.save(path+'/train_loss_direct.npy',train_loss_direct)
# np.save(path+'/train_loss_post.npy',train_loss_post)
# np.save(path+'/val_loss_save_direct.npy',val_loss_save_direct)
# np.save(path+'/val_loss_save_post.npy',val_loss_save_post)

'''AUTOMAP部分'''
path = '../../Results_saved/direct/2020.11.27.1/loss_results'
train_loss = np.load(path+'/train_loss.npy',allow_pickle = True)
val_loss_save = np.load(path+'/val_loss_save.npy',allow_pickle = True)
train_loss = train_loss.astype(float)
val_loss_save = val_loss_save.astype(float)
np.save(path+'/train_loss.npy',train_loss)
np.save(path+'/val_loss_save.npy',val_loss_save)
