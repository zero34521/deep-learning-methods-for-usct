recon_hat=recon_hat.cpu().detach().numpy()
plt.figure()
plt.imshow(recon_hat[0, 0, :, :])
savename1 = '/home/user/yanguo/pycharm_connection/reconstruction_USCT/results/train_results/' + 'train' + str(epoch)
plt.savefig(savename1)
label = label.cpu().detach().numpy()
plt.figure()
plt.imshow(label[0, 0, :, :])
savename2 = '/home/user/yanguo/pycharm_connection/reconstruction_USCT/results/train_results/' + 'label' + str(epoch)
plt.savefig(savename2)
plt.close()

#为上面的模型定制一个第一层
class first_layer(nn.Module):
    def __init__(self,input_nc, ngf):
        super().__init__()
        self.conv1=nn.Conv2d(input_nc, 128, kernel_size=3, padding=1)
        self.relu1=nn.ReLU()
        self.conv2=nn.Conv2d(128,ngf,kernel_size=3,padding=1)
        self.ca=ChannelAttention(ngf)
        self.sa=SpatialAttention(kernel_size=3)
        self.relu2=nn.ReLU()
    def forward(self,x):
        x=self.conv1(x)
        x=self.relu1(x)
        x=self.conv2(x)
        x=x*self.ca(x)
        x=x*self.sa(x)
        x=self.relu2(x)
        return x


class twolynet3d(nn.Module):
    def __init__(self,ttds_x,ttds_y):
        super().__init__()
        self.ttds_x = ttds_x
        self.ttds_y = ttds_y
        self.conv1=nn.Conv3d(1,1,kernel_size=5,stride=1,padding=2)
        self.tanh1=nn.Tanh()
        self.conv2 = nn.Conv3d(1, 1, kernel_size=5, stride=1, padding=2)
        self.tanh2=nn.Tanh()

        self.conv3=nn.Conv2d(128,1,3,padding=1)
        self.tanh3=nn.Tanh()


    def forward(self, x):
        x = x[:, self.ttds_x, self.ttds_y].permute(0, 3, 1, 2)#out:shape(samples,channel,width,length)
        x=x.reshape(x.shape[0],1,x.shape[1],x.shape[2],x.shape[3])
        x=self.conv1(x)
        x=self.tanh1(x)
        x=self.conv2(x)
        x=self.tanh2(x)
        x=x.reshape(x.shape[0],x.shape[2],x.shape[3],x.shape[4])
        x=self.conv3(x)
        x=self.tanh3(x)
        return x

      check=ttdm_train.detach().cpu().numpy()
      plt.figure()
      plt.imshow(check[1,:,:])
      plt.show()

class testnet_reshape(nn.Module):
    def __init__(self,ttds_x,ttds_y):
        super().__init__()
        self.ttds_x=ttds_x
        self.ttds_y=ttds_y
        self.conv0=nn.Conv2d(1,1,3,padding=1)
        self.tanh0 = nn.Tanh()
        self.conv1=nn.Conv2d(256,1,1)
        self.tanh1=nn.Tanh()
    def forward(self,x):
        x=x.view(x.shape[0],1,x.shape[1],x.shape[2])
        x=self.conv0(x)
        x=self.tanh0(x)
        x = x.view(x.shape[0],x.shape[2], x.shape[3])
        x = x[:, self.ttds_x, self.ttds_y].permute(0, 3, 1, 2)
        x=self.conv1(x)
        x=self.tanh1(x)
        return x

print('dddd')
import platform
print(platform.platform())
import tensorflow as tf

a = tf.test.is_built_with_cuda()  # 判断CUDA是否可以用
b = tf.test.is_gpu_available(
    cuda_only=False,
    min_cuda_compute_capability=None
)                                  # 判断GPU是否可以用

print(a)
print(b)
import tensorflow as tf
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '5'

# Creates a graph.
a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)

# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

# Runs the op.
print(sess.run(c))
