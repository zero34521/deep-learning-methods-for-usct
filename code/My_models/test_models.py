from My_models.unet_parts import *
import torch.nn as nn
import matplotlib.pyplot as plt

class testnet(nn.Module):
    def __init__(self,ttds_x,ttds_y):
        super().__init__()
        self.ttds_x=ttds_x
        self.ttds_y=ttds_y

    def forward(self,ttdm):
        ttds=ttdm[:,self.ttds_x,self.ttds_y].permute(0,3,1,2)
        return ttds

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes,ttds_x,ttds_y):
        super(UNet, self).__init__()
        self.ttds_x=ttds_x
        self.ttds_y=ttds_y
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)
        self.nb_class=n_classes

    def forward(self, x):
        x = x[:, self.ttds_x, self.ttds_y].permute(0, 3, 1, 2)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        if self.nb_class==1:# use the sigmoid for dice loss
            x = F.tanh(x)
        return x


class twolayernet(nn.Module):
    def __init__(self,ttds_x,ttds_y):
        super().__init__()
        self.ttds_x=ttds_x
        self.ttds_y=ttds_y
        self.conv1=nn.Conv2d(128,64,3,padding=1)
        #self.bn1=nn.BatchNorm2d(64)
        #self.tanh1=nn.Tanh()
        self.relu1=nn.ReLU()
        self.conv2=nn.Conv2d(64,1,3,padding=1)
        self.tanh2=nn.Tanh()
        #self.relu2 = nn.ReLU()
    def forward(self,x):
        x = x[:, self.ttds_x, self.ttds_y].permute(0, 3, 1, 2)
        x=self.conv1(x)
        #x=self.bn1(x)
        #x=self.tanh1(x)
        x = self.relu1(x)
        x=self.conv2(x)
        #x = self.relu2(x)
        x=self.tanh2(x)
        return x


class fourlayernet(nn.Module):
    def __init__(self,ttds_x,ttds_y):
        super().__init__()
        self.ttds_x=ttds_x
        self.ttds_y=ttds_y
        self.conv1=nn.Conv2d(128,64,3,padding=1)
        self.relu1=nn.ReLU()
        self.conv2=nn.Conv2d(64,32,3,padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(32,16,3,padding=1)
        self.relu3 = nn.ReLU()
        self.conv4=nn.Conv2d(16,1,3,padding=1)
    def forward(self,x):
        x = x[:, self.ttds_x, self.ttds_y].permute(0, 3, 1, 2)
        x=self.conv1(x)
        #x=self.bn1(x)
        #x=self.tanh1(x)
        x = self.relu1(x)
        x=self.conv2(x)
        x = self.relu2(x)
        x=self.conv3(x)
        x = self.relu3(x)
        x=self.conv4(x)
        return x

class resnet1(nn.Module):
    def __init__(self,ttds_x,ttds_y):
        super().__init__()
        self.ttds_x=ttds_x
        self.ttds_y=ttds_y
        self.conv1=nn.Conv2d(128,128,3,padding=1)
        self.relu1=nn.ReLU()
        self.conv2=nn.Conv2d(128,128,3,padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(128,32,3,padding=1)
        self.relu3 = nn.ReLU()
        self.conv4=nn.Conv2d(32,1,3,padding=1)
    def forward(self,x):
        x = x[:, self.ttds_x, self.ttds_y].permute(0, 3, 1, 2)
        identity=x
        x=self.conv1(x)
        x = self.relu1(x)
        x=self.conv2(x)
        x+=identity
        x = self.relu2(x)
        x=self.conv3(x)
        x = self.relu3(x)
        x=self.conv4(x)
        return x

class resnet3(nn.Module):
    def __init__(self,ttds_x,ttds_y):
        super().__init__()
        self.ttds_x=ttds_x
        self.ttds_y=ttds_y
        self.conv1=nn.Conv2d(128,128,3,padding=1)
        self.relu1=nn.ReLU()
        self.conv2=nn.Conv2d(128,128,3,padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(128,32,3,stride=1,padding=1)
        self.pool3=nn.MaxPool2d(2)
        self.relu3 = nn.ReLU()

        self.conv4=nn.Conv2d(32,32,3,padding=1)
        self.relu4 = nn.ReLU()
        self.conv5=nn.Conv2d(32,32,3,padding=1)
        self.relu5 = nn.ReLU()

        self.up6=nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv6=nn.Conv2d(32,4,3,padding=1)
        self.relu6=nn.ReLU()

        self.conv7=nn.Conv2d(4,1,1)
    def forward(self,x):
        x = x[:, self.ttds_x, self.ttds_y].permute(0, 3, 1, 2)
        identity1=x
        x=self.conv1(x)
        x = self.relu1(x)
        x=self.conv2(x)
        x+=identity1
        x = self.relu2(x)

        x=self.conv3(x)
        x=self.pool3(x)
        x = self.relu3(x)
        identity2=x
        x=self.conv4(x)
        x=self.relu4(x)
        x=self.conv5(x)
        x+=identity2
        x=self.relu5(x)

        x=self.up6(x)
        x=self.conv6(x)
        x=self.relu6(x)
        x=self.conv7(x)

        return x

class twolayersnetsd(nn.Module):
    def __init__(self,ttds_x,ttds_y):
        super().__init__()
        self.ttds_x=ttds_x
        self.ttds_y=ttds_y
        self.tanh0=nn.Tanh()
        self.conv1=nn.Conv2d(128,64,3,padding=1)
        self.relu1=nn.ReLU()
        self.conv2=nn.Conv2d(64,1,3,padding=1)
        self.tanh2=nn.Tanh()
    def forward(self,x):
        x=x[:, self.ttds_x, self.ttds_y].permute(0, 3, 1, 2)
        x=self.tanh0(x)
        x=self.conv1(x)
        x=self.relu1(x)
        x=self.conv2(x)
        #x=self.tanh2(x)
        return x

class resnet1_att(nn.Module):
    def __init__(self,ttds_x,ttds_y):
        super().__init__()
        self.ttds_x=ttds_x
        self.ttds_y=ttds_y
        self.sa=SpatialAttention()
        self.conv1=nn.Conv2d(128,128,3,padding=1)
        self.relu1=nn.ReLU()
        self.conv2=nn.Conv2d(128,128,3,padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(128,32,3,padding=1)
        self.tanh3 = nn.Tanh()
        self.conv4=nn.Conv2d(32,1,3,padding=1)
        self.tanh4=nn.Tanh()
    def forward(self,x):
        x = x[:, self.ttds_x, self.ttds_y].permute(0, 3, 1, 2)
        identity=x
        x=self.conv1(x)
        x = self.relu1(x)
        x=self.conv2(x)
        x=self.sa(x)*x
        x+=identity
        x = self.relu2(x)
        x=self.conv3(x)
        x = self.tanh3(x)
        x=self.conv4(x)
        x=self.tanh4(x)
        return x

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1=nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2=nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class recon_resnet_sa(nn.Module):
    def __init__(self, ttds_x, ttds_y):
        super().__init__()
        self.ttds_x = ttds_x
        self.ttds_y = ttds_y
        ngf = 64
        n_blocks = 4
        input_nc = 128
        output_nc = 1
        model = [nn.Conv2d(input_nc, ngf, kernel_size=3, padding=1),
                 nn.ReLU(True)]
        for i in range(n_blocks):
            model += [ResnetBlock_sa(ngf)]
        model += [nn.Conv2d(ngf, 16, kernel_size=3, padding=1),
                  nn.Tanh()]
        model += [nn.Conv2d(16, output_nc, kernel_size=3, padding=1)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)
        print(self.model)

    def forward(self, x):
        # if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
        #    return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        # else:
        x = x[:, self.ttds_x, self.ttds_y].permute(0, 3, 1, 2)
        return self.model(x)


class ResnetBlock_sa(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv_block = self.build_conv_block(dim)
        self.sa=SpatialAttention(kernel_size=3)
        self.ca=ChannelAttention(dim)
    def build_conv_block(self, dim):
        conv_block = []
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=1),
                       nn.ReLU(True)]
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=1)]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        res_conv=self.conv_block(x)
        res_conv_ca=res_conv*self.ca(res_conv)
        res_conv_sa=res_conv*self.sa(res_conv_ca)
        out = x + res_conv_sa
        return out

class ResnetBlock_sa_IN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv_block = self.build_conv_block(dim)
        self.sa=SpatialAttention(kernel_size=3)
    def build_conv_block(self, dim):
        conv_block = []
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=1),
                       nn.ReLU(True),
                       nn.InstanceNorm2d(dim)]
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=1),
                       nn.InstanceNorm2d(dim)]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        res_conv=self.conv_block(x)
        res_conv_sa=res_conv*self.sa(res_conv)
        out = x + res_conv_sa
        return out


class recon_resnet_sa_u_net(nn.Module):
    def __init__(self, ttds_x, ttds_y):
        super().__init__()
        self.ttds_x = ttds_x
        self.ttds_y = ttds_y
        ngf = 64
        n_blocks = 4
        input_nc = 128
        output_nc = 1
        model = [nn.Conv2d(input_nc, ngf, kernel_size=3, padding=1),
                 nn.ReLU(True)]
        model+=[nn.MaxPool2d(2)]
        for i in range(n_blocks):
            model += [ResnetBlock_sa(ngf)]
        model+=[nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)]
        model += [nn.Conv2d(ngf, 16, kernel_size=3, padding=1),
                  nn.Tanh()]
        model += [nn.Conv2d(16, output_nc, kernel_size=3, padding=1)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)
        self.inc = inconv(1, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, 1)
        self.nb_class=1


    def forward(self, x):
        # if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
        #    return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        # else:
        x = x[:, self.ttds_x, self.ttds_y].permute(0, 3, 1, 2)
        x=self.model(x)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        x = F.tanh(x)
        return x

class resnet3_tanh(nn.Module):
    def __init__(self,ttds_x,ttds_y):
        super().__init__()
        self.ttds_x=ttds_x
        self.ttds_y=ttds_y
        self.conv1=nn.Conv2d(128,128,3,padding=1)
        self.tanh1=nn.Tanh()
        self.conv2=nn.Conv2d(128,128,3,padding=1)
        self.tanh2 = nn.Tanh()
        self.conv3 = nn.Conv2d(128,32,3,stride=1,padding=1)
        self.pool3=nn.MaxPool2d(2)
        self.tanh3 = nn.Tanh()

        self.conv4=nn.Conv2d(32,32,3,padding=1)
        self.tanh4 = nn.Tanh()
        self.conv5=nn.Conv2d(32,32,3,padding=1)
        self.tanh5 = nn.Tanh()

        self.up6=nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv6=nn.Conv2d(32,4,3,padding=1)
        self.tanh6=nn.Tanh()

        self.conv7=nn.Conv2d(4,1,1)
        self.tanh7=nn.Tanh()
    def forward(self,x):
        x = x[:, self.ttds_x, self.ttds_y].permute(0, 3, 1, 2)
        identity1=x
        x=self.conv1(x)
        x = self.tanh1(x)
        x=self.conv2(x)
        x+=identity1
        x = self.tanh2(x)

        x=self.conv3(x)
        x=self.pool3(x)
        x = self.tanh3(x)
        identity2=x
        x=self.conv4(x)
        x=self.tanh4(x)
        x=self.conv5(x)
        x+=identity2
        x=self.tanh5(x)

        x=self.up6(x)
        x=self.conv6(x)
        x=self.tanh6(x)
        x=self.conv7(x)
        x=self.tanh7(x)
        return x

class recon_resnet_sa_up_down(nn.Module):
    def __init__(self, ttds_x, ttds_y):
        super().__init__()
        self.ttds_x = ttds_x
        self.ttds_y = ttds_y
        ngf = 64
        n_blocks = 9
        input_nc = 128
        output_nc = 1
        model = [nn.Conv2d(input_nc, ngf, kernel_size=3, padding=1),
                 nn.ReLU(True)]
        model+=[nn.MaxPool2d(2)]
        for i in range(n_blocks):
            model += [ResnetBlock_sa(ngf)]
        model+=[nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)]
        model += [nn.Conv2d(ngf, 16, kernel_size=3, padding=1),
                  nn.Tanh()]
        model += [nn.Conv2d(16, output_nc, kernel_size=3, padding=1)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)
        print(self.model)

    def forward(self, x):
        # if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
        #    return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        # else:
        x = x[:, self.ttds_x, self.ttds_y].permute(0, 3, 1, 2)
        return self.model(x)


class recon_resnet_sa_up_down_ttdm(nn.Module):
    def __init__(self):
        super().__init__()
        ngf = 64
        n_blocks = 9
        input_nc = 1
        output_nc = 1
        model = [nn.Conv2d(input_nc, ngf, kernel_size=3, padding=1),
                 nn.ReLU(True)]
        model+=[nn.MaxPool2d(2)]
        for i in range(n_blocks):
            model += [ResnetBlock_sa(ngf)]
        model+=[nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)]
        model += [nn.Conv2d(ngf, 16, kernel_size=3, padding=1),
                  nn.Tanh()]
        model += [nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)]
        model += [nn.Conv2d(16, output_nc, kernel_size=3, padding=1)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)
        print(self.model)

    def forward(self, x):
        # if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
        #    return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        # else:
        x = x.reshape(x.shape[0],1,x.shape[1],x.shape[2])
        return self.model(x)

class recon_resnet_sa_up_down_norm(nn.Module):
    def __init__(self, ttds_x, ttds_y):
        super().__init__()
        self.ttds_x = ttds_x
        self.ttds_y = ttds_y
        ngf = 64
        n_blocks = 4
        input_nc = 128
        output_nc = 1
        model = [nn.Conv2d(input_nc, ngf, kernel_size=3, padding=1),
                 nn.InstanceNorm2d(ngf),
                 nn.ReLU(True)]
        model+=[nn.MaxPool2d(2)]
        for i in range(n_blocks):
            model += [ResnetBlock_sa_IN(ngf)]
        model+=[nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)]
        model += [nn.Conv2d(ngf, 16, kernel_size=3, padding=1),
                  nn.InstanceNorm2d(16),
                  nn.Tanh()]
        model += [nn.Conv2d(16, output_nc, kernel_size=3, padding=1)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)
        print(self.model)

    def forward(self, x):
        # if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
        #    return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        # else:
        x = x[:, self.ttds_x, self.ttds_y].permute(0, 3, 1, 2)
        return self.model(x)

class recon_resnet_sa_up_down_256(nn.Module):
    def __init__(self, ttds_x, ttds_y):
        super().__init__()
        self.ttds_x = ttds_x
        self.ttds_y = ttds_y
        ngf = 64
        n_blocks = 9
        input_nc = 256
        output_nc = 1
        model = [nn.Conv2d(input_nc, ngf, kernel_size=3, padding=1),
                 nn.ReLU(True)]
        model+=[nn.MaxPool2d(2)]
        for i in range(n_blocks):
            model += [ResnetBlock_sa(ngf)]
        model+=[nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)]
        model += [nn.Conv2d(ngf, 16, kernel_size=3, padding=1),
                  nn.Tanh()]
        model += [nn.Conv2d(16, output_nc, kernel_size=3, padding=1)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)
        print(self.model)

    def forward(self, x):
        # if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
        #    return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        # else:
        x = x[:, self.ttds_x, self.ttds_y].permute(0, 3, 1, 2)
        return self.model(x)

class testnet(nn.Module):
    def __init__(self,ttds_x,ttds_y):
        super().__init__()
        self.ttds_x=ttds_x
        self.ttds_y=ttds_y
        self.conv1=nn.Conv2d(256,256,3,padding=1,groups=256)
        self.tanh1=nn.Tanh()
        self.conv2=nn.Conv2d(256,1,1)
        self.tanh2=nn.Tanh()
    def forward(self,x):
        x = x[:, self.ttds_x, self.ttds_y].permute(0, 3, 1, 2)
        x=self.conv1(x)
        x=self.tanh1(x)
        x=self.conv2(x)
        x=self.tanh2(x)
        return x

class recon_resnet_sa_up_down_256_input1(nn.Module):
    def __init__(self, ttds_x, ttds_y):
        super().__init__()
        self.ttds_x = ttds_x
        self.ttds_y = ttds_y
        ngf = 64
        n_blocks = 9
        input_nc = 1
        output_nc = 1
        model = [nn.Conv2d(input_nc, ngf, kernel_size=3, padding=1),
                 nn.ReLU(True)]
        model+=[nn.MaxPool2d(2)]
        for i in range(n_blocks):
            model += [ResnetBlock_sa(ngf)]
        model+=[nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)]
        model += [nn.Conv2d(ngf, 16, kernel_size=3, padding=1),
                  nn.Tanh()]
        model += [nn.Conv2d(16, output_nc, kernel_size=3, padding=1)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)
        print(self.model)

    def forward(self, x):
        # if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
        #    return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        # else:
        x = x[:, self.ttds_x, self.ttds_y].permute(0, 3, 1, 2)
        x=torch.sum(x,1)/128
        x=x.view(x.shape[0],1,x.shape[1],x.shape[2])
        check=x.detach().cpu().numpy().reshape(256,256)
        plt.figure()
        plt.imshow(check)
        plt.colorbar()
        savename2 = '/home/user/yanguo/pycharm_connection/reconstruction_USCT/results/test_results/' + 'check'
        plt.savefig(savename2)
        plt.close()
        return self.model(x)

class testnet_mul(nn.Module):
    def __init__(self,ttds_x,ttds_y):
        super().__init__()
        self.ttds_x=ttds_x
        self.ttds_y=ttds_y
        self.x1_k=nn.Conv2d(1,1,1,bias=False)
        self.x2_conv1=nn.Conv2d(256,1,1)
        self.x2_tanh1=nn.Tanh()
        self.x3_conv1=nn.Conv2d(256,1,3,padding=1)
        self.x3_tanh1=nn.Tanh()
        self.x_all_conv1=nn.Conv2d(3,1,3,padding=1)
        self.x_all_tanh1=nn.Tanh()
    def forward(self,x):
        x = x[:, self.ttds_x, self.ttds_y].permute(0, 3, 1, 2)
        x1=torch.mean(x, dim=1, keepdim=True)
        x1 = self.x1_k(x1)
        # check=x1[0,:,:]
        # check=check.detach().cpu().numpy().reshape(256,256)
        # plt.figure()
        # plt.imshow(check)
        # plt.colorbar()
        # savename2 = '/home/user/yanguo/pycharm_connection/reconstruction_USCT/results/test_results/' + 'check1'
        # plt.savefig(savename2)
        # plt.close()
        # x2=self.x2_conv1(x)
        # x2=self.x2_tanh1(x2)
        # x3=self.x3_conv1(x)
        # x3=self.x3_tanh1(x3)
        # x_all=x1+x2+x3
        # x_all=torch.cat((x1,x2,x3),dim=1)
        # x_all=self.x_all_conv1(x_all)
        # x_all=self.x_all_tanh1(x_all)
        return x1

class testnet_big(nn.Module):#全连接训练速度有的慢啊，得先增大lr才行
    def __init__(self,ttds_x,ttds_y):
        super().__init__()
        self.ttds_x=ttds_x
        self.ttds_y=ttds_y
        self.conv1=nn.Conv2d(256*256*256,256*256*256,1,groups=256*256*256)
    def forward(self,x):
        x = x[:, self.ttds_x, self.ttds_y].permute(0, 3, 1, 2)
        x=x.reshape(x.shape[0],x.shape[1]*x.shape[2]*x.shape[3],1,1)
        x=self.conv1(x)
        x = x.view(x.shape[0],256,256,256)
        x=torch.mean(x, dim=1, keepdim=True)
        return x

class Postprocessnet(nn.Module):
    def __init__(self):
        super().__init__()
        ngf = 64
        n_blocks = 9
        input_nc = 1
        output_nc = 1
        model = [nn.Conv2d(input_nc, ngf, kernel_size=3, padding=1),
                 nn.ReLU(True)]
        model+=[nn.MaxPool2d(2)]
        for i in range(n_blocks):
            model += [ResnetBlock_sa(ngf)]
        model+=[nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)]
        model += [nn.Conv2d(ngf, 16, kernel_size=3, padding=1),
                  nn.Tanh()]
        model += [nn.Conv2d(16, output_nc, kernel_size=3, padding=1)]
        # model += [nn.Tanh()]
        self.model = nn.Sequential(*model)
        print(self.model)

    def forward(self, x):
        return self.model(x)

class UNet_origin(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)
        self.nb_class=n_classes

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        # if self.nb_class==1:# use the sigmoid for dice loss
        #     x = F.tanh(x)
        return x

class DirectNet(nn.Module):
    def __init__(self, TTDM_size):
        super().__init__()
        self.TTDM_size=TTDM_size
        self.fc1 = nn.Linear(TTDM_size * TTDM_size, TTDM_size * TTDM_size, bias=False)
        # self.tanh1 = nn.Tanh()#如果用激活函数的话，那么吉洪诺夫的初值便没有任何意义了
    def forward(self,x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        # x = self.tanh1(x)
        x = x.view(x.size(0), 1 , self.TTDM_size, self.TTDM_size)
        return x

class DirectNet2(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.out_size=out_size
        self.fc1 = nn.Linear(out_size * out_size, in_size * in_size, bias=False)
        # self.tanh1 = nn.Tanh()#如果用激活函数的话，那么吉洪诺夫的初值便没有任何意义了
    def forward(self,x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        # x = self.tanh1(x)
        x = x.view(x.size(0), 1, self.out_size, self.out_size)
        return x

class DirectPostNet(nn.Module):
    def __init__(self, TTDM_size):
        super().__init__()
        self.TTDM_size=TTDM_size
        self.directnet=DirectNet(TTDM_size)
        self.postnet = UNet_origin(1,1)
    def forward(self,x):
        x = self.directnet(x)
        x = self.postnet(x)
        return x

class PreDirectNet(nn.Module):
    def __init__(self, TTDM_size):
        super().__init__()
        self.TTDM_size=TTDM_size
        self.prenet = UNet_origin(1,1)
        self.directnet=DirectNet(TTDM_size)
    def forward(self,x):
        x = self.prenet(x)
        x = self.directnet(x)
        return x

class AUTOMAP(nn.Module):
    def __init__(self, TTDM_size):
        super().__init__()
        self.TTDM_size=TTDM_size
        self.fc1 = nn.Linear(TTDM_size * TTDM_size, TTDM_size * TTDM_size, bias=False)
        self.fc2 = nn.Linear(TTDM_size * TTDM_size, TTDM_size * TTDM_size, bias=False)
        self.tanh1 = nn.Tanh()#如果用激活函数的话，那么吉洪诺夫的初值便没有任何意义了
        self.tanh2 = nn.Tanh()
        self.conv1 = nn.Conv2d(1, 64, (5, 5), padding=2)
        self.conv2 = nn.Conv2d(64, 64, (5, 5), padding=2)
        # self.dconv = nn.ConvTranspose2d(64, 1, (7, 7), padding=3)
        self.conv3 = nn.Conv2d(64, 1, (7, 7), padding=3)
    def forward(self,x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        # x = self.relu1(x)
        x = self.tanh1(x)
        x = self.fc2(x)
        x = self.tanh2(x)
        x = x.view(x.size(0), 1 , self.TTDM_size, self.TTDM_size)
        x = self.conv1(x)
        x = self.conv2(x)
        # x = self.dconv(x)
        x = self.conv3(x)
        return x

# class AUTOMAP(nn.Module):
#     def __init__(self, TTDM_size):
#         super().__init__()
#         self.TTDM_size=TTDM_size
#         self.fc1 = nn.Linear(TTDM_size * TTDM_size, TTDM_size * TTDM_size, bias=False).to('cuda:0')
#         self.fc2 = nn.Linear(TTDM_size * TTDM_size, TTDM_size * TTDM_size, bias=False).to('cuda:1')
#         self.tanh1 = nn.Tanh()#如果用激活函数的话，那么吉洪诺夫的初值便没有任何意义了
#         self.tanh2 = nn.Tanh()
#         self.conv1 = nn.Conv2d(1, 64, (5, 5), padding=2).to('cuda:1')
#         self.conv2 = nn.Conv2d(64, 64, (5, 5), padding=2).to('cuda:1')
#         # self.dconv = nn.ConvTranspose2d(64, 1, (7, 7), padding=3)
#         self.conv3 = nn.Conv2d(64, 1, (7, 7), padding=3).to('cuda:1')
#     def forward(self,x):
#         x = x.view(x.size(0), -1)
#         x = self.fc1(x.to('cuda:0'))
#         # x = self.relu1(x)
#         x = self.tanh1(x)
#         x = self.fc2(x.to('cuda:1'))
#         x = self.tanh2(x)
#         x = x.view(x.size(0), 1 , self.TTDM_size, self.TTDM_size)
#         x = self.conv1(x.to('cuda:1'))
#         x = self.conv2(x.to('cuda:1'))
#         # x = self.dconv(x)
#         x = self.conv3(x.to('cuda:1'))
#         return x

class AUTOMAP_unet(nn.Module):
    def __init__(self, TTDM_size):
        super().__init__()
        self.TTDM_size=TTDM_size
        self.fc1 = nn.Linear(TTDM_size * TTDM_size, TTDM_size * TTDM_size, bias=False)
        self.fc2 = nn.Linear(TTDM_size * TTDM_size, TTDM_size * TTDM_size, bias=False)
        self.tanh1 = nn.Tanh()#如果用激活函数的话，那么吉洪诺夫的初值便没有任何意义了
        self.tanh2 = nn.Tanh()
        self.postnet = UNet_origin(1,1)
    def forward(self,x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        # x = self.relu1(x)
        x = self.tanh1(x)
        x = self.fc2(x)
        x = self.tanh2(x)
        x = x.view(x.size(0), 1 , self.TTDM_size, self.TTDM_size)
        x = self.postnet(x)
        return x
