import torch.nn as nn
import torch

class recon_resnet(nn.Module):
    def __init__(self,ttds_x,ttds_y):
        super().__init__()
        self.ttds_x = ttds_x
        self.ttds_y = ttds_y
        ngf=64
        n_blocks=4
        input_nc=128
        output_nc=1
        model=[nn.Conv2d(input_nc,ngf,kernel_size=3,padding=1),
               nn.ReLU(True)]
        for i in range(n_blocks):
            model += [ResnetBlock(ngf)]
        model += [nn.Conv2d(ngf, 16, kernel_size=3, padding=1),
                  nn.Tanh()]
        model += [nn.Conv2d(16, output_nc, kernel_size=3, padding=1)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)
        print(self.model)

    def forward(self,x):
        # if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
        #    return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        # else:
        x=x[:,self.ttds_x,self.ttds_y].permute(0,3,1,2)
        return self.model(x)

class ResnetBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv_block = self.build_conv_block(dim)

    def build_conv_block(self, dim):
        conv_block = []
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=1),
                       nn.ReLU(True)]
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=1)]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

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