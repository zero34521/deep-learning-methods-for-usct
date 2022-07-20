import torch.nn as nn
class MultiConv(nn.Module):
    def __init__(self, in_ch, out_ch, filter_size, conv_num, downsample_first=False):
        super().__init__()
        if downsample_first:
            model = [nn.Conv2d(in_ch, out_ch, filter_size, padding=(filter_size-1)//2, stride=2),
                     nn.BatchNorm2d(out_ch),
                     nn.ReLU(inplace=True)]
        else:
            model = [nn.Conv2d(in_ch, out_ch, filter_size, padding=(filter_size-1)//2),
                     nn.BatchNorm2d(out_ch),
                     nn.ReLU(inplace=True)]
        if conv_num>1:
            for _ in range(conv_num-1):
                model+=[nn.Conv2d(out_ch, out_ch, filter_size, padding=(filter_size-1)//2),
                        nn.BatchNorm2d(out_ch),
                        nn.ReLU(inplace=True)]
        self.model = nn.Sequential(*model)
    def forward(self, x):
        x = self.model(x)
        return x
class UpMultiConv(nn.Module):
    '''上采样模块'''
    def __init__(self, in_ch, out_ch, filter_size, conv_num, upsample_size):
        super().__init__()
        model = [nn.Upsample(upsample_size,mode='bilinear',align_corners=True),
                 nn.Conv2d(in_ch, out_ch, filter_size, padding=(filter_size-1)//2),
                 nn.BatchNorm2d(out_ch),
                 nn.ReLU(inplace=True)]
        if conv_num>1:
            for _ in range(conv_num-1):
                model+=[nn.Conv2d(out_ch, out_ch, filter_size, padding=(filter_size-1)//2),
                        nn.BatchNorm2d(out_ch),
                        nn.ReLU(inplace=True)]
        self.model = nn.Sequential(*model)
    def forward(self, x):
        x = self.model(x)
        return x

class DeepPET(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            MultiConv(1, 32, 7, conv_num=2, downsample_first=False),
            MultiConv(32, 64, 5, conv_num=2, downsample_first=True),
            MultiConv(64, 128, 5, conv_num=3, downsample_first=True),
            MultiConv(128, 256, 3, conv_num=3, downsample_first=True),
            MultiConv(256, 512, 3, conv_num=3, downsample_first=True),
            MultiConv(512, 1024, 3, conv_num=5, downsample_first=False),
            MultiConv(1024, 512, 3, conv_num=3, downsample_first=False)
        )
        self.decoder = nn.Sequential(
            UpMultiConv(512, 256, 3, conv_num=3, upsample_size=(26, 26)),
            UpMultiConv(256, 128, 3, conv_num=3, upsample_size=(44, 44)),
            UpMultiConv(128, 64, 3, conv_num=2, upsample_size=(75, 75)),
            UpMultiConv(64, 32, 3, conv_num=1, upsample_size=(128, 128)),
            nn.Conv2d(32, 1, 3, padding=1)
        )
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
