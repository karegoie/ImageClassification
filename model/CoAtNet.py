from torch import nn
import torch
import sys
from math import sqrt

sys.path.append('.')
sys.path.append('..')
from ImageClassification.model.MBConv import MBConvBlock
from ImageClassification.model.SelfAttention import ScaledDotProductAttention


# initialization
def weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

# Model
class CoAtNet(nn.Module):
    def __init__(self, in_ch, image_size, classes, dropout_rate, out_chs=[64, 96, 192, 384, 768]):
        super(CoAtNet, self).__init__()
        self.out_chs = out_chs
        self.classes = classes
        self.maxpool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpool1d = nn.MaxPool1d(kernel_size=2, stride=2)
        self.gap = nn.AdaptiveMaxPool2d(1)
        self.dropout = nn.Dropout(dropout_rate)

        self.s0 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(in_ch),
            nn.GELU(), #nn.Tanh(),
            nn.Conv2d(in_ch, in_ch, kernel_size=(3, 3), padding=1)
        )
        self.mlp0 = nn.Sequential(
            nn.Conv2d(in_ch, out_chs[0], kernel_size=(1,1)),
            nn.BatchNorm2d(out_chs[0]),
            nn.GELU(), # nn.Tanh()
            nn.Conv2d(out_chs[0], out_chs[0], kernel_size=(1, 1))
        )

        self.s1 = MBConvBlock(ksize=3, input_filters=out_chs[0], output_filters=out_chs[0], image_size=image_size // 2)
        self.mlp1 = nn.Sequential(
            nn.Conv2d(out_chs[0], out_chs[1], kernel_size=(1, 1)),
            nn.BatchNorm2d(out_chs[1]),
            nn.GELU(), # nn.Tanh()
            nn.Conv2d(out_chs[1], out_chs[1], kernel_size=(1, 1))
        )

        self.s2 = MBConvBlock(ksize=3, input_filters=out_chs[1], output_filters=out_chs[1], image_size=image_size // 4)
        self.mlp2 = nn.Sequential(
            nn.Conv2d(out_chs[1], out_chs[2], kernel_size=(1, 1)),
            nn.BatchNorm2d(out_chs[2]),
            nn.GELU(), # nn.Tanh()
            nn.Conv2d(out_chs[2], out_chs[2], kernel_size=(1, 1))
        )

        self.s3 = ScaledDotProductAttention(out_chs[2], out_chs[2] // 8, out_chs[2] // 8, 8)
        self.mlp3 = nn.Sequential(
            nn.Linear(out_chs[2], out_chs[3]),
            nn.LazyBatchNorm1d(),
            nn.GELU(), # nn.Tanh()
            nn.Linear(out_chs[3], out_chs[3])
        )

        self.s4 = ScaledDotProductAttention(out_chs[3], out_chs[3] // 8, out_chs[3] // 8, 8)
        self.mlp4 = nn.Sequential(
            nn.Linear(out_chs[3], out_chs[4]),
            nn.LazyBatchNorm1d(),
            nn.GELU(), # nn.Tanh()
            nn.Linear(out_chs[4], out_chs[4])
        )

        self.mlp5 = nn.Sequential(
            nn.Linear(out_chs[4], self.classes, bias=False),
            nn.Softmax()
        )

    def forward(self, x):
        B, C, H, W = x.shape
        # stage0
        y = self.mlp0(self.s0(x))
        y = self.maxpool2d(y)
        # stage1
        y = self.mlp1(self.s1(y))
        y = self.maxpool2d(y)
        # stage2
        # y = self.dropout(y)
        y = self.mlp2(self.s2(y))
        y = self.maxpool2d(y)
        # stage3
        # y = self.dropout(y)
        y = y.reshape(B, self.out_chs[2], -1).permute(0, 2, 1)  # B,N,C
        y = self.mlp3(self.s3(y, y, y))
        y = self.maxpool1d(y.permute(0, 2, 1)).permute(0, 2, 1)
        # stage4
        # y = self.dropout(y)
        y = self.mlp4(self.s4(y, y, y))
        y = self.maxpool1d(y.permute(0, 2, 1))
        N = y.shape[-1]
        y = y.reshape(B, self.out_chs[4], int(sqrt(N)), int(sqrt(N)))
        # stage5
        y = self.dropout(y)
        y = self.gap(y)
        y = y.reshape(B, self.out_chs[4])
        y = self.mlp5(y)
        # print(type(y))
        return y


if __name__ == '__main__':
    x = torch.randn(4, 3, 224, 224)
    # print(x.shape)
    # coatnet = CoAtNet(3, 224, 8)
    # coatnet.apply(weight_init)
    # y = coatnet(x)
    # print(y.shape)
    # print(y)