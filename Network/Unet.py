import torch
import torch.nn as nn


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x



class UNet(nn.Module):
    def __init__(self, img_ch=3, output_ch=1, init_f=32):
        super(UNet, self).__init__()
        f = init_f
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv1 = conv_block(img_ch, f)
        self.Conv2 = conv_block(f, f * 2)
        self.Conv3 = conv_block(f * 2, f * 4)
        self.Conv4 = conv_block(f * 4, f * 8)
        self.bottleneck = conv_block(f * 8, f * 16)

        self.Up4 = nn.ConvTranspose2d(f * 16, f * 8, kernel_size=2, stride=2)
        self.Up_conv4 = conv_block(f * 16, f * 8)

        self.Up3 = nn.ConvTranspose2d(f * 8, f * 4, kernel_size=2, stride=2)
        self.Up_conv3 = conv_block(f * 8, f * 4)

        self.Up2 = nn.ConvTranspose2d(f * 4, f * 2, kernel_size=2, stride=2)
        self.Up_conv2 = conv_block(f * 4, f * 2)

        self.Up1 = nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2)
        self.Up_conv1 = conv_block(f * 2, f)

        self.Conv_1x1 = nn.Conv2d(f, output_ch, kernel_size=1)


    def forward(self, x):
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        bottleneck = self.Maxpool(x4)
        bottleneck = self.bottleneck(bottleneck)

        d = self.Up4(bottleneck)
        d = torch.cat((x4, d), dim=1)
        d = self.Up_conv4(d)

        d = self.Up3(d)
        d = torch.cat((x3, d), dim=1)
        d = self.Up_conv3(d)

        d = self.Up2(d)
        d = torch.cat((x2, d), dim=1)
        d = self.Up_conv2(d)

        d = self.Up1(d)
        d = torch.cat((x1, d), dim=1)
        d = self.Up_conv1(d)

        d = self.Conv_1x1(d)

        return d



if __name__ == '__main__':
    model = UNet(img_ch=1, output_ch=4)
    im = torch.rand((4,1,320,320))
    y = model(im)
    print(y.shape)
