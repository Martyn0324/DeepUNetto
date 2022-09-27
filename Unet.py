# https://link.springer.com/content/pdf/10.1007/978-3-319-24574-4_28.pdf
# Keras version(tutorial): https://towardsdatascience.com/understanding-semantic-segmentation-with-unet-6be4f42d4b47
# Pytorch implementation (Thanks, OpenAI): https://github.com/openai/guided-diffusion/blob/22e0df8183507e13a7813f8d38d51b072ca1e67c/guided_diffusion/unet.py
# PS: OpenAI seems to have made some adaptations. Might be worth studying with tenderness

from torchsummary import summary
import torch
from torch import nn

# Utility functions

def conv2out(input, kernel, stride, padding):
    x = 2*padding
    y = 1*(kernel-1)
    z = (input + x - y - 1)/stride

    output = z + 1
    return output

def transconv2out(input, kernel, stride, padding):
    x = (input-1)*stride
    y = 2*padding
    z = 1*(kernel-1)

    output = x - y + z + 1
    return output
  
  
# Input image = 100x100x3

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1)
        self.conv2 = nn.Conv2d(16, 16, 3, 1, 1)
        self.maxpool3 = nn.MaxPool2d(2, 2) # Perhaps we could use more convs instead of maxpool?
        self.dropout = nn.Dropout(0.1)

        self.conv4 = nn.Conv2d(16, 32, 3, 1, 1)
        self.conv5 = nn.Conv2d(32, 32, 3, 1, 1)
        self.maxpool6 = nn.MaxPool2d(2,2)
        
        self.conv7 = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv8 = nn.Conv2d(64, 64, 3, 1, 1)
        self.maxpool9 = nn.MaxPool2d(2,2)

        self.conv10 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv11 = nn.Conv2d(128, 128, 3, 1, 1)
        self.maxpool12 = nn.MaxPool2d(2,2)

        self.conv13 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv14 = nn.Conv2d(256, 256, 3, 1, 1)
        
        self.transconv15 = nn.ConvTranspose2d(256, 128, 2, 2, 0)
        self.conv16 = nn.Conv2d(256, 128, 3, 1, 1)
        self.conv17 = nn.Conv2d(128, 128, 3, 1, 1)

        self.transconv18 = nn.ConvTranspose2d(128, 64, 3, 2, 0)
        self.conv19 = nn.Conv2d(128, 64, 3, 1, 1)
        self.conv20 = nn.Conv2d(64, 64, 3, 1, 1)

        self.transconv21 = nn.ConvTranspose2d(64, 32, 2, 2, 0)
        self.conv22 = nn.Conv2d(64, 32, 3, 1, 1)
        self.conv23 = nn.Conv2d(32, 32, 3, 1, 1)

        self.transconv24 = nn.ConvTranspose2d(32, 16, 2, 2, 0)
        self.conv25 = nn.Conv2d(32, 16, 3, 1, 1)
        self.conv26 = nn.Conv2d(16, 16, 3, 1, 1)

        self.conv27 = nn.Conv2d(16, 1, 1, 1, 0)

        self.ReLU = nn.ReLU()
        self.sigmoid = nn.Sigmoid()


    def forward(self, input):
        x = self.conv1(input)
        x = self.ReLU(x)
        x = self.conv2(x)
        c1 = self.ReLU(x)
        x = self.maxpool3(c1)
        x = self.dropout(x)

        x = self.conv4(x)
        x = self.ReLU(x)
        x = self.conv5(x)
        c2 = self.ReLU(x)
        x = self.maxpool6(c2)
        x = self.dropout(x)

        x = self.conv7(x)
        x = self.ReLU(x)
        x = self.conv8(x)
        c3 = self.ReLU(x)
        x = self.maxpool9(c3)
        x = self.dropout(x)

        x = self.conv10(x)
        x = self.ReLU(x)
        x = self.conv11(x)
        c4 = self.ReLU(x)
        x = self.maxpool12(c4)
        x = self.dropout(x)

        x = self.conv13(x)
        x = self.ReLU(x)
        x = self.conv14(x)
        c5 = self.ReLU(x)

        u6 = self.transconv15(c5)
        x = torch.cat((u6, c4), 1)
        x = self.dropout(x)

        x = self.conv16(x)
        x = self.ReLU(x)
        x = self.conv17(x)
        c6 = self.ReLU(x)

        u7 = self.transconv18(c6)
        x = torch.cat((u7, c3), 1)
        x = self.dropout(x)

        x = self.conv19(x)
        x = self.ReLU(x)
        x = self.conv20(x)
        c7 = self.ReLU(x)

        u8 = self.transconv21(c7)
        x = torch.cat((u8, c2), 1)
        x = self.dropout(x)

        x = self.conv22(x)
        x = self.ReLU(x)
        x = self.conv23(x)
        c8 = self.ReLU(x)

        u9 = self.transconv24(c8)
        x = torch.cat((u9, c1), 1)
        x = self.dropout(x)

        x = self.conv25(x)
        x = self.ReLU(x)
        x = self.conv26(x)
        c9 = self.ReLU(x)

        x = self.conv27(c9)
        output = self.sigmoid(x)

        return output
      
      
UNet = UNet().cuda()

summary(UNet, (3, 100, 100))
