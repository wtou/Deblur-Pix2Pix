import matplotlib.pyplot as plt
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from multiprocessing import freeze_support
# G(z)
class generator(nn.Module):
    # initializers
    def __init__(self, d=64):
        super(generator, self).__init__()
        self.layer1 = nn.Conv2d(3, d, 4, 2, 1)
        self.layer2 = nn.Sequential(nn.Conv2d(d, d*2, 4, 2, 1),
                                    nn.BatchNorm2d(d*2))
        self.layer3 = nn.Sequential(nn.Conv2d(d*2, d*4, 4, 2, 1),
                                    nn.BatchNorm2d(d*4))
        self.layer4 = nn.Sequential(nn.Conv2d(d*4, d*8, 4, 2, 1),
                                    nn.BatchNorm2d(d*8))
        self.layer5 = nn.Sequential(nn.Conv2d(d*8, d*8, 4, 2, 1),
                                   nn.BatchNorm2d(d*8))
        self.layer6 = nn.Sequential(nn.Conv2d(d*8, d*8, 4, 2, 1),
                                    nn.BatchNorm2d(d*8))
        self.layer7 = nn.Sequential(nn.Conv2d(d*8, d*8, 4, 2, 1),
                                    nn.BatchNorm2d(d*8))
        self.layer8 = nn.Sequential(nn.Conv2d(d*8, d*8, 4, 2, 1))  
        self.dlayer8 = nn.Sequential(nn.ReLU(),
                                    nn.ConvTranspose2d(d*8, d*8, 4, 2, 1),
                                    nn.BatchNorm2d(d*8),
                                    nn.Dropout2d(0.5, inplace=True))
        self.dlayer7 = nn.Sequential(nn.ReLU(),
                                    nn.ConvTranspose2d(d*16, d*8, 4, 2, 1),
                                    nn.BatchNorm2d(d*8),
                                    nn.Dropout2d(0.5, inplace=True))
        self.dlayer6 = nn.Sequential(nn.ReLU(),
                                    nn.ConvTranspose2d(d*16, d*8, 4, 2, 1),
                                    nn.BatchNorm2d(d*8),
                                    nn.Dropout2d(0.5, inplace=True))
        self.dlayer5 = nn.Sequential(nn.ReLU(),
                                    nn.ConvTranspose2d(d*16, d*8, 4, 2, 1),
                                    nn.BatchNorm2d(d*8))
        self.dlayer4 = nn.Sequential(nn.ReLU(),
                                    nn.ConvTranspose2d(d*16, d*4, 4, 2, 1),
                                    nn.BatchNorm2d(d*4))
        self.dlayer3 = nn.Sequential(nn.ReLU(),
                                    nn.ConvTranspose2d(d*8, d*2, 4, 2, 1),
                                    nn.BatchNorm2d(d*2))
        self.dlayer2 = nn.Sequential(nn.ReLU(),
                                    nn.ConvTranspose2d(d*4, d, 4, 2, 1),
                                    nn.BatchNorm2d(d))
        self.dlayer1 = nn.Sequential(nn.ReLU(),
                                    nn.ConvTranspose2d(d*2, 3, 4, 2, 1),
                                    nn.Tanh())

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        out1 = self.layer1(input)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        out6 = self.layer6(out5)
        out7 = self.layer7(out6)
        out8 = self.layer8(out7)
        dout8 = self.dlayer8(out8)
        dout8_out7 = torch.cat([dout8, out7], 1)
        dout7 = self.dlayer7(dout8_out7)
        dout7_out6 = torch.cat([dout7, out6], 1)
        dout6 = self.dlayer6(dout7_out6)
        dout6_out5 = torch.cat([dout6, out5], 1)
        dout5 = self.dlayer5(dout6_out5)
        dout5_out4 = torch.cat([dout5, out4], 1)
        dout4 = self.dlayer4(dout5_out4)
        dout4_out3 = torch.cat([dout4, out3], 1)
        dout3 = self.dlayer3(dout4_out3)
        dout3_out2 = torch.cat([dout3, out2], 1)
        dout2 = self.dlayer2(dout3_out2)
        dout2_out1 = torch.cat([dout2, out1], 1)
        dout1 = self.dlayer1(dout2_out1)
        return dout1

class discriminator(nn.Module):
    # initializers
    def __init__(self, d=64):
        super(discriminator, self).__init__()
        self.main = nn.Sequential(nn.Conv2d(6, d, 4, 2, 1, bias=False),
                                  nn.Conv2d(d, d*2, 4, 2, 1, bias=False),
                                  nn.BatchNorm2d(d*2),
                                  nn.Conv2d(d*2, d*4, 4, 2, 1, bias=False),
                                  nn.BatchNorm2d(d*4),
                                  nn.LeakyReLU(0.2, inplace=True),
                                  nn.Conv2d(d*4, d*8, 4, 1, 1, bias=False),
                                  nn.BatchNorm2d(d*8),
                                  nn.LeakyReLU(0.2, inplace=True),
                                  nn.Conv2d(d*8, 1, 4, 1, 1, bias=False),
                                  nn.Sigmoid())
        
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)
    # forward method
    def forward(self, input):
        output = self.main(input)
        return output

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()
def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().numpy()
    #print(image_numpy.size())
    image_numpy = np.transpose(image_numpy,(0,1,2))
    image_numpy = (image_numpy+1)/2.0*255.0
    return image_numpy.astype(imtype)






