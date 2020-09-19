import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 3, 1, 1) 
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(256)
        self.upsample1 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.bnup1 = nn.BatchNorm2d(128)
        self.upsample2 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.bnup2 = nn.BatchNorm2d(64)
        self.upsample3 = nn.ConvTranspose2d(64, out_channels, 4, 2, 1)

    def forward(self, x): 
        x = F.relu(self.bn1(self.conv1(x))) 
        x = F.max_pool2d(x, 3, 2, 1) 
        x = F.relu(self.bn2(self.conv2(x))) 
        x = F.max_pool2d(x, 3, 2, 1) 
        x = F.relu(self.bn3(self.conv3(x))) 
        x = F.max_pool2d(x, 3, 2, 1) 
        x = F.relu(self.bnup1(self.upsample1(x)))
        x = F.relu(self.bnup2(self.upsample2(x)))
        x = self.upsample3(x)
        return x

class Gen(nn.Module):
    def __init__(self, in_channels):
        super(Gen, self).__init__()
        self.nch = 200
        self.fc1 = nn.linear(in_channels, self.nch*14*14)
        self.bn1 = nn.BatchNorm2d(in_channels*14*14)
        self.upsample1 = nn.UpSample(2)
        self.conv1 = nn.Conv2d(self.nch, self.nch/2, 3, 1)
        self.bn2 = nn.BatchNorm2d(self.nch/2)
        self.conv2 = nn.Conv2d(self.nch/2, self.nch/4, 3, 1)
        self.bn3 = nn.BatchNorm2d(self.nch/2)
        self.conv3 = nn.Conv2d(self.nch/4, 1, 1)

    def forward(self, x): # x shape 100
        x = F.relu(self.bn1(self.fc1(x)))
        x = x.view(-1, self.nch, 14, 14)
        x = self.upsample1(x)
        x = F.relu(self.bn2(self.conv1(x)))
        x = F.relu(self.bn3(self.conv2(x)))
        x = F.sigmoid(self.conv3(x))
        return x
        
        

class Discriminator(nn.Module):
    def __init__(self, in_channels, class_num):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 5, 1) 
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 5, 1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, 1)
        self.bn3 = nn.BatchNorm2d(256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, class_num)

    def forward(self, x):  
        x = F.relu(self.bn1(self.conv1(x))) 
        x = F.max_pool2d(x, 3, 2) 
        x = F.relu(self.bn2(self.conv2(x))) 
        x = F.max_pool2d(x, 3, 2) 
        x = F.relu(self.bn3(self.conv3(x))) 
        x = F.relu(F.avg_pool2d(x, x.size()[3]))
        x = x.view(-1, x.size()[1]*x.size()[2]*x.size()[3])
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


