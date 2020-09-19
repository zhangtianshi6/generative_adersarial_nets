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
        #print(x.size()) 
        x = F.relu(self.bn1(self.conv1(x))) 
        x = F.max_pool2d(x, 3, 2, 1) 
        x = F.relu(self.bn2(self.conv2(x))) 
        x = F.max_pool2d(x, 3, 2, 1) 
        x = F.relu(self.bn3(self.conv3(x))) 
        x = F.max_pool2d(x, 3, 2, 1) 
        #print(x.size())
        x = F.relu(self.bnup1(self.upsample1(x)))
        #print(x.size())
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
        #x = x.view(x, )
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

from torchvision import datasets, transforms
import torch.optim as optim
import torch
import numpy as np

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
batch_size = 64
test_batch_size = 64
input_size = 56

def load_data():

    pin_memory = True if use_cuda else False
    train_loader = torch.utils.data.DataLoader(datasets.MNIST('data', \
        train=True, download=True, transform=transforms.Compose([\
        transforms.RandomResizedCrop(input_size), transforms.ToTensor()])),\
        batch_size = batch_size, shuffle=True, num_workers=1, pin_memory=pin_memory)

    test_loader = torch.utils.data.DataLoader(datasets.MNIST('data', \
        train=False, transform=transforms.Compose([\
        transforms.RandomResizedCrop(input_size), transforms.ToTensor()])),\
        batch_size = test_batch_size, shuffle=True, num_workers=1, pin_memory=pin_memory)
    return train_loader, test_loader

def generate_label(batch_size):
    fake_target = np.zeros((batch_size))
    fake_target[:] = 0
    fake_target = torch.from_numpy(fake_target).long()
    real_target = np.zeros((batch_size))
    real_target[:] = 1
    real_target = torch.from_numpy(real_target).long()
    if use_cuda:
        real_target = real_target.to(device)
        fake_target = fake_target.to(device)
    return real_target, fake_target

def draw(all_g_loss, all_d_loss):
    import matplotlib.pyplot as plt

    all_g_loss = [i for i in all_g_loss]
    all_d_loss = [i for i in all_d_loss]
    epochs = range(len(all_g_loss))
    plt.figure()
    plt.plot(epochs, all_g_loss, 'r-', label="g_loss")
    plt.plot(epochs, all_d_loss, '-', label="d_loss")
    plt.title('loss')
    plt.legend(loc="lower right")
    plt.savefig('epoch.jpg')


def train():
    
    
    # load data
    # img_rows, img_cols = 28, 28
    cc
    input_channels = 1
    lr = 0.01
    momentum = 0.5
    epochs = 200
    lambda_pixel = 100
    #gen = Gen(100)
    gen_model = Generator(input_channels, input_channels)
    disc_model = Discriminator(input_channels, 2)
    optimizer_G = optim.Adam(gen_model .parameters(), lr=lr)
    optimizer_D = optim.Adam(disc_model .parameters(), lr=lr)
    #optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    #piexl_loss = torch.nn.L1Loss()
    piexl_loss = nn.L1Loss()
    disc_loss = nn.CrossEntropyLoss()
    if use_cuda:
        gen_model = gen_model.cuda()
        disc_model = disc_model.cuda()
        piexl_loss = piexl_loss.cuda()
        disc_loss = disc_loss.cuda()
    # prepare fake_real label
    

    epoch_g_loss = []
    epoch_d_loss = []
    fw_log = open('log.txt', 'w')
    for epoch in range(epochs):
        train_loss_G = 0
        train_loss_D = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            
            data, target = data.to(device), target.to(device)
            real_target, fake_target = generate_label(data.size(0))
            # train generators
            optimizer_G.zero_grad()
            fake = gen_model(data)
            real_pred = disc_model(data)
            fake_pred = disc_model(fake)
            disc_loss_real = disc_loss(real_pred, real_target)
            disc_loss_fake = disc_loss(fake_pred, fake_target)
            loss_D = disc_loss_real+disc_loss_fake
            loss_G = piexl_loss(data, fake)
            loss_G = loss_D+lambda_pixel*loss_G
            loss_G.backward()
            optimizer_G.step()
            train_loss_G += loss_G.item()

            # train Discriminator
            # if (batch_idx/50)%2==0:
            if loss_D > 0.0005:
                optimizer_D.zero_grad()
                fake = gen_model(data)
                #print(fake.size())
                real_pred = disc_model(data)
                fake_pred = disc_model(fake)
                disc_loss_real = disc_loss(real_pred, real_target)
                disc_loss_fake = disc_loss(fake_pred, fake_target)
                loss_D = disc_loss_real+disc_loss_fake
                loss_D.backward()
                optimizer_D.step()
                train_loss_D = loss_D.item()
            if batch_idx % 50 == 0:
                print("GAN train Epochs %d %d/%d G_loss %.6f D_loss %.6f"%(epoch, batch_idx, len(train_loader), loss_G.item(), train_loss_D))
            
        epoch_g_loss.append(loss_G.item())
        epoch_d_loss.append(train_loss_D)
        torch.save(gen_model.state_dict(),"model/gen_model_epoch_"+str(epoch)+'_gloss'+str(loss_G.item())[:6]+'_d_loss'+str(train_loss_D)[:6]+".pt")
        fw_log.write(str(epoch)+' '+str(epoch_g_loss)+'\n')
        fw_log.write(str(epoch)+' '+str(epoch_d_loss)+'\n')
        draw(epoch_g_loss, epoch_d_loss)
  
import cv2
def test():
    train_loader, test_loader = load_data()
    model = Generator(1, 1)
    model.load_state_dict(torch.load("model/gen_model.pt"))
    torch.no_grad()
    img = np.zeros((56,56,3))
    # model = torch.load('model.pkl')
    for batch_idx, (data, target) in enumerate(test_loader):
        fake = model(data)
        print(fake.shape)
        fake = fake.data.numpy()
        for k in range(fake.shape[0]):
            img[:, :, 0] = fake[k, 0, :, :]*256
            img[:, :, 1] = fake[k, 0, :, :]*256
            img[:, :, 2] = fake[k, 0, :, :]*256
            cv2.imwrite('img_'+str(k)+'_'+str(batch_idx)+'.jpg', img)
            
    
# train()
test()
