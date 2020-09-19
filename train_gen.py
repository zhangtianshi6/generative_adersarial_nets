from torchvision import datasets, transforms
from torch.utils.data import Dataset
import torch.optim as optim
import torch
import torch.nn as nn
from GAN import Generator, Discriminator
import numpy as np
import cv2


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
batch_size = 16
test_batch_size = 64
input_size = 224

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


class GenertorData(Dataset):
    def __init__(self, real_lines, cartoon_lines, batch_size=16, input_size = 224):
        print(len(real_lines), len(cartoon_lines))
        self.real_lines = real_lines
        self.cartoon_lines = cartoon_lines
        self.real_img = np.array([cv2.resize(cv2.imread(i[:-1]), (input_size, input_size)) for i in real_lines]).transpose(0, 3, 1, 2)*1.0/255
        self.cartoon_img = np.array([cv2.resize(cv2.imread(i[:-1]), (input_size, input_size)) for i in cartoon_lines]).transpose(0, 3, 1, 2)*1.0/255
        
        self.real_img = torch.from_numpy(self.real_img).float()
        self.cartoon_img = torch.from_numpy(self.cartoon_img).float()
        self.batch_size = batch_size

    def __len__(self):
        return len(self.cartoon_img)
    
    def __getitem__(self, idx):
        idx = min(idx, len(self.cartoon_img)-self.batch_size)

        return self.real_img[idx:idx+self.batch_size], self.cartoon_img[idx:idx+self.batch_size]

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

    input_channels = 3
    lr = 0.01
    momentum = 0.5
    epochs = 200
    lambda_pixel = 300
    #gen = Gen(100)
    gen_model = Generator(input_channels, input_channels)
    disc_model = Discriminator(input_channels, 2)
    #optimizer_G = optim.Adam(gen_model .parameters(), lr=lr)
    #optimizer_D = optim.Adam(disc_model .parameters(), lr=lr)
    optimizer_G = optim.SGD(gen_model.parameters(), lr=lr, momentum=momentum)
    optimizer_D = optim.SGD(disc_model.parameters(), lr=lr, momentum=momentum)
    #piexl_loss = torch.nn.L1Loss()
    piexl_loss = nn.L1Loss()
    disc_loss = nn.CrossEntropyLoss()
    if use_cuda:
        gen_model = gen_model.cuda()
        disc_model = disc_model.cuda()
        piexl_loss = piexl_loss.cuda()
        disc_loss = disc_loss.cuda()
    # prepare fake_real label
    real_lines = open('real_face.txt', 'r').readlines()[:1000]
    cartoon_lines = open('cartoon_face.txt', 'r').readlines()[:1000]
    train_loader = GenertorData(real_lines, cartoon_lines, batch_size, input_size)
    epoch_g_loss = []
    epoch_d_loss = []
    fw_log = open('log.txt', 'w')
    for epoch in range(epochs):
        train_loss_G = 0
        train_loss_D = 0
        #for batch_idx, (data, target) in enumerate(train_loader):
        for batch_idx in range(len(train_loader)):
            data, target = train_loader[batch_idx]
            data, target = data.to(device), target.to(device)
            real_target, fake_target = generate_label(data.size(0))
            # train generators
            optimizer_G.zero_grad()
            fake = gen_model(data)
            real_pred = disc_model(target)
            fake_pred = disc_model(fake)
            disc_loss_real = disc_loss(real_pred, real_target)
            disc_loss_fake = disc_loss(fake_pred, fake_target)
            loss_D = disc_loss_real+disc_loss_fake
            loss_G = piexl_loss(target, fake)
            loss_G = loss_D+lambda_pixel*loss_G
            loss_G.backward()
            optimizer_G.step()
            train_loss_G += loss_G.item()

            # train Discriminator
            if (batch_idx/50)==epoch%(len(train_loader)/50):
            # if loss_D > 0.05:
                optimizer_D.zero_grad()
                fake = gen_model(data)
                #print(fake.size())
                real_pred = disc_model(target)
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
        torch.save(gen_model.state_dict(),"model/gen_cartoon_model_epoch_"+str(epoch)+'_gloss'+str(loss_G.item())[:6]+'_d_loss'+str(train_loss_D)[:6]+".pt")
        fw_log.write(str(epoch)+' '+str(epoch_g_loss)+'\n')
        fw_log.write(str(epoch)+' '+str(epoch_d_loss)+'\n')
        draw(epoch_g_loss, epoch_d_loss)
  
import cv2
def test():
    real_lines = open('real_face.txt', 'r').readlines()[1000:2000]
    cartoon_lines = open('cartoon_face.txt', 'r').readlines()[1000:2000]
    test_loader = GenertorData(real_lines, cartoon_lines, batch_size, input_size)
    model = Generator(3, 3)
    model.load_state_dict(torch.load("model/gen_cartoon_model_epoch_1_gloss68.747.pt"))
    torch.no_grad()
    img = np.zeros((224,224,3))
    # model = torch.load('model.pkl')
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = test_loader[batch_idx]
        fake = model(data)
        print(fake.shape)
        fake = fake.data.numpy()
        for k in range(fake.shape[0]):
            img[:, :, 0] = fake[k, 0, :, :]*255
            img[:, :, 1] = fake[k, 1, :, :]*255
            img[:, :, 2] = fake[k, 2, :, :]*255
            cv2.imwrite('img_'+str(k)+'_'+str(batch_idx)+'.jpg', img)
            
    
#train()
test()
