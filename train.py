import os
import pickle
import time
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
from multiprocessing import freeze_support
from train_dataset import *
from train_model import *
from train_option import *
#from visdom import Visdom
import matplotlib.pyplot as plt
# training parameters

lr = 0.00001
train_epoch = 100

train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
train_hist['per_epoch_ptimes'] = []
#vis = Visdom()

def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().numpy()
    #print(image_numpy.size())
    image_numpy = np.transpose(image_numpy,(0,1,2))
    image_numpy = (image_numpy+1)/2.0*255.0
    return image_numpy.astype(imtype)

start_time = time.time()
def train(train_loader,start_epoch):
    trainLogger = open('%s/train_2.log' % opt.checkpoints_dir, 'a')
    for epoch in range(start_epoch,train_epoch):
        epoch_start_time = time.time()
        D_losses = []
        G_losses = []
        for blur_,sharp_ in train_loader:
            #blur = blur_.view(-1,3,256,256)
            #sharp = sharp_.view(-1,3,256,256)
            
            real_sharp.data.resize_as_(sharp_).copy_(sharp_)
            real_blur.data.resize_as_(blur_).copy_(blur_)
            
            with torch.no_grad():
            	label_one.resize_((1, 1, 30, 30)).fill_(0.9)
            	label_zero.resize_((1,1,30,30)).fill_(0)
            #train discriminator D
            for p in D.parameters(): 
                p.requires_grad = True
            for i in range(1):
                D.zero_grad()
                D_result = D(torch.cat([real_sharp, real_blur], 1))
                D_real_loss = BCE_loss(D_result,label_one)
                
                G_fake = G(real_blur)
                G_result = G_fake.detach()
                D_result = D(torch.cat([G_result, real_blur], 1))
                D_fake_loss = BCE_loss(D_result,label_zero)
                
                D_train_loss = D_real_loss + D_fake_loss
                D_train_loss.backward()
                D_optimizer.step()
                #D_scheduler.step()
                D_losses.append(D_train_loss.item())
            
            # train generator G
            
            # prevent computing gradients of weights in Discriminator
            for p in D.parameters(): 
                p.requires_grad = False
            for i in range(1):
                G.zero_grad()
                G_l1_loss = L1_loss(G_fake,real_sharp)
                G_l1_loss = G_l1_loss * 0.1
               
                D_result = D(torch.cat([G_result, real_blur], 1))
                G_gan_loss = BCE_loss(D_result, label_one)
                G_train_loss = G_l1_loss + G_gan_loss
                
                G_train_loss.backward()
                G_optimizer.step()
                #G_scheduler.step()
                G_losses.append(G_train_loss.item())
            
 #           fake_sharp.data.resize_as_(blur_).copy_(blur_)
 #           fake_ = G(fake_sharp)
            #fake = fake.view(1,3,128,128)
#            blur = tensor2im(blur_.data)
#            sharp = tensor2im(sharp_.data)
#            fake = tensor2im(G_result.data)
#            vis.image(blur,opts=dict(title='Blur_images'),win='x')
#            vis.image(fake,opts=dict(title='Restored_images'),win='y')
#            vis.image(sharp,opts = dict(title='Sharp_images'),win='z')
#            print('[%d/%d]: loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), train_epoch,D_train_loss.item(), G_train_loss.item()))
#           
                 
        epoch_finish_time = time.time()
        per_epoch_time = epoch_finish_time - epoch_start_time
        print('[%d/%d]: loss_d: %.3f, loss_g: %.3f, time:[%fs/%fs]' % ((epoch + 1), train_epoch,D_train_loss.item(), G_train_loss.item(),per_epoch_time,epoch_finish_time - start_time))
        trainLogger.write('[%d/%d]: loss_d: %.3f, loss_g: %.3f, time:[%fs/%fs]\n' % ((epoch + 1), train_epoch,D_train_loss.item(), G_train_loss.item(),per_epoch_time,epoch_finish_time - start_time))
        trainLogger.flush()
        train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
        train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))
        train_hist['per_epoch_ptimes'].append(per_epoch_time)
        torch.save(G.state_dict(), "checkpoints/experiment_name2/generator_param%04d.pkl"%(epoch+1))
        torch.save(D.state_dict(), "checkpoints/experiment_name2/discriminator_param.pkl")
    trainLogger.close()
if __name__=='__main__':
    freeze_support()
    opt = TrainOptions().parse()
     #data loader
    data = MyTrainData(opt,True)
    train_loader = torch.utils.data.DataLoader(data)
    torch.cuda.empty_cache()
    
    real_sharp = torch.Tensor(1,3,256,256)
    real_blur = torch.Tensor(1,3,256,256)
    fake_sharp = torch.Tensor(1,3,256,256)
    label_one = torch.Tensor(1)
    label_zero = torch.Tensor(1)
    
    real_sharp = Variable(real_sharp.cuda())
    real_blur = Variable(real_blur.cuda())
    fake_sharp = Variable(fake_sharp.cuda())
    label_one = Variable(label_one.cuda())
    label_zero = Variable(label_zero.cuda())
    
    BCE_loss = nn.BCELoss()
    L1_loss = nn.L1Loss()
    BCE_loss.cuda()
    L1_loss.cuda()

    # network
    G = generator(64)
    D = discriminator(64)
    G.weight_init(mean=0.0,std=0.02)
    D.weight_init(mean=0.0,std=0.02)
    if opt.epoch != 0:
        G.load_state_dict(torch.load('./checkpoints/experiment_name2/generator_param%04d.pkl'%(opt.epoch)))
        D.load_state_dict(torch.load('./checkpoints/experiment_name2/discriminator_param.pkl'))
    G.cuda()
    D.cuda()
    
    #lr = lr * (0.9**(opt.epoch//5))
    # Adam optimizer
    G_optimizer = optim.Adam(G.parameters(), lr=lr,betas=(0.5,0.999))
    D_optimizer = optim.Adam(D.parameters(), lr=lr,betas=(0.5,0.999))
    #G_scheduler = optim.lr_scheduler.StepLR(G_optimizer,step_size=5,gamma = 0.9)
    #D_scheduler = optim.lr_scheduler.StepLR(D_optimizer,step_size=5,gamma = 0.9)
    # results save folder
    if not os.path.isdir(opt.checkpoints_dir):
        os.makedirs(opt.checkpoints_dir)
    if not os.path.isdir('%s/experiment_name2'%(opt.checkpoints_dir)):
        os.makedirs('%s/experiment_name2'%(opt.checkpoints_dir))
    print("Prepare to train")
    train(train_loader,opt.epoch)
    print("Training finish!... save training results")
    torch.save(G.state_dict(), "%s/experiment_name2/generator_param.pkl"%(opt.checkpoints_dir))
    torch.save(D.state_dict(), "%s/experiment_name2/discriminator_param.pkl"%(opt.checkpoints_dir))
    with open('%s/experiment_name2/train_hist.pkl'%(opt.checkpoints_dir), 'wb') as f:
        pickle.dump(train_hist, f)

    
    
