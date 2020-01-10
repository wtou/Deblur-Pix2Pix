import imageio
import torch
from PIL import Image
from torch.autograd import Variable
from multiprocessing import freeze_support
from test_dataset import *
from test_model import *
from test_option import *
#from visdom import Visdom

from skimage.measure import compare_ssim
#from skimage.measure import compare_psnr
import numpy as np
import math
import time
train_epoch = 100


def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = np.transpose(image_numpy,(1,2,0))
    image_numpy = (image_numpy+1)/2.0*255.0
    return image_numpy.astype(imtype)

def psnr(img1,img2):
    mse=np.mean((img1/255.-img2/255.)**2)
    if(mse<1.0e-10):
        return 100
    PIXEL_MAX=1
    return 20*math.log10(PIXEL_MAX/math.sqrt(mse))

def test(test_loader,start):
    if not os.path.isdir('imgs3'):
        os.makedirs('imgs3')
    Logger = open('%s/evaluation3.log' % opt.checkpoints_dir, 'a')
    for epoch in range(start,train_epoch):
        PSNRave=0
        SSIMave=0
        cnt=0
        G.load_state_dict(torch.load('./checkpoints/experiment_name2/generator_param%04d.pkl'%(epoch)))
        new_fake = Image.new('RGB',(1400,1400),(128,128,128))
        for imgs,sharp in test_loader:
            cnt += 1
            #new_fake = Image.new('RGB',(1400,1400),(128,128,128))
            fake_imgs=[]
            for i in range(imgs[2].item()):
                for j in range(imgs[3].item()):
                    with torch.no_grad():
                        real_blur.copy_(imgs[5+j+i*imgs[3].item()][0])
                    G_fake = G(real_blur)
                    G_fake=tensor2im(G_fake.data)
                    G_fake=Image.fromarray(G_fake)
                    fake_imgs.append(G_fake)
            
            for i in range(imgs[2].item()):
                for j in range(imgs[3].item()):
                    new_fake.paste(fake_imgs[j+i*imgs[3].item()],(i*256,j*256))
            result=new_fake.crop([0,0,imgs[0].item(),imgs[1].item()])
            
            
            sharp = tensor2im(sharp.data)
            result = np.array(result)
            deblur_PSNR = psnr(result,sharp)
            deblur_SSIM = compare_ssim(result,sharp, data_range=255,multichannel=True)
            
            PSNRave += deblur_PSNR
            SSIMave += deblur_SSIM
            
            if (epoch+1)%10==0 and cnt%100 == 0:
                if not os.path.isdir('imgs3/Epoch%02d'%(epoch)):
                    os.makedirs('imgs3/Epoch%02d'%(epoch))
                savepath='imgs3/Epoch%02d/%s'%(epoch,imgs[4][0])
                imageio.imwrite(savepath,result)
                #print('Current Epoch: %02d, Current images:%s, PSNR = %f, SSIM = %f' % (epoch,imgs[4][0],deblur_PSNR,deblur_SSIM))
        PSNRave /= cnt
        SSIMave /= cnt
        Logger.write('[%d/%d] PSNR = %f, SSIM = %f\n' % (epoch+1,train_epoch,PSNRave,SSIMave))
        Logger.flush()
    Logger.close()

def test_only_blur(test_loader):
    G.load_state_dict(torch.load('./checkpoints/experiment_name2/generator_param0099.pkl'))
    new_fake = Image.new('RGB',(1400,1400),(128,128,128))
    if not os.path.isdir('results'):
        os.makedirs('results');
    for imgs in test_loader:
        fake_imgs=[]
        for i in range(imgs[2].item()):
            for j in range(imgs[3].item()):
                with torch.no_grad():
                    real_blur.copy_(imgs[5+j+i*imgs[3].item()][0])
                G_fake = G(real_blur)
                G_fake=tensor2im(G_fake.data)
                G_fake=Image.fromarray(G_fake)
                fake_imgs.append(G_fake)
            
        for i in range(imgs[2].item()):
            for j in range(imgs[3].item()):
                new_fake.paste(fake_imgs[j+i*imgs[3].item()],(i*256,j*256))
        result=new_fake.crop([0,0,imgs[0].item(),imgs[1].item()])
        
        result = np.array(result)
        savepath=('results/%s'%(imgs[4][0]))
        imageio.imwrite(savepath,result)
        print('Current images:%s' % (imgs[4][0]))
    
            
if __name__=='__main__':
    freeze_support()
    opt = TestOptions().parse()
     #data loader
    data = MyTestData(opt)
    test_loader = torch.utils.data.DataLoader(data)
    torch.cuda.empty_cache()

    real_blur = torch.Tensor(1,3,256,256)
    real_blur = Variable(real_blur).cuda()
    
    # network
    G = generator(64)
    G.cuda()
    
   
    print("Prepare to test")
    if opt.hasGT == 1:#needs both blur images and sharp images so as to calculate PSNR and SSIM
        test(test_loader,opt.epoch)
    else:#no ground truth (just test for fun...
        test_only_blur(test_loader)
    print("Test finish!... save testing results")

  
