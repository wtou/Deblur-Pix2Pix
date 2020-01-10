#encoding:utf-8
import torch.utils.data as data
import os
import os.path
import glob
from PIL import Image
from torchvision import transforms
 
def make_dataset_AB(root): #读取自己的数据的函数
    dataset = []
    dir_blur = os.path.join(root, 'A') 
    dir_sharp = os.path.join(root, 'B')
 
    for fGT in glob.glob(os.path.join(dir_sharp, '*.png')):
        fName = os.path.basename(fGT)    
        dataset.append( [os.path.join(dir_blur, fName), os.path.join(dir_sharp, fName)] )
    return dataset

def make_dataset(root):
    dataset = []
    for fGT in glob.glob(os.path.join(root, '*.png')):
        dataset.append(fGT)
    return dataset

def get_transform(opt):
    transform_list = []
    transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

class MyTestData(data.Dataset):   #需要继承data.Dataset
    def __init__(self, opt): #初始化文件路進或文件名
        if opt.hasGT == 1:
            self.test_set_path = make_dataset_AB(opt.dataroot)
        else:
            self.test_set_path = make_dataset(opt.dataroot)
        self.opt = opt
        self.transform=get_transform(opt)
    
    def __getitem__(self, idx):
        if self.opt.hasGT == 1:
            path_blur,path_sharp = self.test_set_path[idx]
        else:
            path_blur = self.test_set_path[idx]
        img = Image.open(path_blur)
        w=img.size[0] #宽
        h=img.size[1] #高
        m=(w-1)//256+1
        n=(h-1)//256+1
        '''for i in range(len(path_blur)):
            if(path_blur[len(path_blur)-i-1]=='\\'):
                start=len(path_blur)-i-1
                break
        name=[]
        for i in range(len(path_blur)-start-1):
            name+=[path_blur[start+i+1]]
        name=''.join(name)'''
        
        name = os.path.basename(path_blur)
        
        imgs=[]
        imgs+=[w,h,m,n,name]
        for i in range(m):
            for j in range(n):
                if(i<m-1 and j<n-1):
                    cropimg=img.crop([i*256,j*256,(i+1)*256,(j+1)*256])
                    smallimg = self.transform(cropimg)
                    imgs+=[smallimg]
                elif(i==m-1 and j<n-1):
                    cropimg=img.crop([i*256,j*256,w,(j+1)*256])
                    newimg=Image.new('RGB',(256,256),(128,128,128))
                    newimg_w=w%256
                    if(newimg_w==0):
                        newimg.paste(cropimg,(0,0))
                    else:
                        r=255//newimg_w+1
                        for k in range(r):
                            if(k%2==0):
                                pasteimg=cropimg
                                newimg.paste(pasteimg,(k*newimg_w,0))
                            else:
                                pasteimg=cropimg.transpose(Image.FLIP_LEFT_RIGHT)
                                newimg.paste(pasteimg,(k*newimg_w,0))
                            
                    #newimg.paste(cropimg,(0,0))
                    smallimg = self.transform(newimg)
                    #imgs.append(newimg)
                    imgs+=[smallimg]
                elif(i<m-1 and j==n-1):
                    cropimg=img.crop([i*256,j*256,(i+1)*256,h])
                    newimg=Image.new('RGB',(256,256),(128,128,128))
                    newimg_h=h%256
                    if(newimg_h==0):
                        newimg.paste(cropimg,(0,0))
                    else:
                        r=255//newimg_h+1
                        for k in range(r):
                            if(k%2==0):
                                pasteimg=cropimg
                                newimg.paste(pasteimg,(0,k*newimg_h))
                            else:
                                pasteimg=cropimg.transpose(Image.FLIP_TOP_BOTTOM)
                                newimg.paste(pasteimg,(0,k*newimg_h))
                    #newimg.paste(cropimg,(0,0))
                    smallimg = self.transform(newimg)
                    #imgs.append(newimg)
                    imgs+=[smallimg]
                else:
                    cropimg=img.crop([i*256,j*256,w,h])
                    newimg=Image.new('RGB',(256,256),(128,128,128))
                    #newimg.paste(cropimg,(0,0))
                    newimg_w=w%256
                    newimg_h=h%256
                    if(newimg_w==0 and newimg_h==0):
                        newimg.paste(cropimg,(0,0))
                    elif(newimg_w==0 and newimg_h!=0):
                        r=255//newimg_h+1
                        for k in range(r):
                            if(k%2==0):
                                pasteimg=cropimg
                                newimg.paste(pasteimg,(0,k*newimg_h))
                            else:
                                pasteimg=cropimg.transpose(Image.FLIP_TOP_BOTTOM)
                                newimg.paste(pasteimg,(0,k*newimg_h))
                    elif(newimg_w!=0 and newimg_h==0):
                        r=255//newimg_w+1
                        for k in range(r):
                            if(k%2==0):
                                pasteimg=cropimg
                                newimg.paste(pasteimg,(k*newimg_w,0))
                            else:
                                pasteimg=cropimg.transpose(Image.FLIP_LEFT_RIGHT)
                                newimg.paste(pasteimg,(k*newimg_w,0))
                    else:
                        r1=255//newimg_w+1
                        r2=255//newimg_h+1
                        for k in range(r1):
                            for t in range(r2):
                                if(k%2==0):
                                    pasteimg=cropimg
                                else:
                                    pasteimg=cropimg.transpose(Image.FLIP_LEFT_RIGHT)
                                if(t%2==1):
                                    pasteimg=pasteimg.transpose(Image.FLIP_TOP_BOTTOM)
                                newimg.paste(pasteimg,(k*newimg_w,t*newimg_h))
                        #newimg.paste(cropimg,(0,0))
                    smallimg = self.transform(newimg)
                    #imgs.append(newimg)
                    imgs+=[smallimg]

        if self.opt.hasGT == 1:
            imgs_sharp = Image.open(path_sharp)
            imgs_sharp = self.transform(imgs_sharp)
            return imgs,imgs_sharp
        else:
            return imgs
 
    def __len__(self):
        return len(self.test_set_path)
