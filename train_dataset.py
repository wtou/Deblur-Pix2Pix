#encoding:utf-8
import torch.utils.data as data
import os
import os.path
import glob
from PIL import Image
from torchvision import transforms
import random
 
def make_dataset(root, train=True): #读取自己的数据的函数
    dataset = []
    dir_blur = os.path.join(root, 'A') 
    dir_sharp = os.path.join(root, 'B')
 
    for fGT in glob.glob(os.path.join(dir_sharp, '*.png')):
        fName = os.path.basename(fGT)    
        dataset.append( [os.path.join(dir_blur, fName), os.path.join(dir_sharp, fName)] )
    return dataset

def get_transform(randx,randy):
    transform_list = []
    transform_list.append(transforms.Lambda(lambda img: _random_crop(img,randx,randy)))
    transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def _random_crop(img,randx,randy):
    ow,oh = img.size
    return img.crop((randx-128,randy-128,randx+128,randy+128))

class MyTrainData(data.Dataset):   #需要继承data.Dataset
    def __init__(self, opt,train = True): #初始化文件路進或文件名
        self.train = train
        self.train_set_path = make_dataset(opt.dataroot, train)
    
    def __getitem__(self, idx):
        path_blur,path_sharp = self.train_set_path[idx]
 
        img_blur = Image.open(path_blur).convert('RGB')
        img_sharp = Image.open(path_sharp).convert('RGB')
            
        x,y = img_blur.size
        while x<256 or y < 256:
            img_blur = img_blur.resize((256,int(256*y/x)))
            x,y = img_blur.size
                
        randx = random.randint(128,x-128)
        randy = random.randint(128,y-128)
        self.transform = get_transform(randx,randy);
        img_blur = self.transform(img_blur)
        img_sharp = self.transform(img_sharp)
        return img_blur,img_sharp  
       
 
    def __len__(self):
        return len(self.train_set_path)