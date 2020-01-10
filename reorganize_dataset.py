import torch.utils.data as data
import os
import os.path
import argparse
#from util import util
import torch
from shutil import copyfile
import glob

class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False
    
    def initialize(self):
        self.parser.add_argument('--dir_in', type=str, default="GOPRO_Large")
        self.parser.add_argument('--dir_out',type=str,default="./")
        
        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        args = vars(self.opt)
        return self.opt

if __name__ == "__main__":
    opt = Options().parse()
    if not os.path.isdir(opt.dir_out):
        os.makedirs(opt.dir_out)
    for fGT in glob.glob(os.path.join(opt.dir_in, '*')):
        fName = os.path.basename(fGT)
        output_directory = os.path.join(opt.dir_out, fName)
        output_directory_A = os.path.join(output_directory, 'A')
        output_directory_B = os.path.join(output_directory, 'B')
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
        if not os.path.isdir(output_directory_A):
            os.makedirs(output_directory_A)
        if not os.path.isdir(output_directory_B):
            os.makedirs(output_directory_B)
            
        #cur_path=os.path.join(opt.dir_in,fGT)
        cur_path = fGT
        print(cur_path)
        for cur_subpath in glob.glob(os.path.join(cur_path,'*')):
            print(cur_subpath)
            image_folder = os.path.basename(cur_subpath)
            blur_folder = os.path.join(cur_subpath,'blur')
            for blur in glob.glob(os.path.join(blur_folder, '*.png')):
                blur = os.path.basename(blur)
                current_blur_path = os.path.join(blur_folder, blur)
                output_blur_path = os.path.join(output_directory_A, image_folder + "_" + blur)
                #print(current_blur_path)
                #print(output_blur_path)
                copyfile(current_blur_path, output_blur_path)
            
            sharp_folder = os.path.join(cur_subpath,'sharp')
            for sharp in glob.glob(os.path.join(sharp_folder, '*.png')):
                sharp = os.path.basename(sharp)
                current_sharp_path = os.path.join(sharp_folder,sharp)
                output_sharp_path = os.path.join(output_directory_B, image_folder + "_" + sharp)
                copyfile(current_sharp_path, output_sharp_path)