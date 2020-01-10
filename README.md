# Deblur-Pix2Pix

本项目为SJTU CS386数字图像处理大作业项目Deblur，采用了Pix2Pix神经网络对图像进行去模糊。可实现对任意大小的图片在不降低分辨率的前提下进行去糊处理。

## 运行所需条件：
cuda + cudnn + pytorch，具体版本无特殊要求

## 数据集说明：
本项目采用GoPro数据集进行训练和测试，数据集地址为https://drive.google.com/file/d/1H0PIXvJH4c40pk7ou6nAwoxuR4Qh_Sa2/view?usp=sharing，下载后需重新组织一下图片编排，可运行python reorganize_dataset.py进行重排，参数--dir_in为原始数据路径，--dir_out为重排后数据路径。

如果想使用你自己的数据集，请注意以下几点：
- 在该目录下新建train文件夹，train目录下，需有A和B两个子文件夹，其中A存放模糊图，B存放清晰图，A和B中对应图片命名需一致。
- 目前仅支持png文件，若数据集均为jpg文件，可在train_dataset.py文件第15行将 '\*.png'改为'\*.jpg'。


## 测试方法：
我们提供了两种测试方法，一种是没有ground truth的，可用来对自己的图片进行测试；另一种是有ground truth的，可计算psnr,ssim指标。默认为第一种。
- 将你想要去糊的图片放在test文件夹中，运行python test.py即可，结果会存放在results文件夹中。使用的已训练好的模型在：https://pan.baidu.com/s/1ma0Wdrqzrc_SwXgledXkUw 
提取码：6by3 
将其放在./checkpoints/experiment_name2中。
- 运行python test.py --haveGT 1 则为第二种测试方法。该测试方法是我们测试自己训练结果的时候用的。目录下需有一个test文件夹，文件格式与注意事项同训练集部分一样。测试时除了将原模糊图经过网络，还会计算psnr和ssim指标，最后将所有图片指标取平均。每100张测试图片会保存一次。

**注意**
目前暂时只支持png文件，若需要测试jpg文件，请在test_dataset.py文件中第21行的'\*.png'改为'\*.jpg'（第二种测试方法则改第14行）。另外，当前代码能够去糊的最大图片大小为1400*1400，若被测试图片大小超出这个范围，请在test.py文件中第83行的（1400,1400）改为需要的大小（第二种测试方法则改第41行）。


## 训练方法：
运行python train.py即可进行训练，
--dataroot可改变数据集路径，默认为./train
--epoch可改变开始的epoch数，除非为中断训练后重新开始，否则不建议修改
其余参见help

## 结果展示：
demo文件夹放了若干张模糊图与去糊后的结果图，采用的模型是1e-4+1e-5训练结果、padding方式为镜像填补。
