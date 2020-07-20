
This project is to implement “Attention-Adaptive and Deformable Convolutional Modules for Dynamic Scene Deblurring(with ERCNN)” .



 To run this project you need to setup the environment, download the dataset,
 and then you can train and test the network models. 

## Prerequiste
The project is tested on Ubuntu 16.04, GPU Titan XP. Note that one GPU is required to run the code.
Otherwise, you have to modify code a little bit for using CPU. If  using CPU for training, it may too slow.
So I recommend you using GPU strong enough and about 12G RAM.

## Dependencies

Python 3.5 or 3.6 are recommended.
```
tqdm==4.19.9
numpy==1.17.3
torch==1.0.0
Pillow==6.1.0
torchvision==0.2.2
```

## Environment

I recommend using ```virtualenv``` for making an environment. 


## Dataset

I use GOPRO dataset for training and testing. __Download links__:
 [GOPRO_Large](https://drive.google.com/file/d/1H0PIXvJH4c40pk7ou6nAwoxuR4Qh_Sa2/view?usp=sharing)

| Statistics  | Training | Test | Total |
| ----------- | -------- | ---- | ----- |
| sequences   | 22       | 11   | 33    |
| image pairs | 2103     | 1111 | 3214  |

After downloading dataset successfully, you need to put images in right folders. By default, you should have images on dataset/train and dataset/valid folders.

## Demo

## Training

Run the following command

```
python demo_train.py ('data_dir' is needed before running )

```

I used ADAM optimizer with a mini-batch size 16 for training. The learning rate is 1e-4. Total training takes 1000 epochs to converge.  To prevent our network from overfitting, several data augmentation techniques are involved. In terms of geometric transformations, patches are randomly rotated by 90, 180, and 270 degrees. To take image degradations into account, saturation in HSV colorspace is multiplied by a random number within [0.8, 1.2].   


## Testing

Run the following command

```
python demo_test.py ('data_dir' is needed before running )
```
## pretrained models
if you need the pretrained models,please contact us by chenleinj@njust.edu.cn

## Acknowledge

Our code is based on Deep Multi-scale Convolutional Neural Network for Dynamic Scene Deblurring [MSCNN](https://github.com/mingyuliutw/UNIT), which is a nice work for dynamic scene deblurring .
