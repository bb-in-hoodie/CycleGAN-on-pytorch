# CycleGAN on Pytorch

Implementation of CycleGAN on Pytorch. The model learns how to convert an image of domain A to an image of domain B and vice versa. On this project, CelebA dataset has been used as the main dataset. The model has learned how to translate a female image to a male image and vice versa.


## Prerequites
* [Python 3.6](https://www.continuum.io/downloads)
* [PyTorch 0.2.0](http://pytorch.org/) (PyTorch is currently available only on Linux and OSX)
* The code has been written on Linux (ubuntu 16.04 LTS) system
* CPU or CUDA-available-GPU


## How to
#### 1. Install Python and PyTorch (from the link above or on your own way)
#### 2. Prepare your own dataset, store it in the project folder (it should contain the images of two domains)
#### 3. Change the dataset location part of 'train.py' code like below
```python
...
# The lowest folder should have two sub-folders, each containing images from another domain
image_location = './data/your_dataset/train'
...
```
#### 4. Run the command below on Terminal
```bash
$ python train.py
```


## Results
### 1) Female to Male
![FtoM_01](result/examples/FtoM_01.png)
![FtoM_02](result/examples/FtoM_02.png)
![FtoM_03](result/examples/FtoM_03.png)
![FtoM_04](result/examples/FtoM_04.png)
![FtoM_05](result/examples/FtoM_05.png)
![FtoM_06](result/examples/FtoM_06.png)
![FtoM_07](result/examples/FtoM_07.png)
![FtoM_08](result/examples/FtoM_08.png)
![FtoM_09](result/examples/FtoM_09.png)
![FtoM_10](result/examples/FtoM_10.png)
### 2) Male to Female
![MtoF_01](result/examples/MtoF_01.png)
![MtoF_02](result/examples/MtoF_02.png)
![MtoF_03](result/examples/MtoF_03.png)
![MtoF_04](result/examples/MtoF_04.png)
![MtoF_05](result/examples/MtoF_05.png)
![MtoF_06](result/examples/MtoF_06.png)
![MtoF_07](result/examples/MtoF_07.png)
![MtoF_08](result/examples/MtoF_08.png)
![MtoF_09](result/examples/MtoF_09.png)
![MtoF_10](result/examples/MtoF_10.png)