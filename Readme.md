# Twin-Adversarial-Contrastive-Learning-for-Underwater-Image-Enhancement-and-Beyond
This is an implement of the TACL,
**“[Twin-Adversarial-Contrastive-Learning-for-Underwater-Image-Enhancement-and-Beyond](https://ieeexplore.ieee.org/document/9832540)”**, 
Risheng Liu*, Zhiying Jiang, Shuzhou Yang, Xin Fan, IEEE Transactions on Image Processing __(TIP)__, 2022.

## Overview
![avatar](Overview.PNG)

## Prerequisites
- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## 🔑 Installation
Type the command:
```
pip install -r requirements.txt
```

## 🤖 Download
Download the pre-trained model and put it in _./checkpoints_
- [Google Drive](https://drive.google.com/file/d/1VEx7CR0iFJNesCS_Ci98CEAaFhhzsela/view?usp=sharing)
- [Baidu Yun](https://pan.baidu.com/s/1WZ79-GbJEoJkNDMrgVBtVw) \
code:
​```
YSZD
​```

## 🚀 Quick Run
- Create directories `./dataset/testA` and `./dataset/testB`. Put your test images in `./dataset/testA` (And you should keep whatever one image in `./dataset/testB` to make sure program can start.)
- To test the pre-trained models for Underwater Enhancement on your own images, run
```bash
python test.py --dataroot ./datasets/[YOUR-DATASETS] --name underwater --model cycle_gan
```
Results will be shown in _results_ folder.

## Train Backbone
- First, you need to train a base backbone:
```bash
python train.py --dataroot ./datasets/[YOUR-DATASETS] --name chinamm_train --model cycle_gan
```

## Training TAF
- Second, you need to train a TAF module (here we adopt SSD): 
  * Download an Underwater Detection Dataset (**[Chinamm](https://rwenqi.github.io/chinaMM2019uw/)**).
  * Run this to make Chinamm in VOC format:
  ```bash
  python makeTXT.py
  ```
  * Use the trained backbone to enhance JPEGImages of chinamm.
  * cd ./ssd.pytorch-master

- Download the fc-reduced [VGG-16](https://arxiv.org/abs/1409.1556) PyTorch base network weights at:              https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth
- By default, we assume you have downloaded the file in the `ssd.pytorch/weights` dir:

```Shell
mkdir weights
cd weights
wget https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth
```

- To train SSD using the train script simply specify the parameters listed in `train.py` as a flag or manually change them.

```Shell
python train.py
```

- Note:
  * For training, an NVIDIA GPU is strongly recommended for speed.
  * For instructions on Visdom usage/installation, see the <a href='#installation'>Installation</a> section.
  * You can pick-up training from a checkpoint by specifying the path as one of the training parameters (again, see `train.py` for options)

- Evaluation
To evaluate a trained network:

```Shell
python eval.py
```

You can specify the parameters listed in the `eval.py` file by flagging them or manually changing them.  

## Training
cd ./ssd.pytorch-master \
Run
```Shell
python trainall.py
```
- Test final version:
```Shell
python visual.py
```


## 📌 Citation

If you find this code useful for your research, please use the following BibTeX entry.

```bibtex
@ARTICLE{9832540,
  author={Liu, Risheng and Jiang, Zhiying and Yang, Shuzhou and Fan, Xin},
  journal={IEEE Transactions on Image Processing}, 
  title={Twin Adversarial Contrastive Learning for Underwater Image Enhancement and Beyond}, 
  year={2022},
  volume={31},
  number={},
  pages={4922-4936},
  doi={10.1109/TIP.2022.3190209}}
```
