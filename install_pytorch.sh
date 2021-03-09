#!/bin/bash

# This script will install pytorch, torchvision, torchtext and spacy on nano. 
# If you have any of these installed already on your machine, you can skip those.

sudo apt-get -y update
sudo apt-get -y upgrade
#Dependencies
sudo apt-get install python3-setuptools

#Installing PyTorch
#For latest PyTorch refer original Nvidia Jetson Nano thread - https://devtalk.nvidia.com/default/topic/1049071/jetson-nano/pytorch-for-jetson-nano/.
#Choose the version for pytorch and torchvision by changing version
wget https://nvidia.box.com/shared/static/9eptse6jyly1ggt9axbja2yrmj6pbarc.whl -O torch-1.6.0-cp36-cp36m-linux_aarch64.whl
sudo apt-get install python3-pip libopenblas-base
sudo pip3 install Cython
sudo pip3 install torch-1.6.0-cp36-cp36m-linux_aarch64.whl

#Installing torchvision
#For latest torchvision refer original Nvidia Jetson Nano thread - https://devtalk.nvidia.com/default/topic/1049071/jetson-nano/pytorch-for-jetson-nano/.
sudo apt-get install libjpeg-dev zlib1g-dev
git clone --branch release/0.7 https://github.com/pytorch/vision torchvision   # see below for version of torchvision to download
cd torchvision
sudo python3 setup.py install