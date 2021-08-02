#!/bin/bash
# install pytorch 1.8
wget https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl -O torch-1.8.0-cp36-cp36m-linux_aarch64.whl
sudo apt-get install python3-pip libopenblas-base libopenmpi-dev 
pip3 install Cython
pip3 install numpy torch-1.8.0-cp36-cp36m-linux_aarch64.whl

# install torch vision 0.9.0
sudo apt-get install libjpeg-dev zlib1g-dev libpython3-dev libavcodec-dev libavformat-dev libswscale-dev
git clone --branch release/0.9 https://github.com/pytorch/vision torchvision 
cd torchvision
export BUILD_VERSION=0.9.0
python3 setup.py install --user
cd ../
pip3 install 'pillow<7'
