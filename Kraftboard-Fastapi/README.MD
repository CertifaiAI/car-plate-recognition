## FastAPI server for running python scripts

## Available python scripts
1. PaddlePaddle OCR
- Convert image to texts or numbers
- Input: Text image
- Output: OCR result

2. MobileFaceNet 
- Convert face image to embeddings
- Input: Face Image
- Output: Base64 Embeddings

## Install Pytorch and PaddlePaddle
There are two version of pytorch and paddlepaddle that can be installed. Please select a version that suitable to your machine

### Installation on linux (tested on Ubuntu 20.04)
```
# CPU
pip3 install torch==1.9.0+cpu torchvision==0.10.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
python -m pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple

# GPU (cuda 10.2)
pip3 install torch==1.9.0+cu102 torchvision==0.10.0+cu102 -f https://download.pytorch.org/whl/torch_stable.html
python -m pip install paddlepaddle-gpu -i https://mirror.baidu.com/pypi/simple
```

### Installation on Windows (tested on Windows 10)
```
# CPU
pip3 install torch torchvision
python -m pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple

# GPU (cuda 10.2)
pip3 install torch==1.9.0+cu102 torchvision==0.10.0+cu102 -f https://download.pytorch.org/whl/torch_stable.html
python -m pip install paddlepaddle-gpu -i https://mirror.baidu.com/pypi/simple
```

Notes:
1. Please modify the Settings.DEVICE variable to cuda (Using pytorch GPU) or cpu (Using pytorch cpu) under the file Core/config.py
2. Please refer to Pytorch [installation guide page](https://pytorch.org/get-started/locally/) for other pytorch gpu cuda version
3. Please refer to Paddle Paddle [installation guide page](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html
) for other PaddlePaddle gpu cuda version

## Install Requirments
Install the required dependencies using the following command
```
pip install -r requirements.txt
```

## Download Weights
Download the weights and add them into your working directory

### Command on linux (tested on Ubuntu 20.04)
```
./install_req.sh 
```

### Command on Windows (tested on Windows 10)
```
.\install_req.sh 
```

## Getting Started
```
python3 main.py
```
This will start the server on localhost:8000 (Can modify port on core/config.py)

## Docs
```
http://localhost:8000/docs
```
This will direct web page to fastapi built in docs

## Testing API
```
python3 tests/send_images.py
```
Testing images and api can be done using the codes in tests folder
