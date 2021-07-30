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

## Install Pytorch
There are two version of pytorch can be installed. Please select a version that suitable to your machine
```
# CPU
pip3 install torch==1.9.0+cpu torchvision==0.10.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

# GPU
pip3 install torch==1.9.0+cu102 torchvision==0.10.0+cu102 -f https://download.pytorch.org/whl/torch_stable.html
```

## Installation on linux (tested on Ubuntu 20.04)
```
pip install -r requirements.txt
./install_req.sh 
```

## Installation on Windows (tested on Windows 10)
```
pip install -r requirements.txt
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