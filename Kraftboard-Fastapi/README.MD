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

## Installation 
```
./install_req.sh 
```
Run the install requirements scripts

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