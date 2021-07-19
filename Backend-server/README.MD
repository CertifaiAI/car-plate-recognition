## FastApi Server
This FastAPI server is used to process PPOCR in PC. You can send cropped car plate images to server for OCR result to return.

## Install other libraries
```
pip3 install -r requirements
```

## Weights for PPOCR
Please download the weights from [here](https://drive.google.com/drive/folders/1IIJOGPCsX21uGJEoyqGvSQjv3jxedgKn?usp=sharing) and put both folder into the Backend-server/ppocr/weights folder.


## Starting the server (in this directory)
```
- uvicorn --host 0.0.0.0 main:app --reload
```
This command allow jetson nano to access the server via same network (wifi)

## Send image to server (Please refer to send_image.py)
You can send cropped car plate image to the server (from your pc for testing) using send_image scipt. 
```
response = requests.post('http://localhost:8000/plate', data=json.dumps(data))
```

## Common problems
1. Could not find module 'goes_c.dll'
- Download the goes_c.dll from [here](https://www.dll-files.com/geos_c.dll.html)
- put the dll file into your python lib