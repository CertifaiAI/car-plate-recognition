## Cargate Project Implementation on Jetson Nano
#### The system is built using yolov5 (objects detector) [Reference 1] and PaddlePaddle OCR (license plate recognition) [Reference 2]

### **Prerequisite**
#### 1. Install pytorch using the script provided
```
$ ./install_pytorch.sh
```

#### 2. Download required dependencies 
```
$ pip3 install -r requirements.txt
```

#

### **Run program**
```
$ python3 main.py
```

### **Demo**
![car and license plate detection](./result.jpg)
#

### **References**
#### 1. https://github.com/ultralytics/yolov5
#### 2. https://github.com/PaddlePaddle/PaddleOCR
