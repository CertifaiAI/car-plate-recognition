## Automatic License Plate Recognition Implementation on Jetson Nano
#### The system is built using yolov4-tiny (car and license plate detector) and crnn (license plate recognition) [Reference 2]

### **Prerequisite**
#### 1. Requires TensorRT 6.x+
#### Use the command below to check tensorrt version
```
$ ls /usr/lib/aarch64-linux-gnu/libnvinfer.so*
```

#### 2. Install pycuda using the script provided
```
$ ./install_pycuda.sh
```

#### 3. Install pytorch using the script provided
```
$ ./install_pytorch.sh
```

#### 4. Build plugin 
```
$ cd plugins
$ make
```

#### 5. Download required dependencies 
```
$ pip3 install -r requirements.txt
```

#

### **Run image detection**
```
$ python3 main.py --image {image_path}
```
```
$ python3 main.py --video {video_path} --save (to save video)
```

#

### **Demo**
![car and license plate detection](./result.jpg)
#

### **References**
#### 1. https://github.com/jkjung-avt/tensorrt_demos
#### 2. https://github.com/kfengtee/crnn-license-plate-OCR
