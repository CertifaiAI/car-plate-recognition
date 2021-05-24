## CarGate 
### Automatic License Plate Recognition System on Jetson Nano
#### The system is built using yolov4-tiny with tensorRT (vehicle and license plate detector) [Reference 1] and PPOCR (OCR) [Reference 2]

### ** Project Structure**
Jetson Nano
- Run object detection model (Yolov4-tiny) to detect vehicle and vehicle plate
- The model will be able to detect Car, Van and licence plate

Server
- Run OCR(PPOCR) on vehicle plate
- Run Thingsboard FrontEnd

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
$ cd src/plugins
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
#### save tag is used to save video
```
$ python3 main.py --video {video_path} --save
```

#

### **Demo**
![car and license plate detection](./result.jpg)
#

### **References**
#### 1. https://github.com/jkjung-avt/tensorrt_demos
#### 2. https://github.com/kfengtee/crnn-license-plate-OCR
