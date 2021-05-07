# Model Training Pipeline for CarGate

## Target objects for object detection model
1. Car
2. Van 
3. NumberPlate

## Data Augmentation 
Currently there are 4 types of augmentations using the script, you can add up the methods to get different result

There are 4 augmentation methods
1. Add noise
2. Add brightness or remove brightness
3. Rotate
4. Horizontal Flip

## Train, Test, Split scripts for object detections
python train_test_valid.py --dir experiments --train_out train --test_out test --valid_out valid
dir         = directory of database
train_out   = path of train dataset output
test_out    = path of ts

## Experiments on different kind of models
1. Yolov4-tiny + TensorRT
2. Mobilenetv2-ssd + TensorRT
3. Yolov5 + TensorRT

## Experiments on two kind of training:
1.   Train from scratch
2.   Train on pre-trained weight of COCO dataset

## Training on batches 
Training are based on batches (Assuming that each week will have 1000 images). Each new batch will add on to previous batch and retrain as a whole. For examples, 1000 for first batch, 2000 (1000 + 1000) for second batch.

## Model Evaluation
1. Speed of model on Jetson Nano (FPS, execution time)
2. Accuracy of model on Jetson Nano (MAP, IOU)

## TODO List
- [ ] Determine a standard model evaluation method
- [ ] Script and Flow for converting models to tensorRT
- [ ] Define final structure of CarGate for easy plug in / plug out of model
- [ ] Model Training script for YOLOv4-tiny and Mobilenetv2-ssd
- [ ] Train-Test-Valid Split for object detection
