'''
This script perform data augmentation with bounding boxes for object detection project
Support Yolo txt file format 

There are 4 augmentation methods
1. Add noise
2. Add rain
3. Add brightness or remove brightness
4. Rotate
5. Horizontal Flip

References:
https://www.freecodecamp.org/news/image-augmentation-make-it-rain-make-it-snow-how-to-modify-a-photo-with-machine-learning-163c0cb3843f/
https://github.com/aleju/imgaug

'''

import imageio
import imgaug as ia
import imgaug.augmenters as iaa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
from yolo2annotation import Yolo2Annotation, Annotation2Yolo
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import argparse
import os
from pathlib import Path
import cv2
import random

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def change_brightness(image, gamma_value):
    if gamma_value==-1:
        gamma_value = np.random.uniform(2.0,2.6)
    contrast=iaa.GammaContrast(gamma=gamma_value)
    contrast_image =contrast.augment_image(image)
    return contrast_image

def horizontal_flip(image, bbs):
    flip_hr=iaa.Fliplr(p=1.0)
    flip_hr_image, bbs_aug= flip_hr(image=image, bounding_boxes=bbs)
    return flip_hr_image, bbs_aug

def add_noise(image, noise):
    if noise==-1:
        noise = np.random.randint(40,80)
    gaussian_noise=iaa.AdditiveGaussianNoise(10,noise)
    noise_image=gaussian_noise.augment_image(image)
    return noise_image

def add_snow(image):    
    image_HLS = cv2.cvtColor(image,cv2.COLOR_RGB2HLS) 
    ## Conversion to HLS    
    image_HLS = np.array(image_HLS, dtype = np.float64)     
    brightness_coefficient = 2.5     
    snow_point=140 ## increase this for more snow    
    image_HLS[:,:,1][image_HLS[:,:,1]<snow_point] = image_HLS[:,:,1][image_HLS[:,:,1]<snow_point]*brightness_coefficient 
    ## scale pixel values up for channel 1(Lightness)    
    image_HLS[:,:,1][image_HLS[:,:,1]>255]  = 255 
    ##Sets all values above 255 to 255    
    image_HLS = np.array(image_HLS, dtype = np.uint8)    
    image_RGB = cv2.cvtColor(image_HLS,cv2.COLOR_HLS2RGB) 
    ## Conversion to RGB    
    return image_RGB

def generate_random_lines(imshape,slant,drop_length):    
    drops=[]    
    for i in range(800):
        ## If You want heavy rain, try increasing this        
        if slant<0:            
            x= np.random.randint(slant,imshape[1])
        else:            
            x= np.random.randint(0,imshape[1]-slant)        
        y= np.random.randint(0,imshape[0]-drop_length)
        drops.append((x,y))
    return drops            
    
def add_rain(image, slant):
    if slant==-1:
        slant = random.randint(-10, 10)
    imshape = image.shape
    # Rain drop numbers
    drop_length=random.randint(10, 15)
    drop_width=1
    drop_color=(200,200,200) 
    ## a shade of gray    
    rain_drops= generate_random_lines(imshape,slant,drop_length)        
    for rain_drop in rain_drops:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Adding this line solves the "Overload resolution" error van-back
        cv2.line(image,(rain_drop[0],rain_drop[1]),(rain_drop[0]+slant,rain_drop[1]+drop_length),drop_color,drop_width)
    image= cv2.blur(image,(7,7))  ## rainy view are blurry        
    brightness_coefficient = 0.7 ## rainy days are usually shady     
    image_HLS = cv2.cvtColor(image,cv2.COLOR_RGB2HLS) ## Conversion to HLS    
    image_HLS[:,:,1] = image_HLS[:,:,1]*brightness_coefficient ## scale pixel values down for channel 1(Lightness)    
    image_RGB = cv2.cvtColor(image_HLS,cv2.COLOR_HLS2RGB) ## Conversion to RGB    
    return image_RGB

def rotate_img(image, bbs, degree):
    if degree==-1:
        degree = np.random.randint(-10,10)
    rotate_bb=iaa.Affine(rotate=(degree))
    image_aug, bbs_aug = rotate_bb(image=image, bounding_boxes=bbs)
    return image_aug, bbs_aug

def get_relative_path(dir):
    relative = Path(dir)
    absolute = relative.absolute()
    return absolute

def check_file_exist(path):
    if os.path.exists(path):
        return True
    return False  

def create_dir(path):
    # Create dir if not exists
    if not os.path.exists(path):
        os.makedirs(path)
        return

args = argparse.ArgumentParser()
args.add_argument('--dir', help="Path to data directory to augment", required=True)
args.add_argument('--dest', help="Path to save directory", required=True)
args.add_argument('--bright', help="brightness value for image, more than 3 will be dark", default=0.5, type=float)
args.add_argument('--noise', help="noise value for image, higher will have more noise", default=90, type=float)
args.add_argument('--degree', help="degree value for rotation", default=10, type=float)
args.add_argument('--slant', help="number of raindrops, higher is more rain", default=8, type=int)
args.add_argument("-N", type=str2bool, nargs='?', const=True, default=False, help="use noise")
args.add_argument("-R", type=str2bool, nargs='?', const=True, default=False, help="use rotate")
args.add_argument("-B", type=str2bool, nargs='?', const=True, default=False, help="use brightness contrast")
args.add_argument("-F", type=str2bool, nargs='?', const=True, default=False, help="use flip image")
args.add_argument("-Rain", type=str2bool, nargs='?', const=True, default=False, help="add raining effect to image")
args.add_argument("-Snow", type=str2bool, nargs='?', const=True, default=False, help="add snow effect to image")
cfg = args.parse_args()

def main():
    create_dir(cfg.dest)
    # Files in dir
    all_files = os.listdir(os.path.abspath(cfg.dir))
    image_files = []
    path = get_relative_path(cfg.dir)
    new_path = get_relative_path(cfg.dest)

    # Filter through all file and organize to specific category
    for fileName in all_files:
        if fileName[-3:] == 'jpg' or fileName[-3:] == 'png' or fileName[-3:] == 'JPG':
            image_files.append(fileName)
        elif fileName[-4:] == 'jpeg':
            image_files.append(fileName)

    # Assume txt file has the same name as image
    # Cant get names of "jpeg" images because of the 4 characters
    for names in image_files:
        if names[-3:] == 'jpg' or names[-3:] == 'png' or names[-3:] == 'JPG':
            imgFormat = names[-3:]
            names = names[:-4]
        elif names[-4:] == 'jpeg':
            imgFormat = names[-4:]
            names = names[:-5]
        # print(imgFormat)
        image_path = str(path) + '/' + names + '.' + imgFormat
        anno_path = str(path) + '/' + names + '.txt'
        # new_image_path = str(new_path) + '/' + names + 'aug.' + imgFormat
        # new_anno_path = str(new_path) + '/' + names + 'aug.txt'

        # Extract coordinates from txt file
        txt_file = open(anno_path, 'r')
        lines = txt_file.readlines()

        # import image
        imageORi = imageio.imread(image_path)

        # loop through all annotations
        for line in lines:
            image = imageORi
            # Yolo text format first is class
            targetClass = line.split(' ')[0]
            coordinates = line.split(' ')[1:5]
            x, y, w, h = coordinates
            x_min, x_max, y_min, y_max = Yolo2Annotation(image_path, float(x), float(y), float(w), float(h))
            bbs = BoundingBoxesOnImage([BoundingBox(x1=x_min, x2=x_max, y1=y_min, y2=y_max)], shape=imageORi.shape)

            # Augmentation on image
            if cfg.N:
                # Add noise -> higher more noise
                image = add_noise(image, noise=float(cfg.noise))
                new_image_path = str(new_path) + '/' + names + 'Noise.' + imgFormat
                new_anno_path = str(new_path) + '/' + names + 'Noise.txt'

            if cfg.Snow:
                image = add_snow(image)

            if cfg.Rain:
                # Add noise -> higher more noise
                image = add_rain(image, slant=int(cfg.slant))
                new_image_path = str(new_path) + '/' + names + 'Rain.' + imgFormat
                new_anno_path = str(new_path) + '/' + names + 'Rain.txt'

            if cfg.B:
                # Change brightness on image only -> higher darker (3.0)
                image = change_brightness(image, gamma_value=float(cfg.bright))
                new_image_path = str(new_path) + '/' + names + 'Night.' + imgFormat
                new_anno_path = str(new_path) + '/' + names + 'Night.txt'

            # Augmentation on bounding box and image
            if cfg.R:
                # Rotate image
                image, bbs = rotate_img(image, bbs, degree=float(cfg.degree))
                new_image_path = str(new_path) + '/' + names + 'Rotate.' + imgFormat
                new_anno_path = str(new_path) + '/' + names + 'Rotate.txt'

            if cfg.F:
                # flipping image horizontally
                image, bbs = horizontal_flip(image, bbs)

            # Sanity Check
           # ia.imshow(bbs.draw_on_image(image, size=2))

            # extract coordinates from BoundingBoxesOnImage class
            x_min, y_min = bbs[0][0]
            x_max, y_max = bbs[0][1]

            # convert back to yolo format
            x, y, w, h = Annotation2Yolo(image_path, x_min, x_max, y_min, y_max)
            single_line = [str(targetClass), str(x), str(y), str(w), str(h)]
            # Save annotations to a file
            if check_file_exist(new_anno_path):
                f = open(new_anno_path, "a")
                f.write('\n')
                for element in single_line:
                    f.write(element + ' ')
                f.close()
            else:
                f = open(new_anno_path, "w")
                for element in single_line:
                    f.write(element + ' ')
                f.close()
        
        # Save image 
        imageio.imsave(new_image_path, image)

if __name__=="__main__":
    main()