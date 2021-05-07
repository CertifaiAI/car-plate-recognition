'''
This script perform data augmentation with bouding boxes for object detection project
Support Yolo txt file format 

There are 4 augmentation methods
1. Add noise
2. Add brightness or remove brightness
3. Rotate
4. Horizontal Flip
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
    contrast=iaa.GammaContrast(gamma=gamma_value)
    contrast_image =contrast.augment_image(image)
    return contrast_image

def horizontal_flip(image, bbs):
    flip_hr=iaa.Fliplr(p=1.0)
    flip_hr_image, bbs_aug= flip_hr(image=image, bounding_boxes=bbs)
    return flip_hr_image, bbs_aug

def add_noise(image, noise):
    gaussian_noise=iaa.AdditiveGaussianNoise(10,noise)
    noise_image=gaussian_noise.augment_image(image)
    return noise_image

# def sheer_image(image, bbs):
#     scale_im=iaa.Affine(scale={"x": (1.5, 1.0), "y": (1.5, 1.0)})
#     scale_image, bbs_aug=scale_im(image=image, bounding_boxes=bbs)
#     return scale_image, bbs_aug

def rotate_img(image, bbs, degree):
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
args.add_argument("-N", type=str2bool, nargs='?', const=True, default=False, help="use noise")
args.add_argument("-R", type=str2bool, nargs='?', const=True, default=False, help="use rotate")
args.add_argument("-B", type=str2bool, nargs='?', const=True, default=False, help="use brightness contrast")
args.add_argument("-F", type=str2bool, nargs='?', const=True, default=False, help="use flip image")
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
        if fileName[-3:] == 'jpg' or fileName[-3:] == 'png':
            image_files.append(fileName[:-4])
        elif fileName[-4:] == 'jpeg':
            image_files.append(fileName)

    # Assume txt file has the same name as image
    for names in image_files:
        image_path = str(path) + '/' + names + '.jpg'
        anno_path = str(path) + '/' + names + '.txt'
        new_image_path = str(new_path) + '/' + names + 'aug.jpg'
        new_anno_path = str(new_path) + '/' + names + 'aug.txt'

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

            if cfg.B:
                # Change brightness on image only -> higher darker (3.0)
                image = change_brightness(image, gamma_value=float(cfg.bright))

            # Augmentation on bounding box and image
            if cfg.R:
                # Rotate image
                image, bbs = rotate_img(image, bbs, degree=float(cfg.degree))

            if cfg.F:
                # flipping image horizontally
                image, bbs = horizontal_flip(image, bbs)

            # Sanity Check
            ia.imshow(bbs.draw_on_image(image, size=2))

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