'''
This Script will perform train-test-valid split for object detection dataset
'''

import argparse
import os 
from random import shuffle
import shutil

# Specify arguements
arg = argparse.ArgumentParser()
arg.add_argument('--dir', help='Path to dataset', required=True, type=str)
arg.add_argument('--train', help='Ratio for train dataset', default=0.7, type=float)
arg.add_argument('--test', help='Ratio for test dataset', default=0.1, type=float)
arg.add_argument('--valid', help='Ratio for valid dataset', default=0.2, type=float)
arg.add_argument('--train_out', help='Train dataset output path', type=str)
arg.add_argument('--test_out', help='Test dataset output path', type=str)
arg.add_argument('--valid_out', help='Valid dataset output path', type=str)
# arg.add_argument('--annotation', help='Annotation file type', required=True, type=str)
cfg = arg.parse_args()

def check_dir_exist(path):
    # Create dir if not exists
    if not os.path.exists(path):
        os.makedirs(path)
        return

def check_ratio(train_ratio, test_ratio, valid_ratio):
    if not (round(train_ratio + test_ratio + valid_ratio)) == 1:
        raise Exception("Please Enter a valid ratio")

def split(train_ratio, test_ratio, valid_ratio, image_files):
    data_count = len(image_files)
    train_num = int(data_count * train_ratio)
    test_num = int(data_count * test_ratio)
    valid_num = int(data_count - train_num - test_num)
    print("Training dataset will have {} images".format(train_num))
    print("Valid dataset will have {} images".format(valid_num))
    print("Test dataset will have {} images".format(test_num))
    # Get train data
    train_data = image_files[:train_num]
    image_files = image_files[train_num:]
    # Get test data
    test_data = image_files[:test_num]
    image_files = image_files[test_num:]
    # Get valid data
    valid_data = image_files[:valid_num]
    return train_data, test_data, valid_data

def get_file_lists(dir):
    all_files = os.listdir(os.path.abspath(dir))
    image_files = []
    annotation_files = []
    count = 0
    for fileName in all_files:
        if fileName[-3:] == 'jpg' or fileName[-4:] == 'jpeg' or fileName[-3:] == 'png' or fileName[-3:] == 'JPG':
            image_files.append(fileName)
        elif fileName[-3:] == 'txt':
            annotation_files.append(fileName)
        count +=1
    # Shuffle files
    shuffle(image_files)
    return image_files, annotation_files, count

# Copy files to folder
def move_files(ori_dir, data, dest_dir, name):
    # Train data
    # print((data[0]))
    count = 0
    for fileName in data:
        # print(fileName)
        # Images
        oldPath = ori_dir + '/' + fileName
        newPath = dest_dir + '/' + fileName
        shutil.copy(oldPath, newPath)

        # remove .jpeg from filename to get annotation filename
        if fileName[-4:] == 'jpeg':
            annotation_oldPath = ori_dir + '/' + fileName[:-5] + '.txt'
            annotation_newPath = dest_dir + '/' + fileName[:-5] + '.txt'
            shutil.copy(annotation_oldPath, annotation_newPath)
        elif fileName[-3:] == 'jpg' or fileName[-3:] == 'png' or fileName[-3:] == 'JPG':
        # Annotation
        # remove .png or .jpg from filename to get annotation filename
            # print(fileName[:-4])
            annotation_oldPath = ori_dir + '/' + fileName[:-4] + '.txt'
            annotation_newPath = dest_dir + '/' + fileName[:-4] + '.txt'
            shutil.copy(annotation_oldPath, annotation_newPath)
        count += 1
    print('{} images copied to {} folder'.format(count, name))


def main():
    # Check correct ratio
    check_ratio(cfg.train, cfg.test, cfg.valid)
    # Check entered path exists or not
    # check_dir_exist(cfg.train_out)
    # check_dir_exist(cfg.test_out)
    # check_dir_exist(cfg.valid_out)
    # Get files from directory
    images, annotations, counts = get_file_lists(cfg.dir)
    print('Total number of images detected: {}'.format(len(images)))
    # Split files
    train_data, test_data, valid_data = split(cfg.train, cfg.test, cfg.valid, images)
    # Copy files into train test valid folder
    # move_files(cfg.dir, train_data, cfg.train_out, 'train')
    # move_files(cfg.dir, test_data, cfg.test_out, 'test')
    # move_files(cfg.dir, valid_data, cfg.valid_out, 'valid')
    os.makedirs(cfg.dir+'/train')
    os.makedirs(cfg.dir+'/test/')
    os.makedirs(cfg.dir+'/valid')
    move_files(cfg.dir, train_data, cfg.dir+'/train', 'train')
    move_files(cfg.dir, test_data, cfg.dir+'/test', 'test')
    move_files(cfg.dir, valid_data, cfg.dir+'/valid', 'valid')
    # Zip the folder

if __name__ == '__main__':
    main()