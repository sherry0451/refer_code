import numpy as np
# import pandas as pd
import os
import cv2
import csv

def readImage(path):
    # OpenCV reads the image in bgr format by default
    bgr_img = cv2.imread(path)
    # We flip it to rgb for visualization purposes
    b,g,r = cv2.split(bgr_img)
    rgb_img = cv2.merge([r,g,b])
    return rgb_img


tif_folder = '/media/cz/E/Ubuntu/Research/Code/Medical/pcam_kaggle/input/train/'
csv_folder = '/media/cz/E/Ubuntu/Research/Code/Medical/pcam_kaggle/input/cz/split_csv/'
csv_prefix = 'train_shuffle_split'
save_folder = '/media/cz/E/Ubuntu/Research/Code/Medical/pcam_kaggle/kaggle_npy/'


imgs_dict = {}
labels_dict = {}
for iteration in range(10):
    with open(csv_folder+csv_prefix+str(iteration)+'.csv', 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)
        print(headers)
        for index, (id_, label) in enumerate(reader):
            img = readImage(tif_folder+id_+'.tif')
            label = int(label)

            imgs_dict[str(index)] = np.asarray(img, dtype=np.uint8)
            labels_dict[str(index)] = np.asarray(label, dtype=np.int8)

            # del img, label

            if int(index) % 10000 == 0:
                print(index, id_, label)
            del img, label

    np.save(save_folder+csv_prefix+str(iteration)+'_img', imgs_dict)
    np.save(save_folder+csv_prefix+str(iteration)+'_label', labels_dict)
