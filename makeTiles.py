'''
Usage: python3 makeTiles.py -image_dir ... -jeojson_dir ... -output_dir ... \
--chip_size ... --desired_classes ...

Output: This makeTiles python script with return chips of specific classes
'''
import aug_util as aug
import wv_util as wv
import matplotlib.pyplot as plt
import numpy as np
import csv
from tqdm import tqdm_notebook
import pandas as pd


# input params
real_train_images_dir = '/content/drive/MyDrive/111 Rendered.ai/xview/real_data/train_images/'
jeojson_f = '/content/drive/MyDrive/111 Rendered.ai/xview/real_data/xView_train.geojson'
output_dir = '/content/drive/MyDrive/111 Rendered.ai/xview/real_data/real_data_chips/'


def findCranesImages(coords, chips, classes):
    desired_img_files = []
    # We only want to coordinates and classes that are within our chip
    for chip_name in tqdm_notebook(list(np.unique(chips))):
        # _coords = coords[chips==chip_name]
        _classes = classes[chips == chip_name].astype(np.int64)
        for c in [32, 54, 59]:
            if c in np.unique(_classes):
                desired_img_files.append(chip_name)
                print('keep image: {}'.format(chip_name))
    return desired_img_files


def load_img_tiles(chip_name, coords, classes, img_dir=''):
    arr = wv.get_image(img_dir + chip_name)
    # We only want to coordinates and classes that are within our chip
    arr_coords = coords[chips == chip_name]
    arr_classes = classes[chips == chip_name].astype(np.int64)
    # We can chip the image into 500x500 chips
    c_img, c_box, c_cls = wv.chip_image(img=arr, coords=arr_coords, classes=arr_classes, shape=(512, 512))
    return c_img, c_box, c_cls


def findCraneTiles(c_img, c_box, c_cls, desired_labels=(32, 54, 59)):
    idx_to_keep = []
    for i in c_cls:
        tmp = 0
        for cls in desired_labels:
            if cls in c_cls[i]:
                tmp += 1
        if tmp > 0:
            idx_to_keep.append(i)
    c_imgs_to_keep = c_img[idx_to_keep]
    c_box_to_keep = [c_box[i] for i in idx_to_keep]
    c_cls_to_keep = [c_cls[i] for i in idx_to_keep]
    return c_imgs_to_keep, c_box_to_keep, c_cls_to_keep


# load an image
arr = wv.get_image(real_train_images_dir + chip_name)
# load labels
coords, chips, classes = wv.get_labels(jeojson_f)
# get coordinates and classes that are within our chip
coords = coords[chips == chip_name]
classes = classes[chips == chip_name].astype(np.int64)
# get chips of certain labels
desired_img_files = findCranesImages(coords, chips, classes)
# save desired chip file names
# pd.DataFrame(np.asarray(desired_img_files), columns=['crane_imgs']).to_csv('/content/drive/MyDrive/111 Rendered.ai/xview/real_data/crane_img_list.csv')

c_imgs, c_boxes, c_clses = [], [], []
tiles_count = 0
for _, chip_name in tqdm_notebook(enumerate(desired_img_files)):
    # img_idx=4
    # chip_name = desired_img_files[img_idx]
    c_img, c_box, c_cls = load_img_tiles(chip_name, coords, classes, img_dir='')

    c_imgs_to_keep, c_box_to_keep, c_cls_to_keep = findCraneTiles(c_img, c_box, c_cls)
    if c_cls_to_keep:
        tiles_count += len(c_cls_to_keep)
        print('tiles count now:', tiles_count)
        c_imgs.append(c_imgs_to_keep)
        c_boxes.append(c_box_to_keep)
        c_clses.append(c_cls_to_keep)

tile_images = np.vstack(c_imgs)
tile_labels = np.asarray([list(l) for tile in c_clses for l in tile])
tile_bboxes = np.array([bbox for tile in c_boxes for bbox in tile])
print(len(tile_images), len(tile_labels), len(tile_bboxes))

for i, img in tqdm_notebook(enumerate(tile_images)):
    Image.fromarray(img).save(output_dir + str(i) + '.jpg')