import aug_util as aug
import wv_util as wv
import matplotlib.pyplot as plt
import numpy as np
import csv
from tqdm import tqdm
import numpy as np
from PIL import Image
import tensorflow as tf
from PIL import Image, ImageDraw
import skimage.filters as filters


def findCranesImages(coords, chips, classes):
    desired_img_files = []
    # We only want to coordinates and classes that are within our chip
    for chip_name in tqdm(list(np.unique(chips))):
        # _coords = coords[chips==chip_name]
        _classes = classes[chips==chip_name].astype(np.int64)
        for c in [32, 54, 59]:
            if c in np.unique(_classes):
                desired_img_files.append(chip_name)
                print('keep image: {}'.format(chip_name))
    return desired_img_files



def load_img_tiles(chip_name, coords, classes, img_dir=''):
    arr = wv.get_image(img_dir+chip_name)
    # We only want to coordinates and classes that are within our chip
    arr_coords = coords[chips==chip_name]
    arr_classes = classes[chips==chip_name].astype(np.int64)

    #We can chip the image into 512x512 chips
    c_img, c_box, c_cls = wv.chip_image(img=arr, coords=arr_coords, classes=arr_classes, shape=tile_size)
    # print("Num Chips: %d" % c_img.shape[0])
    return c_img, c_box, c_cls


def findCraneTiles(c_img, c_box, c_cls, desired_labels = (32, 54, 59)): 
    idx_to_keep = []
    for i in c_cls:
        tmp = 0
        for cls in desired_labels:
            if cls in c_cls[i]:
                tmp+=1
        if tmp>0:
            idx_to_keep.append(i)
    c_imgs_to_keep = c_img[idx_to_keep]
    c_box_to_keep = [c_box[i] for i in idx_to_keep]
    c_cls_to_keep = [c_cls[i] for i in idx_to_keep]
    return c_imgs_to_keep, c_box_to_keep, c_cls_to_keep



def get_bboxes(img, boxes, classes, output_dir, desired_classes=(32,54,59)):
    """
    A helper function to draw bounding box rectangles on images

    Args:
        img: image to be drawn on in array format
        boxes: An (N,4) array of bounding boxes
        classes: array of labels

    Output:
        Image with drawn bounding boxes
    """
    source = Image.fromarray(img)
    draw = ImageDraw.Draw(source)
    w2,h2 = (img.shape[0],img.shape[1])

    classes = list(classes)

    count = 0
    # cropped_imgs = []
    for i, b in enumerate(boxes):
        xmin,ymin,xmax,ymax = [int(x) for x in b]
        cropped_img = img[ymin:ymax, xmin:xmax, :]
        # print(cropped_img)
        cropped_source = Image.fromarray(cropped_img)
        label = classes[i]
        if label in desired_classes:
            count += 1
            cropped_source.save(output_dir+'img='+str(img_id)+'_label='+str(label)+'_count='+str((count))+".tif")
    print('Now saved {} bbox at: {}'.format(count, output_dir))
        # cropped_imgs.append(cropped_source)
        # for j in range(3):
        #     draw.rectangle(((xmin+j, ymin+j), (xmax+j, ymax+j)), outline="red")
    # return cropped_imgs



real_train_images_dir = '/content/drive/MyDrive/111 Rendered.ai/xview/real_data/train_images/'
jeojson_f = '/content/drive/MyDrive/111 Rendered.ai/xview/real_data/xView_train.geojson'
chip_name = '104.tif'
output_dir = '/content/drive/MyDrive/111 Rendered.ai/xview/real_data/real_data_chips/'
tile_size = (512, 512)


coords, chips, classes = wv.get_labels(jeojson_f)
desired_img_files=findCranesImages(coords, chips, classes)

c_imgs, c_boxes, c_clses = [], [], []
tiles_count = 0
for _, chip_name in tqdm(enumerate(desired_img_files)):
    c_img, c_box, c_cls = load_img_tiles(chip_name, coords, classes, img_dir=real_train_images_dir)
    c_imgs_to_keep, c_box_to_keep, c_cls_to_keep = findCraneTiles(c_img, c_box, c_cls)
    if c_cls_to_keep:
        tiles_count += len(c_cls_to_keep)
        # print('tiles count now:', tiles_count)
        c_imgs.append(c_imgs_to_keep)
        c_boxes.append(c_box_to_keep)
        c_clses.append(c_cls_to_keep)

tile_images = np.vstack(c_imgs)
tile_labels = np.asarray([list(l) for tile in c_clses for l in tile])
tile_bboxes = np.array([bbox for tile in c_boxes for bbox in tile])

# save images
# for i, img in tqdm(enumerate(tile_images)):
#     Image.fromarray(img).save(output_dir+str(i)+'.jpg')

# save coco ann
# ann = pd.concat([pd.DataFrame(tile_labels, columns=['labels']),pd.DataFrame(tile_bboxes, columns=['bbox'])], axis=1).reset_index().rename(columns={'index':'image_id'})
# ann.to_csv(output_dir+'annotations.csv')

# We can visualize the chips with their labels
# for img_id in tqdm(range(tile_images.shape[0])):
#     get_bboxes(tile_images[img_id], tile_bboxes[img_id], tile_labels[img_id], output_dir)
