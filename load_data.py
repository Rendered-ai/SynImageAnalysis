import numpy as np
import pandas as pd
import json
from tqdm import tqdm
import cv2
def load_preds(coco_dir, images_dir):
    '''
    load wrong predictions on real data
    '''
    with open(coco_dir) as json_file:
        preds_coco = json.load(json_file)
    preds_coco = pd.DataFrame(preds_coco)
    return preds_coco


def get_image_ann(coco_dir, images_dir):
    '''
    load image annotations
    '''
    with open(coco_dir) as json_file:
        test_coco = json.load(json_file)
    img_list = pd.DataFrame(test_coco['images'])
    annotations = pd.DataFrame(test_coco['annotations'])
    ann = annotations.merge(img_list, how='left', left_on="image_id", right_on="id")
    return ann


def get_wrong_pred_img_id(preds_coco, threshold = 0.8):
    '''
    get hesitate predicted img ids
    '''
    img_ids = preds_coco[preds_coco["score"]<threshold].image_id.drop_duplicates().reset_index().image_id
    # print("There are {} images with with a bit hesitation (prob<0.8). Let's study those images!".format(img_ids.shape[0]))
    return img_ids


def id2filename(img_ids, ann):
    # get img file dir
    return pd.DataFrame(img_ids, columns=["image_id"]).merge(ann[["image_id","file_name"]], 
                                                             how="left", on="image_id").drop_duplicates()


def load_wrong_images(img_dirs):
    images_dir = "/content/drive/MyDrive/111 Rendered.ai/RarePlanes/datasets/real/test/RarePlanes_test_PS-RGB_tiled.tar.gz (Unzipped Files)/PS-RGB_tiled/"
    img_collections = []
    for f in tqdm(img_dirs.file_name):
        im = cv2.imread(images_dir+f)
        img_collections.append(im)
    img_collections = np.asarray(img_collections)  
    print("img_collections shape:",img_collections.shape)
    return img_collections