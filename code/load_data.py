import numpy as np
import pandas as pd
import json
from tqdm import tqdm
import cv2


def load_preds(coco_dir):
    """
    Load predictions of fasterRCNN

    :param coco_dir: A prediction coco file's directory
    :return:  coco file in pandas df format.
    """
    with open(coco_dir) as json_file:
        preds_coco = json.load(json_file)
    preds_coco = pd.DataFrame(preds_coco)
    return preds_coco


def get_image_ann(coco_dir):
    """
    Load coco annotations.

    :param coco_dir: coco annotation file's directory.
    :return: coco annotation file in pandas df format.
    """
    with open(coco_dir) as json_file:
        test_coco = json.load(json_file)
    img_list = pd.DataFrame(test_coco['images'])
    annotations = pd.DataFrame(test_coco['annotations'])
    ann = annotations.merge(img_list, how='left', left_on="image_id", right_on="id")
    return ann


def get_wrong_pred_img_id(preds_coco, threshold=0.8):
    """
    Get wrong / hesitated image ids.
    
    :param preds_coco:  A coco annotation file in pandas df format.
    :param threshold: Specify prediction score. by default=0.8
    :return: A list of unconfident image ids.
    """
    img_ids = preds_coco[preds_coco["score"] < threshold].image_id.drop_duplicates().reset_index().image_id
    # print("There are {} images with with a bit hesitation (prob<0.8). Let's study those images!".format(
    # img_ids.shape[0]))
    return img_ids


def id2filename(img_ids, ann):
    """
    Turn image id list to image filename list.

    :param img_ids: A list of not confident image ids.
    :param ann: Coco annotation file in pandas df format.
    :return: A list of unconfident predictions' image information, including filename.
    """
    return pd.DataFrame(img_ids, columns=["image_id"]).merge(ann[["image_id", "file_name"]],
                                                             how="left", on="image_id").drop_duplicates()


def load_wrong_images(img_info, images_dir):
    """
    Load unconfident images.

    :param img_info: A list of unconfident predictions' image information, including filename.
    :param images_dir: The directory of images.
    :return: A collection of images in numpy arrays.
    """
    img_collections = []
    for f in tqdm(img_info.file_name):
        im = cv2.imread(images_dir + f)
        img_collections.append(im)
    img_collections = np.asarray(img_collections)
    print("img_collections shape:", img_collections.shape)
    return img_collections
