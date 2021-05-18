'''
Localization
Input:  (any pretrained model, im(numpy.array, tensor))
Specify the models:
Detectron2, resnet,  [optional all models inherited from torch]
Get feature maps | activation
Take avg
Take gram matrix: style representation Optional
Get spatial attention maps, order=2
	Output: map arr
'''

# Some basic setup:
import torch
import torchvision
import torch.nn as nn
from IPython.display import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import random
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 12345
random.seed(seed)
torch.manual_seed(seed)
import PIL
import requests
from torchvision.models._utils import IntermediateLayerGetter

# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger

setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.modeling import build_model
from detectron2.engine import DefaultTrainer
from detectron2.checkpoint import DetectionCheckpointer

import json
import pandas as pd
import math
from tqdm import tqdm
import cv2
import numpy as np
import os

import torch
import torchvision
import torch.nn as nn
from IPython.display import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import random
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 12345
random.seed(seed)
torch.manual_seed(seed)
import PIL
import requests
from PIL import Image

from sklearn import decomposition
from sklearn.preprocessing import MinMaxScaler


def attention_map(X, fast_rcnn, power=2):
    """
    Generate a localisation map for the predicted class given an input
    image and a pretrained CNN

    Inputs:
    - X: Input image: synthetic data, Size of input torch.Size([1, 3, 512, 512])
    - fast_rcnn: Pretrained Fast RCNN on real data

    Returns:
    - localisation_maps : spatial attention map from FPN(feature pyramid network), p2-p6
    """
    fast_rcnn.eval()
    feature_map = fast_rcnn.backbone(X)
    localisation_maps = []
    for i in range(2, 7):
        pi = feature_map["p" + str(i)]
        att_pi = pi.pow(power).mean(1)
        localisation_maps.append(att_pi)
    return np.asarray(localisation_maps)


def localisation(X, fast_rcnn):
    """
    Generate a localisation map for the predicted class given an input
    image and a pretrained CNN

    Inputs:
    - X: Input image: synthetic data, Size of input torch.Size([1, 3, 512, 512])
    - fast_rcnn: Pretrained Fast RCNN on real data

    Returns:
    - localisation_map : feature map from FPN(feature pyramid network), p2-p6
    """
    fast_rcnn.eval()
    feature_map = fast_rcnn.backbone(X)
    localisation_maps = []
    for i in range(2, 7):
        feature_map_pi = feature_map["p" + str(i)]
        map = feature_map_pi.mean(1)
        localisation_maps.append(map)
    return np.asarray(localisation_maps)


def objectness_logit_map(X, fast_rcnn):
    fast_rcnn.eval()
    feature_map = fast_rcnn.backbone(X)
    objectness_logit_maps = []
    for i in range(2, 7):
        feature_map_pi = feature_map["p" + str(i)]
        out = fast_rcnn.proposal_generator.rpn_head.conv(feature_map_pi)
        # print(out.shape)
        out = fast_rcnn.proposal_generator.rpn_head.objectness_logits(out)
        # print(out.shape)
        sigmoid_fn = nn.Sigmoid()
        out = sigmoid_fn(out)
        map = out[0]
        objectness_logit_maps.append(map)
    return np.asarray(objectness_logit_maps)


def plotting(localization_maps, map_name):
    f, axarr = plt.subplots(1, 5, figsize=(17, 17))
    for i, map in enumerate(localization_maps):
        axarr[i].imshow(map[0].cpu().detach().numpy())
        axarr[i].set_title(str(map_name) + " map of p" + str(i + 2))


def upsampling(p_i, local_maps, upsampling_mode='nearest'):
    """
    Input:
    p_i: which pyramid in the FPN
    local_maps: a list of local_maps, which contains feature maps from p2 to p6.
    upsampling_mode: which upsampling method you choose from nn.upsample. default is 'nearest'.
    """
    p_i_to_scale_factor = {0: 4, 1: 8, 2: 16, 3: 32, 4: 64}
    input = local_maps[p_i].view(1, local_maps[p_i].shape[0], local_maps[p_i].shape[1], -1)
    upsampler = nn.Upsample(scale_factor=p_i_to_scale_factor[p_i], mode='nearest')
    out = upsampler(input)
    return out


def overlay_heatmap(local_map, image, alpha=0.7, colormap=cv2.COLORMAP_JET):
    """
    Input:
    local_map: upsampled localization map size (512,512)
    image: orginial image (512, 512, 3) from cv2.imread()
    alpha: weights of activation map vs orginal image
    colormap: you choose whatever you like
    """
    # normalization
    heatmap = (local_map - local_map.min()) / (local_map.max() - local_map.min())
    # apply the supplied color map to the heatmap
    CAM = cv2.applyColorMap(np.uint8(255 * heatmap), colormap)
    # overlay the heatmap on the input image
    output = cv2.addWeighted(image, alpha, CAM, 1 - alpha, 0)
    return (heatmap, output)


def plot_maps(im, heatmap, bbox=bbox):
    '''
    :param im:
    :param heatmap:
    :return:
    '''
    f, axarr = plt.subplots(1,3,figsize=(15,15))
    axarr[0].imshow(im)
    axarr[0].set_title("original image")
    # you need to use bbox to show labels on im
    pass
    #
    axarr[1].imshow(heatmap)
    axarr[1].set_title('Heatmap')
    axarr[2].imshow(output)
    plt.title('Activation Map')
    plt.show()