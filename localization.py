import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, json, cv2, random, math
from google.colab.patches import cv2_imshow
import torch 
import torchvision
import torch.nn as nn 
from torchvision import transforms
from torchvision.models._utils import IntermediateLayerGetter
from torch.utils.data import DataLoader
import PIL
from IPython.display import Image 
# from PIL import Image
import requests
from tqdm import tqdm 
from sklearn import decomposition    
from sklearn.preprocessing import MinMaxScaler
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.modeling import build_model
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import load_data  


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 12345
random.seed(seed)
torch.manual_seed(seed)


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
    for i in range(2,7):
        pi = feature_map["p"+str(i)]
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
    for i in range(2,7):
        feature_map_pi = feature_map["p"+str(i)]
        map = feature_map_pi.mean(1)
        localisation_maps.append(map)
    return np.asarray(localisation_maps)


def objectness_logit_map(X, fast_rcnn):
    fast_rcnn.eval()
    feature_map = fast_rcnn.backbone(X)
    objectness_logit_maps = []
    for i in range(2,7):
        feature_map_pi = feature_map["p"+str(i)]
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
    f, axarr = plt.subplots(1,5,figsize=(17,17))
    for i, map in enumerate(localization_maps):
        axarr[i].imshow(map[0].cpu().detach().numpy())
        axarr[i].set_title(str(map_name)+ " map of p"+str(i+2))


def upsampling(p_i, local_maps, upsampling_mode='nearest'):
    """
    Input:
    p_i: which pyramid in the FPN
    local_maps: a list of local_maps, which contains feature maps from p2 to p6.
    upsampling_mode: which upsampling method you choose from nn.upsample. default is 'nearest'.
    """
    p_i_to_scale_factor = {0:4, 1:8, 2:16, 3:32, 4:64}
    input = local_maps[p_i].view(1,local_maps[p_i].shape[0],local_maps[p_i].shape[1],-1)
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
    heatmap = (local_map -local_map.min()) / (local_map.max()-local_map.min())
    # apply the supplied color map to the heatmap 
    CAM = cv2.applyColorMap(np.uint8(255 * heatmap), colormap)
    # overlay the heatmap on the input image
    output = cv2.addWeighted(image, alpha, CAM, 1 - alpha, 0)
    # plot the output
    # f, axarr = plt.subplots(1,3,figsize=(15,15))
    # axarr[0].imshow(local_map)
    # axarr[0].set_title("local map")
    # axarr[1].imshow(heatmap)
    # axarr[1].set_title('Nomalized Heatmap')
    # axarr[2].imshow(output)
    # plt.title('Activation Map')
    return (heatmap, output)


real_coco_dir = "/content/drive/MyDrive/111 Rendered.ai/RarePlanes/datasets/coco_data/aircraft_real_test_coco.json"
real_images_dir = "/content/drive/MyDrive/111 Rendered.ai/RarePlanes/datasets/real/test/RarePlanes_test_PS-RGB_tiled.tar.gz (Unzipped Files)/PS-RGB_tiled"
pred_coco_dir = "/content/drive/MyDrive/111 Rendered.ai/RarePlanes/output/coco_instances_results.json"
pred_images_dir = "/content/drive/MyDrive/111 Rendered.ai/RarePlanes/datasets/synthetic/test/images/"

model_config_file = "COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"
model_weight_file = "/content/drive/My Drive/111 Rendered.ai/rareplane_models/model_0043999.pth"

# set output dir 
npy_dir = "/content/sample_data/"
jpg_dir = "/content/sample_data/"


# load model
cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file(model_config_file))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_weight_file
model = build_model(cfg)

# load data
ann = load_data.get_image_ann(real_coco_dir, real_images_dir)
preds_coco = load_data.load_preds(pred_coco_dir, pred_images_dir)
img_ids = load_data.get_wrong_pred_img_id(preds_coco)
img_dirs = load_data.id2filename(img_ids, ann)
img_collections = load_data.load_wrong_images(img_dirs)

# send both the model to gpu
fast_rcnn = model.to(device)

imagenet_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),])

n_imgs = img_collections.shape[0]
for i in tqdm(range(n_imgs)):
    im = img_collections[i]
    # send data to gpu
    X = imagenet_transform(im).unsqueeze(0).to(device)
    attention_maps = attention_map(X, fast_rcnn)
    for fpn_layer in range(5):
        att = upsampling(fpn_layer, attention_maps, upsampling_mode='nearest')
        att = att.cpu().detach().numpy()[0][0]
        (heatmap, output) = overlay_heatmap(att, im, alpha=0.75)
        # save as numpy arr
        output_name = "att_map_"+str(i)+"_fpn_layer="+str(fpn_layer+2)
        np.save(npy_dir+output_name, output)
        # save a jpg
        PIL.Image.fromarray(output).save(jpg_dir+output_name+".jpeg")