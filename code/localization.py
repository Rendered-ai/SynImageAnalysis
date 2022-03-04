import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch.nn as nn
from detectron2.utils.logger import setup_logger
setup_logger()


def attention_map(X, fast_rcnn, power=2):
    """
    Generate a localisation map for the predicted class given an input
    image and a pretrained CNN

    Input:
    - X: Input image: synthetic data, Size of input torch.Size([1, 3, 512, 512])
    - fast_rcnn: Pretrained Fast RCNN on real data

    Return:
    - localisation_maps : spatial attention map from FPN(feature pyramid network), p2-p6
    """
    fast_rcnn.eval()
    feature_map = fast_rcnn.backbone(X)
    localisation_maps = []
    for i in range(2, 7):
        pi = feature_map["p" + str(i)]
        att_pi = pi.pow(power).mean(1)
        localisation_maps.append(att_pi.cpu().detach())
    return np.asarray(localisation_maps)


def localisation(X, fast_rcnn):
    """
    Generate a localisation map for the predicted class given an input
    image and a pretrained CNN

    Input:
    - X: Input image: synthetic data, Size of input torch.Size([1, 3, 512, 512])
    - fast_rcnn: Pretrained Fast RCNN on real data

    Return:
    - localisation_map : feature map from FPN(feature pyramid network), p2-p6
    """
    fast_rcnn.eval()
    feature_map = fast_rcnn.backbone(X)
    localisation_maps = []
    for i in range(2, 7):
        feature_map_pi = feature_map["p" + str(i)]
        map = feature_map_pi.mean(1)
        localisation_maps.append(map.cpu().detach())
    return np.asarray(localisation_maps)


def objectness_logit_map(X, fast_rcnn):
    """
    Generate a objectness_logit_map for the predicted class given an input
    image and a pretrained detectron2 model.

    Input:
    - X: Input image: synthetic data, Size of input torch.Size([1, 3, 512, 512])
    - fast_rcnn: Pretrained Fast RCNN on real data

    Return:
    - objectness_logit_map : objectness_logit_map from FPN(feature pyramid network), p2-p6
    """
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
    """
    Plot out the localization map of level 2 to 6 of FasterRCNN.

    Input:
    - X: Input image: synthetic data, Size of input torch.Size([1, 3, 512, 512])
    - fast_rcnn: Pretrained Fast RCNN on real data

    Return:
    - A localization map from p2 - p6.
    """
    f, axarr = plt.subplots(1, 5, figsize=(17, 17))
    for i, map in enumerate(localization_maps):
        axarr[i].imshow(map[0].cpu().detach().numpy())
        axarr[i].set_title(str(map_name) + " map of p" + str(i + 2))


def upsampling(p_i, local_maps, upsampling_mode='nearest'):
    """
    Upsample the localization maps to make them into the same size of the original image.

    Input:
    - p_i: which pyramid in the FPN
    - local_maps: a list of local_maps, which contains feature maps from p2 to p6.
    - upsampling_mode: which upsampling method you choose from nn.upsample. default is 'nearest'.

    Return:
    - An upsampled localization maps of a particular level.
    """
    p_i_to_scale_factor = {0: 4, 1: 8, 2: 16, 3: 32, 4: 64}
    input = local_maps[p_i].view(1, local_maps[p_i].shape[0], local_maps[p_i].shape[1], -1)
    upsampler = nn.Upsample(scale_factor=p_i_to_scale_factor[p_i], mode='nearest')
    out = upsampler(input)
    return out


def overlay_heatmap(local_map, image, alpha=0.7, colormap=cv2.COLORMAP_JET):
    """
    Overlay the localisation map onto the original image.
    You could either plot out the maps or just return without visualization.

    Input:
    - local_map: upsampled localization map size (512,512)
    - image: orginial image (512, 512, 3) from cv2.imread()
    - alpha: weights of activation map vs orginal image
    - colormap: you choose whatever you like

    Return:
    - heatmap: the normalized localization map
    - output: the localization map overlaid on the original image.
    """
    # normalization
    heatmap = (local_map - local_map.min()) / (local_map.max() - local_map.min())
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
