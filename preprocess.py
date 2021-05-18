'''
Preprocess
Make tiles add input here
Input:
coco_json | geojson What are some other formats? Optional?
Img_folder_dir
Output_dir
Tile_size
Only select tiles with desired classes. Folder dir:
- tiles
<name of the tiles> match the syn data.
Coco - format
Output_img_name: 001-img_<tile_order>
Make bbox add input here
Input
coco_json | geojson
Img_folder_dir
output_dir
File dir
			-bbox
				-class1
				-class2
				-class3
				…
			<img_name> _<class_id >_<obj_id>
'''


import numpy as np
from PIL import Image
import tensorflow as tf
from PIL import Image, ImageDraw
import skimage.filters as filters
from tqdm import tqdm_notebook


def findCranesImages(chips, classes):
    '''
    :param chips: a list of image filenames, whose size is (512, 512, 3)
    :param classes: xview image names.
    :return:
    '''
    desired_img_files = []
    # We only want to coordinates and classes that are within our chip
    for chip_name in tqdm_notebook(list(np.unique(chips))):
        _classes = classes[chips == chip_name].astype(np.int64)
        for c in [32, 54, 59]:
            if c in np.unique(_classes):
                desired_img_files.append(chip_name)
                print('keep image: {}'.format(chip_name))
    return desired_img_files

coords, chips, classes = wv.get_labels(jeojson_f)
desired_img_files=findCranesImages(coords, chips, classes)

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

    classes = list(classes)
    count = 0
    cropped_imgs = []

    for i, b in enumerate(boxes):
        xmin, ymin, xmax, ymax = [int(x) for x in b]
        cropped_img = img[ymin:ymax, xmin:xmax, :]
        print(cropped_img)
        cropped_source = Image.fromarray(cropped_img)
        label = classes[i]
        if label in desired_classes:
            count += 1
            cropped_source.save(output_dir+'img='+str(img_id)+'_label='+str(label)+'_count='+str((count))+".tif")

    print('Now saved {} bbox at: {}'.format(count, output_dir))

    #     cropped_imgs.append(cropped_source)
    #     for j in range(3):
    #         draw.rectangle(((xmin+j, ymin+j), (xmax+j, ymax+j)), outline="red")
    # return cropped_imgs


# register dataset
coco_dir = 'satrgb-cyclegan-10000-8/coco.json'
img_dir = 'satrgb-cyclegan-10000-8/images/'
coco = load_ann(coco_dir)

def crop_image(original_img_folder, file_name, bbox):
    ori_img_dir = original_img_folder + file_name
    my_image = cv2.imread(ori_img_dir)
    buffer = 0
    cropped_im = my_image[math.floor(bbox[1]) - buffer: math.floor(bbox[1]) + math.ceil(bbox[3]) + buffer,
                 math.floor(bbox[0]) - buffer: math.floor(bbox[0]) + math.ceil(bbox[2]) + buffer, :]
    return cropped_im


def get_bbox(image_dir, ann, n_instances=5000, save_bbox=False, output_dir=None):
    crop_ims = []
    for i in tqdm(range(n_instances)):
        instance = ann.iloc[i]
        file_name = instance['file_name']
        bbox = instance['bbox']
        category_id = instance['category_id']
        if category_id == 99:
            continue
        cropped_im = crop_image(image_dir, file_name, bbox)
        crop_ims.append(cropped_im)
        # print(category_id)
        # cv2_imshow(cropped_im)
        # save cropped images
        if save_bbox:
            output_name = str(i)+'_'+file_name[:-4]+'_'+str(category_id)
            # np.save(output_dir+output_name, cropped_im)
            Image.fromarray(cropped_im).save(output_dir+output_name+".jpeg")
    return crop_ims

bboxs = get_bbox(img_dir, coco, n_instances=5000, save_bbox=True, output_dir='clustering_sampled_data/bbox/syn/')