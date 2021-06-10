import wv_util as wv
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm


def findDesiredClassImages(coords, chips, classes, desired_classes=(32,54,59)):
    """
    A helper function to return a list of image filenames with desired objects.

    Args:
        coords: An (N, 4) array of bbox coordinates.
        chips: A list of image filenames.
        classes: A list of image labels.
        desired_classes: A list of desired labels.
    Output:
        A list pf image filenames with desired labels.
    """
    desired_img_files = []
    # We only want to coordinates and classes that are within our chip
    for chip_name in tqdm(list(np.unique(chips))):
        # _coords = coords[chips==chip_name]
        _classes = classes[chips==chip_name].astype(np.int64)
        for c in desired_classes:
            if c in np.unique(_classes):
                desired_img_files.append(chip_name)
                print('keep image: {}'.format(chip_name))
    return desired_img_files


def get_img_tiles(chip_name, coords, chips, classes, img_dir='', tile_size=(512,512)):
    """
    A helper function to chip original images into tiles.

    Args:
        chip_name: Image filename.
        coords: An (N, 4) array of bbox coordinates.
        chips: A list of image filenames.
        classes: A list of image labels.
        img_dir: The file location of the image/chip_name.
        tile_size: Desired tile size. Default by (512, 512)
    Output:
        By default tile_size = (512, 512)
        c_img: A collection of tiled chips in an (N, 512, 512, 3) array.
        c_box: A dictionary of bbox coordinates for each tiled chip. e.g. {tile_id: array([[506., 461., 512., 512.],[432., 415., 446., 422.]])}
        c_cls: A dictionary of tiled chip labels, where key is tile_id, and values are list of labels.
    """
    arr = wv.get_image(img_dir+chip_name)
    # We only want to coordinates and classes that are within our chip
    arr_coords = coords[chips==chip_name]
    arr_classes = classes[chips==chip_name].astype(np.int64)

    # We can chip the image into 512x512 chips
    c_img, c_box, c_cls = wv.chip_image(img=arr, coords=arr_coords, classes=arr_classes, shape=tile_size)
    # print("Num Chips: %d" % c_img.shape[0])
    return c_img, c_box, c_cls


def findDesiredClassTiles(c_img, c_box, c_cls, desired_classes=(32,54,59)):
    """
    A helper function to keep the tiled images with desired objects only.

    Args:
        By default tile_size = (512, 512)
        c_img: A collection of tiled chips in an (N, 512, 512, 3) array.
        c_box: A dictionary of bbox coordinates for each tiled chip. e.g. {tile_id: array([[506., 461., 512., 512.],[432., 415., 446., 422.]])}
        c_cls: A dictionary of tiled chip labels, where key is tile_id, and values are list of labels.
        desired_classes: A list of desired labels.
    Output:
        c_imgs_to_keep: A collection of tiled chips in an (N, 512, 512, 3) array.
        c_box_to_keep: A dictionary of tiled chips' bbox coordinates.
        c_cls_to_keep: A dictionary of tiled chips' labels.
    """
    idx_to_keep = []
    for i in c_cls:
        tmp = 0
        for cls in desired_classes:
            if cls in c_cls[i]:
                tmp+=1
        if tmp>0:
            idx_to_keep.append(i)
    c_imgs_to_keep = c_img[idx_to_keep]
    c_box_to_keep = [c_box[i] for i in idx_to_keep]
    c_cls_to_keep = [c_cls[i] for i in idx_to_keep]
    return c_imgs_to_keep, c_box_to_keep, c_cls_to_keep


def get_bboxes(img_id, img, boxes, classes, output_dir, desired_classes=(32,54,59)):
    """
    A helper function to draw bounding box rectangles on images

    Args:
        img: image to be drawn on in array format
        boxes: An (N,4) array of bounding boxes
        classes: array of labels

    Output:
        Save image with drawn bounding boxes to the output directory
    """
    source = Image.fromarray(img)
    draw = ImageDraw.Draw(source)
    w2, h2 = (img.shape[0], img.shape[1])

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
