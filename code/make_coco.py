'''
Python script to convert annotations and metadata from satrgb into a coco file
'''

# Import Python libraries
import os
import json
import yaml
import cv2
from shapely.geometry import Polygon, MultiPolygon
import numpy as np
from tqdm import tqdm_notebook

# Data Directories and files
anndir = 'annotations/'
mtddir = 'metadata/'
imgdir = 'images/'
imgtype = 'png'
ctgfile = 'categories/type_to_category.yml'
ctgrsfile = 'categories/categories.yml'

# Create coco file
typ = {"type": "instances"}

# Get the categories
print("Getting categories...")
with open(ctgfile) as file:
    ctg = yaml.load(file, Loader=yaml.FullLoader)

# Create a list of all the categories in all the metadata files
cat = []
for filename in tqdm_notebook(sorted(os.listdir(mtddir))):
    with open(mtddir + filename) as json_file:
        mtdcc = json.load(json_file)
    for obj in mtdcc['objects']:
        cat.append(obj['type'])
cat = set(cat)
cat = list(cat)

with open(ctgrsfile, "r") as file:
    ctgrs = yaml.load(file, Loader=yaml.FullLoader)["categories"]

# Create the annotations and images
anntns = []
imgs = []

print("Creating coco annotation file...")
annid = 0
for ifile, filename in tqdm_notebook(enumerate(sorted(os.listdir(anndir)))):
    with open(anndir + filename) as json_file:
        anncc = json.load(json_file)

    imgname = anncc['filename']
    imgname = imgname.split('.')[0]
    img = cv2.imread(imgdir + imgname + '.' + imgtype,0)
    width = img.shape[1]
    height = img.shape[0]
    imgann = {"id": ifile, "file_name": anncc['filename'], "width": width, "height": height}
    imgs.append(imgann)

    with open(mtddir + filename.replace("-ana.json", "-metadata.json")) as json_file:
        mtdcc = json.load(json_file)
    iddict = {}
    for obj in mtdcc['objects']:
        iddict[obj['id']] = obj['type']

    for iann, ann in enumerate(anncc['annotations']):
        anntn = {}
        seg_list = ann['segmentation']
        anntn['segmentation'] = []
        polygons = []
        for seg in seg_list:
            pts = np.array(seg).reshape(int(len(seg)/2),2)
            coords = np.vstack((pts, pts[0,:]))
            sgmtn = coords.flatten()
            sgmtn = sgmtn.tolist()
            if (len(sgmtn) == 4) or (len(sgmtn) == 6):
                continue # won't include single point or line in annotations
            polygons.append(Polygon(coords))
            anntn['segmentation'].append(sgmtn)
        if len(polygons) == 0:
            continue
        anntn['iscrowd'] = 0
        anntn['image_id'] = ifile
        objtype = mtdcc['objects'][iann]['type']
        anntn['category_id'] = ctg[iddict[ann['id']]]
        anntn['id'] = annid
        annid = annid + 1
        anntn['bbox'] = ann['bbox']
        # Going to calculate the area of the segmentation bounding box
        multi_poly = MultiPolygon(polygons)
        anntn['area'] = int(multi_poly.area)
        anntns.append(anntn)

# Output json to coco file
categories = {"categories": ctgrs}
annotations = {"annotations": anntns}
images = {"images": imgs}
coco = {**typ, **categories, **annotations, **images}
with open('coco.json', 'w') as f:
    json.dump(coco, f)