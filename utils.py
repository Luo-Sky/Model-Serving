import numpy as np
from skimage.morphology import remove_small_objects
import cv2
ip = "http://faye.nat300.top"
road_extraction_confidence = {
    "high": 0.6,
    "medium":0.5,
    "low": 0.4
}
water_extraction_confidence = {
    "high": 0.6,
    "medium":0.5,
    "low": 0.4
}
buildup_extraction_confidence = {
    "high": 0.6,
    "medium":0.5,
    "low": 0.4
}
#
change_detection_confidence = {
    "high": 0.6,
    "medium":0.5,
    "low": 0.4
}

object_detection_confidence = {
    "high": 0.4,
    "medium":0.3,
    "low": 0.2
}

segmentation_5_confidence = {
    "high": 0.6,
    "medium":0.5,
    "low": 0.4
}

segmentation_15_confidence = {
    "high": 0.5,
    "medium":0.4,
    "low": 0.3
}

# BGR
change_detection_color = [0,0,255]
water_extraction_color = [255,0,0]
buildup_extraction_color = [0,0,255]
road_extraction_color = [255,255,255]

gid5_classname = ['buildup', 'farmland', 'forest', 'meadow', 'water']
gid5_classname_CN = ['建筑', '田地', '林地', '草地', '水体']
gid15_classname = ['industrial_land', 'urban_residential', 'rural_residential',
                    'traffic_land', 'paddy_field', 'irrigated_land', 'dry_cropland',
                    'garden_plot', 'arbor_woodland', 'shrub_land', 'natural_grassland',
                    'artificial_grassland', 'river', 'lake', 'pond']
gid15_classname_CN = ['工业用地','城市住宅','农村住宅',
                      '交通用地','水田','灌溉地','旱地',
                      '花园小区','乔木林地','灌木地','天然草地',
                      '人工草地','河流','湖泊','池塘']
def get_gid5_rgb(shape):
    lut = np.zeros((512,3), dtype=np.uint8)
    lut[0] = [255,0,0]
    lut[1] = [0,255,0]
    lut[2] = [0,255,255]
    lut[3] = [255,255,0]
    lut[4] = [0,0,255]
    lut[5] = [0,0,0]
    return lut

def get_gid15_rgb(shape):
    lut = np.zeros((512,3), dtype=np.uint8)
    lut[0] = [200,0,0]
    lut[1] = [250,0,150]
    lut[2] = [200,150,150]
    lut[3] = [250,150,150]
    lut[4] = [0,200,0]
    lut[5] = [150,250,0]
    lut[6] = [150,200,150]
    lut[7] = [200,0,200]
    lut[8] = [150,0,250]
    lut[9] = [150,150,250]
    lut[10] = [250,200,0]
    lut[11] = [200,200,0]
    lut[12] = [0,0,200]
    lut[13] = [0,150,200]
    lut[14] = [0,200,250]
    lut[15] = [0,0,0]
    return lut

def trans_bbox(bboxes, w_ratio, h_ratio):
    for i in range(len(bboxes)):
        bboxes[i]["bbox"][0] = bboxes[i]["bbox"][0] * w_ratio
        bboxes[i]["bbox"][1] = bboxes[i]["bbox"][1] * h_ratio
        bboxes[i]["bbox"][2] = bboxes[i]["bbox"][2] * w_ratio
        bboxes[i]["bbox"][3] = bboxes[i]["bbox"][3] * h_ratio
    return bboxes

def filter_bin_img(label, pixels):
    new_label = np.zeros_like(label)
    arr = label > 0
    new_label[remove_small_objects(arr, connectivity=1, min_size=pixels)] = 1
    return new_label

def filter_seg_img(label, pixels, class_num):
    new_label = np.ones(label.shape) * class_num
    for i in range(class_num):
        bin_label = np.zeros(label.shape)
        bin_label[label == i] = 1
        arr = bin_label > 0
        new_label[remove_small_objects(arr, connectivity=1, min_size=pixels)] = i
    return new_label

def bin2color(label, type):
    if type == "change_detection":
        label[np.all(label == (int(255), int(255), int(255)), axis=-1)] = (int(change_detection_color[0]), int(change_detection_color[1]), int(change_detection_color[2]))
    elif type == "water_extraction":
        label[np.all(label == (int(255), int(255), int(255)), axis=-1)] = (int(water_extraction_color[0]), int(water_extraction_color[1]), int(water_extraction_color[2]))
    elif type == "buildup_extraction":
        label[np.all(label == (int(255), int(255), int(255)), axis=-1)] = (int(buildup_extraction_color[0]), int(buildup_extraction_color[1]), int(buildup_extraction_color[2]))
    elif type == "road_extraction":
        label[np.all(label == (int(255), int(255), int(255)), axis=-1)] = (int(road_extraction_color[0]), int(road_extraction_color[1]), int(road_extraction_color[2]))
    return label

def transparent(label):
    tmp = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
    _, alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY)
    b, g, r = cv2.split(label)
    rgba = [b, g, r, alpha]
    dst = cv2.merge(rgba, 4)
    return dst
