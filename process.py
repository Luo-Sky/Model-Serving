# -*- coding: utf-8 -*-
import codecs
import cv2
import numpy as np
import sys
import paddlers as pdrs
import os
import json
import sys
from paddlers.tasks.utils.visualize import visualize_detection
import paddle 
from utils import *
sys.path.append('./PaddleRS')
# 服务端地址
host = ('localhost', 8000)
# 模型加载
# 地物提取
road_extraction_predictor = pdrs.deploy.Predictor(model_dir='./model/road_extraction_model', use_gpu=True, gpu_id=0)
water_extraction_predictor = pdrs.deploy.Predictor(model_dir='./model/water_extraction_model', use_gpu=True, gpu_id=0)
buildup_extraction_predictor = pdrs.deploy.Predictor(model_dir='./model/buildup_extraction_model', use_gpu=True, gpu_id=0)
# 变化检测
change_dectation_predictor = pdrs.deploy.Predictor(model_dir='./model/change_detection_model', use_gpu=True, gpu_id=0)
# 目标检测
object_detection_playground_predictor = pdrs.deploy.Predictor(model_dir='./model/object_detection_playground_model', use_gpu=True, gpu_id=0)
object_detection_overpass_predictor = pdrs.deploy.Predictor(model_dir='./model/object_detection_overpass_model', use_gpu=True, gpu_id=0)
object_detection_aircraft_predictor = pdrs.deploy.Predictor(model_dir='./model/object_detection_aircraft_model', use_gpu=True, gpu_id=0)
# 地物分类
segmentation_5_predictor = pdrs.deploy.Predictor(model_dir='./model/segmentation_gid5_model', use_gpu=True, gpu_id=0)
segmentation_15_predictor = pdrs.deploy.Predictor(model_dir='./model/segmentation_gid15_model', use_gpu=True, gpu_id=0)
print("load model finished!")

#检测
def Detection(msg, if_bbox):
    # flag1 = 'left' in msg and 'right' in msg and 'top' in msg and 'bottom' in msg
    # if flag1:
    #     flag2 = msg['left'] != '' and msg['right'] != '' and msg['top'] != '' and msg['bottom'] != ''

    # 裁剪
    if msg['type'] == 'change_detection':
        img_path = msg['img']
        img_path2 = msg['img2']
        img = cv2.imread(img_path)
        img2 = cv2.imread(img_path2)
        if if_bbox:
            img = img[msg['top']:msg['bottom'], msg['left']:msg['right']]
            img2 = img2[msg['top']:msg['bottom'], msg['left']:msg['right']]
    else:
        img_path = msg['img']
        img = cv2.imread(img_path)
        # print(img_path)
        if if_bbox:
          img = img[msg['top']:msg['bottom'], msg['left']:msg['right']]

    if msg['type'] == 'road_extraction':
        with paddle.no_grad():
            result = road_extraction_predictor.predict(img_file=img)
        # 置信度过滤
        pre_label = result['label_map']
        pre_score = result["score_map"]
        pre_score = np.max(pre_score, axis=2)
        pre_label[pre_score < road_extraction_confidence[msg['confidence']]] = 0
        # 最小像素块过滤
        pre_label = filter_bin_img(pre_label, msg['min_pixel'])
        pre_label[pre_label == 1] = 255
        # data信息
        data = []
        data.append({'category': 'positive', 'num': int(np.sum(pre_label==255)), 'color': [255,255,255]})
        data.append({'category': 'negative', 'num': int(np.sum(pre_label==0)), 'color': [0,0,0]})

    if msg['type'] == 'water_extraction':
        with paddle.no_grad():
            result = water_extraction_predictor.predict(img_file=img)
        # 置信度过滤
        pre_label = result['label_map']
        pre_score = result["score_map"]
        pre_score = np.max(pre_score, axis=2)
        pre_label[pre_score < water_extraction_confidence[msg['confidence']]] = 0
        # 最小像素块过滤
        pre_label = filter_bin_img(pre_label, msg['min_pixel'])
        pre_label[pre_label == 1] = 255
        # data信息
        data = []
        data.append({'category': 'positive', 'num': int(np.sum(pre_label==255)), 'color': [255,255,255]})
        data.append({'category': 'negative', 'num': int(np.sum(pre_label==0)), 'color': [0,0,0]})

    if msg['type'] == 'buildup_extraction':
        with paddle.no_grad():
            result = buildup_extraction_predictor.predict(img_file=img)
        # 置信度过滤
        pre_label = result['label_map']
        pre_score = result["score_map"]
        pre_score = np.max(pre_score, axis=2)
        pre_label[pre_score < buildup_extraction_confidence[msg['confidence']]] = 0
        # 最小像素块过滤
        pre_label = filter_bin_img(pre_label, msg['min_pixel'])
        pre_label[pre_label == 1] = 255
        # data信息
        data = []
        data.append({'category': 'positive', 'num': int(np.sum(pre_label==255)), 'color': [255,255,255]})
        data.append({'category': 'negative', 'num': int(np.sum(pre_label==0)), 'color': [0,0,0]})

    if msg['type'] == 'change_detection':
        with paddle.no_grad():
            result = change_dectation_predictor.predict((img, img2))
        # 置信度过滤
        pre_label = result['label_map']
        pre_score = result["score_map"]
        pre_score = np.max(pre_score, axis=2)
        pre_label[pre_score < change_detection_confidence[msg['confidence']]] = 0
        # 最小像素块过滤
        pre_label = filter_bin_img(pre_label, msg['min_pixel'])
        pre_label[pre_label==1] = 255
        # data信息
        data = []
        data.append({'category': 'positive', 'num': int(np.sum(pre_label==255)), 'color':[255,255,255]})
        data.append({'category': 'negative', 'num': int(np.sum(pre_label==0)), 'color':[0,0,0]})

    if msg['type'] == 'object_detection_playground':
        with paddle.no_grad():
            result = object_detection_playground_predictor.predict(img_file=img)
        # 置信度过滤
        best_result = []
        for item in result:
            if item['score'] > object_detection_confidence[msg['confidence']]:
                item['bbox'] = [int(p) for p in item['bbox']]
                best_result.append(item)

        # 转换坐标 TODO:是否需要？
        # _, _, w, h = object_detection_playground_predictor._model.fixed_input_shape
        # ori_w, ori_h, _ = img.shape
        # best_result = trans_bbox(best_result, ori_w/w, ori_h/h)
        return {'label': img, 'data': best_result}

    if msg['type'] == 'object_detection_overpass':
        with paddle.no_grad():
            result = object_detection_overpass_predictor.predict(img_file=img)
        # 置信度过滤
        best_result = []
        for item in result:
            if item['score'] > object_detection_confidence[msg['confidence']]:
                item['bbox'] = [int(p) for p in item['bbox']]
                best_result.append(item)

        # 转换坐标 TODO:是否需要？
        # _, _, w, h = object_detection_playground_predictor._model.fixed_input_shape
        # ori_w, ori_h, _ = img.shape
        # best_result = trans_bbox(best_result, ori_w/w, ori_h/h)
        return {'label': img, 'data': best_result}

    if msg['type'] == 'object_detection_aircraft':
        with paddle.no_grad():
            result = object_detection_aircraft_predictor.predict(img_file=img)
        # 置信度过滤
        best_result = []
        for item in result:
            if item['score'] > object_detection_confidence[msg['confidence']]:
                item['bbox'] = [int(p) for p in item['bbox']]
                best_result.append(item)

        # 转换坐标 TODO:是否需要？
        # _, _, w, h = object_detection_playground_predictor._model.fixed_input_shape
        # ori_w, ori_h, _ = img.shape
        # best_result = trans_bbox(best_result, ori_w/w, ori_h/h)
        return {'label': img, 'data': best_result}
    
    if msg['type'] == 'segmentation_5classes':
        with paddle.no_grad():
            result = segmentation_5_predictor.predict(img_file=img)
        # 置信度过滤
        pre_label = result['label_map']
        pre_score = result["score_map"]
        pre_score = np.max(pre_score, axis=2)
        pre_label[pre_score < segmentation_5_confidence[msg['confidence']]] = 5
        # 最小像素块过滤
        pre_label = filter_seg_img(pre_label, msg['min_pixel'], 5)
        # data信息
        data = []
        lut = get_gid5_rgb(0)
        for i in range(5):
            data.append({'category': gid5_classname[i], 'num': int(np.sum(pre_label==i)), 'color':[int(lut[i][0]),int(lut[i][1]),int(lut[i][2])]})
        
    if msg['type'] == 'segmentation_15classes':
        with paddle.no_grad():
            result = segmentation_15_predictor.predict(img_file=img)
        # 置信度过滤
        pre_label = result['label_map']
        pre_score = result["score_map"]
        pre_score = np.max(pre_score, axis=2)
        pre_label[pre_score < segmentation_15_confidence[msg['confidence']]] = 15
        # 最小像素块过滤
        pre_label = filter_seg_img(pre_label, msg['min_pixel'], 15)
        # data信息
        data = []
        lut = get_gid15_rgb(0)
        for i in range(15):
            data.append({'category': gid15_classname[i], 'num': int(np.sum(pre_label==i)), 'color':[int(lut[i][0]),int(lut[i][1]),int(lut[i][2])]})

    #判断图片是否有效
    if msg['type'] != 'object_dectation' and pre_label.shape != img.shape[:-1]:
        print('img broken!')
        return {"code": "503", "result":'null', 'msg': '处理异常'}

    del result
    paddle.device.cuda.empty_cache()

    return {'label': pre_label.astype(np.uint8), 'data': data}
