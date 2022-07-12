from concurrent.futures import thread
from flask import Flask, request
import codecs
import cv2
import numpy as np
import sys
import os
import json
import sys
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
from paddlers.tasks.utils.visualize import visualize_detection
import paddle 
from utils import *
sys.path.append('./PaddleRS')
from process import Detection

app = Flask(__name__)

@app.route('/', methods=['POST'])
def process_json():
    content_type = request.headers.get('Content-Type')
    req_datas = request.get_data()
    req = json.loads(req_datas.decode())

    # for index in req:
    #     req[index] =req[index][0]
    if "confidence" not in req:
        req["confidence"] = 'medium'
    if "min_pixel" not in req:
        req["min_pixel"] = 50
    else:
        req["min_pixel"] = int(req["min_pixel"])
    # print(req)
    if_bbox_0 = "left" in req and "right" in req and "bottom" in req and "top" in req
    if if_bbox_0:
        if_bbox_1 = req['left'] != -1 and req['right'] != -1 and req['bottom'] != -1 and req['top'] != -1
    if_bbox = if_bbox_0 and if_bbox_1
    
    if if_bbox:
        req["left"] = int(req["left"])
        req["right"] = int(req["right"])
        req["bottom"] = int(req["bottom"])
        req["top"] = int(req["top"])
    #检测
    with paddle.no_grad():
        result = Detection(req, if_bbox)
    label = result['label']
    data = result['data']

    # 语义分割转换label为RGB
    if req['type'] == 'segmentation_5classes': 
        lut = get_gid5_rgb(label.shape[0])
        label = lut[label]
        label = cv2.cvtColor(label, cv2.COLOR_RGB2BGR)
    if req['type'] == 'segmentation_15classes':
        lut = get_gid15_rgb(label.shape[0])
        label = lut[label]
        label = cv2.cvtColor(label, cv2.COLOR_RGB2BGR)

    # 目标检测画框
    if 'object_detection' in req['type']:
        if len(data) != 0:
            label = visualize_detection(
                np.array(label), data, 
                color=np.asarray([[0,255,0]], dtype=np.uint8), 
                threshold=object_detection_confidence[req['confidence']], save_dir=None
            )

    # 保存预测标签
    if len(label.shape) == 2:
        label = np.expand_dims(label, axis=2)
        label = np.concatenate((label, label, label), axis=-1)

    bin2color(label, req['type'])
    label = transparent(label)
    
    if if_bbox:
        new_name = os.path.splitext(req['img'])[0] + '_predict_{}_{}_{}_{}.png'.format(str(req['left']), str(req['top']), str(req['bottom']), str(req['top']))
    else:
        new_name = os.path.splitext(req['img'])[0] + '_predict.png'
    cv2.imwrite(new_name, label)

    # 传回ip地址
    download_name = ip + "/files/" + new_name.split('/')[-1].split('.')[0]
    if if_bbox:
        return_data = {"type": req['type'],
                    "label": download_name,
                    "data": data,
                    "confidence": req['confidence'], 
                    "min_pixel": req['min_pixel'],
                    "left": req['left'],
                    "right": req['right'],
                    "bottom": req['bottom'],
                    "top": req['top']}
    else:
        return_data = {"type": req['type'],
                    "label": download_name,
                    "data": data,
                    "confidence": req['confidence'], 
                    "min_pixel": req['min_pixel']}

    #返回结果
    del result
    return return_data

if __name__ == "__main__":
    app.run('localhost', 8000, threaded=True)