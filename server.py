# -*- coding: utf-8 -*-
import codecs
import cv2
import numpy as np
import sys
import os
import json
import sys
import threading
import collections
import socket
import time
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
from http.server import HTTPServer, BaseHTTPRequestHandler, ThreadingHTTPServer
from socketserver import ThreadingMixIn
from paddlers.tasks.utils.visualize import visualize_detection
import paddle 
from utils import *
sys.path.append('./PaddleRS')
from process import Detection
host = ('localhost', 8000)


class Resquest(BaseHTTPRequestHandler):
    def handler(self):
        print("data:", self.rfile.readline().decode())
        self.wfile.write(self.rfile.readline())
 
    def do_GET(self):
        print(self.requestline)
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        
    #接受post请求
    def do_POST(self):
        #读取数据
        req_datas = self.rfile.read(int(self.headers['content-length']))  
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
        paddle.device.cuda.empty_cache()
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(return_data).encode('utf-8'))

# 实现多线程  
class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True

server = ThreadedHTTPServer(host, Resquest)
try:
    server.serve_forever()
except KeyboardInterrupt:
    pass

# 直接使用线程
# sock = socket.socket (socket.AF_INET, socket.SOCK_STREAM)
# sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
# sock.bind(host)
# sock.listen(5)
# class Thread(threading.Thread):
#     def __init__(self, i):
#         threading.Thread.__init__(self)
#         self.i = i
#         self.daemon = True
#         self.start()
#     def run(self):
#         httpd = HTTPServer(host, Resquest, False)
#         httpd.socket = sock
#         httpd.server_bind = self.server_close = lambda self: None
#         httpd.serve_forever()
# [Thread(i) for i in range(100)]
# trusty_sleep(9e3)