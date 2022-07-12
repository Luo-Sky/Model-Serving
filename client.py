import requests
import json
import numpy as np
import cv2
import base64
import aiohttp
import asyncio
import os
import time
# datas_example = {'type': 'road_extraction',
#         'confidence': 'high', # low, medium, high
#         'min_pixel': 50, #
#         'img': img_str,
#         'img2': img_str2,
#         'left': 10,
#         'right': 10,
#         'bottom': 10,
#         'top': 10
#         }

# 道路提取
# for dirname, dirnames, filenames in os.walk('./data/road_extraction/'):
#     if len(filenames) == 0:
#         continue
#     for filename in filenames:
#         img_str = dirname + '/' + filename
#         datas = {'type': 'road_extraction', 'confidence': 'high', 'min_pixel': 50, 'img': img_str, 'left': -1, 'right': -1, 'bottom': -1, 'top': -1}
#         req = json.dumps(datas)
#         r = requests.post("http://127.0.0.1:8000", data = req)
#         print(r.text)

# 复杂的post请求/道路提取
async def main(name):
    pyload = {'type': 'road_extraction', 'confidence': 'high', 'min_pixel': 50, 'img': name, 'left': -1, 'right': -1, 'bottom': -1, 'top': -1}
    async with aiohttp.ClientSession() as sess:
        async with sess.post('http://127.0.0.1:8000', json=pyload) as resp:
            print(resp.status)
            print(await resp.text())

if __name__ == '__main__':
    tasks = []
    loop = asyncio.get_event_loop()
    for dirname, dirnames, filenames in os.walk('./data/road_extraction/'):
        if len(filenames) == 0:
            continue
        for filename in filenames:
            img_str = dirname + '/' + filename
            task = asyncio.ensure_future(main(img_str))
            tasks.append(task)
    loop.run_until_complete(asyncio.gather(*tasks))

# 水体提取
# img_str = './data/water_extraction/GF2_PMS1__L1A0000564539-MSS1_1_4_img.tif'
# datas = {'type': 'water_extraction', 'confidence': 'low', 'min_pixel': 50, 'img': img_str}
# req = json.dumps(datas)
# r = requests.post("http://127.0.0.1:8000", data = req)
# print(r.text)

# 建筑物提取
# img_str = './data/buildup_extraction/GF2_PMS1__L1A0000564539-MSS1_1_7_img.tif'
# datas = {'type': 'buildup_extraction', 'confidence': 'medium', 'min_pixel': 50, 'img': img_str}
# req = json.dumps(datas)
# r = requests.post("http://127.0.0.1:8000", data = req)
# print(r.text)

# 变化检测
# img_str = '/media/dell/DATA/wy/2022-cup/RS-serving/data/change_detection/test_29_A.png'
# img_str2 = '/media/dell/DATA/wy/2022-cup/RS-serving/data/change_detection/test_29_B.png'
# datas = {'type': 'change_detection', 'confidence': 'high', 'min_pixel': 1000, 'img': img_str, 'img2': img_str2, 'left': 100, 'right': 800, 'bottom': 100, 'top': 800}
# req = json.dumps(datas)
# r = requests.post("http://127.0.0.1:8000", data = req)
# print(r.text)

# 操场目标检测
# img_str = './data/object_detection_playground/playground_12.jpg'
# datas = {'type': 'object_detection_playground', 'confidence': 'high', 'img': img_str}
# req = json.dumps(datas)
# r = requests.post("http://127.0.0.1:8000", data = req)
# print(r.text)

# 飞机目标检测
# img_str = '/media/dell/DATA/wy/2022-cup/RS-serving/data/object_detection_aircraft/test.png'
# datas = {'type': 'object_detection_aircraft', 'confidence': 'low', 'img': img_str}
# req = json.dumps(datas)
# r = requests.post("http://127.0.0.1:8000", data = req)
# print(r.text)

# 立交桥目标检测
# img_str = './data/object_detection_overpass/overpass_11.jpg'
# datas = {'type': 'object_detection_overpass', 'confidence': 'low', 'img': img_str}
# req = json.dumps(datas)
# r = requests.post("http://127.0.0.1:8000", data = req)
# print(r.text)

# 地物分类
# img_str = '/media/dell/DATA/wy/2022-cup/RS-serving/data/change_detection/test_29_A.png'
# datas = {'type': 'segmentation_5classes', 'min_pixel': 100, 'confidence': 'medium', 'img': img_str, 'left':, 'right': , 'bottom': 100, 'top': 400}
# req = json.dumps(datas)
# r = requests.post("http://127.0.0.1:8000", data = req)
# print(r.text)