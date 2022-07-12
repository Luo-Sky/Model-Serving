# 黑色变透明
import cv2
crop_image = cv2.imread('/media/dell/DATA/wy/2022-cup/RS-serving/data/road_extraction/10378780_15_predict.jpg')
tmp = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
_, alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY)
b, g, r = cv2.split(crop_image)
rgba = [b, g, r, alpha]
dst = cv2.merge(rgba, 4)
cv2.imwrite("/media/dell/DATA/wy/2022-cup/RS-serving/data/road_extraction/test.png", dst)