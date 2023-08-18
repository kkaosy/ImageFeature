import cv2
import requests
import numpy as np
import pickle
import os
import base64

url = "http://localhost:8080/api/imagefeature"
def img2vec(img):
    v, buffer = cv2.imencode(".jpg", img)
    img_str = base64.b64encode(buffer)
    data = "image data,"+str.split(str(img_str),"'")[1]
    response = requests.get(url, json={"img":data})
    return response.json()

img = cv2.imread('D:/ImagFeature/train/Audi/1.jpg')
print(img2vec(img))