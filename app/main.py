import cv2
import numpy as np
import base64
from app.code import getHog
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

def readB64(url):
    encodeED_data = url.split(',',1)
    img_str = encodeED_data[1]
    decode = base64.b64decode(img_str)
    img = cv2.imdecode(np.frombuffer(decode, np.uint8),cv2.IMREAD_GRAYSCALE)
    return img

@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI!"}

@app.get("/api/getHog")
async def read_str(data: Request):
    json = await data.json()
    item_str = json["img"]
    img = readB64(item_str)
    hog = getHog(img)
    return {"message":hog}
