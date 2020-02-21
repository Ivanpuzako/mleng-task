import torch
import torchvision
import argparse
import matplotlib.pyplot as plt
from torchvision import transforms
import pycocotools##cython
import numpy as np
from PIL import Image, ImageDraw
import cv2 as cv
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor




cats = {1: 'person',
 2: 'bicycle',
 3: 'car',
 4: 'motorcycle',
 5: 'airplane',
 6: 'bus',
 7: 'train',
 8: 'truck',
 9: 'boat',
 10: 'traffic light',
 11: 'fire hydrant',
 12: 'street sign',
 13: 'stop sign',
 14: 'parking meter',
 15: 'bench',
 16: 'bird',
 17: 'cat',
 18: 'dog',
 19: 'horse',
 20: 'sheep',
 21: 'cow',
 22: 'elephant',
 23: 'bear',
 24: 'zebra',
 25: 'giraffe',
 26: 'hat',
 27: 'backpack',
 28: 'umbrella',
 29: 'shoe',
 30: 'eye glasses',
 31: 'handbag',
 32: 'tie',
 33: 'suitcase',
 34: 'frisbee',
 35: 'skis',
 36: 'snowboard',
 37: 'sports ball',
 38: 'kite',
 39: 'baseball bat',
 40: 'baseball glove',
 41: 'skateboard',
 42: 'surfboard',
 43: 'tennis racket',
 44: 'bottle',
 45: 'plate',
 46: 'wine glass',
 47: 'cup',
 48: 'fork',
 49: 'knife',
 50: 'spoon',
 51: 'bowl',
 52: 'banana',
 53: 'apple',
 54: 'sandwich',
 55: 'orange',
 56: 'broccoli',
 57: 'carrot',
 58: 'hot dog',
 59: 'pizza',
 60: 'donut',
 61: 'cake',
 62: 'chair',
 63: 'couch',
 64: 'potted plant',
 65: 'bed',
 66: 'mirror',
 67: 'dining table',
 68: 'window',
 69: 'desk',
 70: 'toilet',
 71: 'door',
 72: 'tv',
 73: 'laptop',
 74: 'mouse',
 75: 'remote',
 76: 'keyboard',
 77: 'cell phone',
 78: 'microwave',
 79: 'oven',
 80: 'toaster',
 81: 'sink',
 82: 'refrigerator',
 83: 'blender',
 84: 'book',
 85: 'clock',
 86: 'vase',
 87: 'scissors',
 88: 'teddy bear',
 89: 'hair drier',
 90: 'toothbrush',
 91: 'hair brush'}
parser = argparse.ArgumentParser(description='Add path to photo')
parser.add_argument('source')
parser.add_argument('dest')
args = parser.parse_args()
img = args.source
dest = args.dest
Img = Image.open(img).convert('RGB')
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
pretrained =True
num_classes = 81
in_features = model.roi_heads.box_predictor.cls_score.in_features

transform = torchvision.transforms.ToTensor()
transformed = transform(Img).unsqueeze(0)
model.eval()
result = model(transformed)
def show_result(img, result, threshold = 0.7):
    image = cv.imread(img)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    result = result[0]
    boxes = result['boxes']
    labels = result['labels']
    scores = result['scores']
    for box, label, score in zip(boxes, labels, scores):
        if score < threshold:
            continue
        else:
            box = box.detach().numpy()
            cv.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0,255,0), 4)
            cv.putText(image, cats[int(label)], (int(box[0]), int(box[1])-10), cv.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 3)
    return image
output_image = cv.cvtColor(show_result(img, result), cv.COLOR_RGB2BGR)   

cv.imwrite(dest, output_image)
