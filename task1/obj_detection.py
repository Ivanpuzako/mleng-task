import torch
import torchvision
import argparse
from torchvision import transforms
import numpy as np
from PIL import Image, ImageDraw
import cv2 as cv
from cats import cats
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

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
