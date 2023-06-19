import numpy as np
import torch
import os
import sys
import argparse
sys.path.append(os.path.abspath('../yolov5'))
from utils.general import non_max_suppression, scale_coords
# from ai_core.object_detection.yolov5_custom.od.data.datasets import letterbox
from typing import List
# from dynaconf import settings
from models.experimental import attempt_load
import cv2
import glob
import json

class LicensePlateDet:
    def __init__(self, weights_path='.pt',size=(640,640),device='cpu',iou_thres=None,conf_thres=None):
        cwd = os.path.dirname(__file__)
        self.device=device
        self.char_model, self.names = self.load_model(weights_path)
        self.size=size
        
        self.iou_thres=iou_thres
        self.conf_thres=conf_thres

        self.plate_feature = []
        self.yellow_plate_count = 0


    def load_model(self,path):
        """ Load model to inference"""
        model = attempt_load(path, map_location=self.device)  # load FP32 model
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        model.eval()
        return model, names

    def detect(self, image):
        """ Interface Method for detect license plate 
        image:  BGR image for detection
        return: results:[["square license plate"(classname), confidence, rect(x,y,w,h)],]
                resized_img: bgr image.

        """
        results, resized_img = self.char_detection_yolo(image)

        return results, resized_img
    
   
    def char_detection_yolo(self, image):
        """ Implementation Method for detect 
            imgae: BGR image
        """

        img,resized_bgr_img = self.preprocess_image(image.copy())
        pred = self.char_model(img, augment=False)[0]
        
        detections = non_max_suppression(   pred, 
                                            conf_thres=self.conf_thres,
                                            iou_thres=self.iou_thres,
                                            classes=None,
                                            agnostic=True,
                                            multi_label=True,
                                            labels=(),
                                            max_det=2000)
        results=[]
        for i, det in enumerate(detections):
            det=det.tolist()
            if len(det):
                for *xywh, conf, cls in det:
                    if self.names[int(cls)] == "square license plate":
                        result=[self.names[int(cls)], str(conf), (xywh[0],xywh[1],xywh[2],xywh[3])]
                        results.append(result)
        return results, resized_bgr_img
        
    def preprocess_image(self, original_image):
        """ Change BGR image to:
            image: re1sized, RGB order, Channel first, norm ,add Batch for prediction
            resized_img: resized BGR image for visualization
        """
        resized_bgr_img = self.resize(original_image,size=self.size)
        image = resized_bgr_img.copy()[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB and Channel first
        image = np.ascontiguousarray(image)

        image = torch.from_numpy(image).to(self.device)
        image = image.float()
        image = image / 255.0
        if image.ndimension() == 3:
            image = image.unsqueeze(0)
        return image, resized_bgr_img


    def resize(self, img, size):
        """ Resize with padding """
        h, w = size
        height, width = img.shape[:2]

        aspect_ratio = width / height

        if aspect_ratio > 1:
            new_width = w
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = h
            new_width = int(new_height * aspect_ratio)

        resized_img = cv2.resize(img, (new_width, new_height))

        top = (h - new_height) // 2
        bottom = h - new_height - top
        left = (w - new_width) // 2
        right = w - new_width - left

        padded_resized_img = cv2.copyMakeBorder(resized_img, top, bottom, left, right, cv2.BORDER_CONSTANT)
        return padded_resized_img
    

    def extract_yellow_plate_image_feature(self, file_name, box, resized_img):
        """ Save plate image, Extract feature, Detect Yellow Plate
            file_name  : file_name for license plate image
            box        : list of x,y,w,h
            resized_img: BGR image 
            return     : Yellow plate or not.
        """

        # Save plate image
        license_cropped = resized_img[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
        cv2.imwrite(os.path.join('out', file_name), license_cropped)
        
        # Feature extraction
        license_cropped  = license_cropped/255.0
        bgr = np.sum(license_cropped, axis=(0, 1))
        b, g, r = bgr[0], bgr[1], bgr[2]
        norm_r, norm_g, norm_b = r/(r+g+b), g/(r+g+b), b/(r+g+b)
        self.plate_feature.append({"file": file_name, "r": norm_r, "g":norm_g, "b": norm_b})
        
        # yellow plate detection 
        # 1.5 is selected by checking plate_feature 
        if norm_r/norm_b > 1.5:
           self.yellow_plate_count += 1
           return True
        else:
           return False

    def save_plate_feature(self):
        feature = json.dumps(self.plate_feature)
        with open(os.path.join('out', 'plate_feature.json'), "w") as file:
            file.write(feature)

    def save_resized_image(self, img_name, results, resized_img):
        """save image to out dir"""
        for i, (name,conf,box) in enumerate(results):
            y = self.extract_yellow_plate_image_feature('plate_' + str(i) + '_' + os.path.basename(img_name), box, resized_img)
            if y == True:
                color = (0,255,255) #Yellow By BGR..
            else:
                color = (0,0,255)

            resized_img = cv2.rectangle(resized_img, 
                    (int(box[0]),int(box[1])), 
                    (int(box[2]),int(box[3])), color=color, thickness=3)

        if not os.path.exists(os.path.join('out')):
            os.makedirs(os.path.join('out'))

        print(os.path.join('out',os.path.basename(img_name)))
        cv2.imwrite(os.path.join('out',os.path.basename(img_name)),resized_img)

    def show_yellow_plate_count(self):
        print(self.yellow_plate_count)

def main():
    
    model=LicensePlateDet( size=[640,640],
                                weights_path='object.pt',
                                device='cpu',
                                iou_thres=0.5,
                                conf_thres=0.1)
    path='samples-confidential'

    img_names = glob.glob(path+"/*.jpg")
    for img_name in img_names:
        img_bgr =cv2.imread(img_name)
        results, resized_img=model.detect(img_bgr.copy())
        model.save_resized_image(img_name, results, resized_img)

    model.save_plate_feature()
    model.show_yellow_plate_count()

if __name__ == '__main__':
    main()
