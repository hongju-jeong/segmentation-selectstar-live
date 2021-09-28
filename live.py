import cv2
import numpy as np
import torch
from torchvision import transforms
from modules.dataloaders.utils import decode_segmap
from modules.models.deeplab_xception import DeepLabv3_plus
from modules.models.sync_batchnorm.replicate import patch_replication_callback
from PIL import Image
import os.path as osp

import aigo_destination_route
import threading
#from ublox_gps import UbloxGps
import serial
import math


#aigo_destination_route.get_route("경희대학교 중앙도서관")



#임시 linestring 향후 linestring.txt 파싱 후
linestring = [[127.08341510840815, 37.240436069188185] , #출발
                [127.08341233624519, 37.24024442466403 ], #도로 인식 후 보행자도로 gps로 교체
                [127.08212911848908, 37.24027217627202 ],
                [127.08205412528905, 37.240272174930624 ],
                [127.08013207664933, 37.24029713765783 ],
                [127.07963767687961, 37.24030268372765 ],
                [127.07956268360199, 37.24030545984253 ],
                [127.07943769454693, 37.240319344887844 ],
                [127.07924326470169, 37.240427662200275 ]]

global current_point
current_point = 0

def gpsTracking():
    port = serial.Serial('/dev/ttyACM0', baudrate=460800, timeout=1)
    gps = UbloxGps(port)
    
    global g_lon                #다른 함수에서 받아올 현재 lon, lat
    global g_lat
    global current_point
    global point_check
    point_check = 0
    with open('points.txt', 'r') as points:
        currentline = points.readline()
        splitline = currentline.split(',')
        while True:
            try: 
                coords = gps.geo_coords()
                print(coords.lon, coords.lat)
                current_location_lon = coords.lon
                current_location_lat = coords.lat
                g_lon = current_location_lon
                g_lat = current_location_lat
            except (ValueError, IOError) as err:
                print(err) 

            if(point_check == 1):
                currentline = points.readline()
                if(currentline == ''):
                    break
                splitline = currentline.split(',')
                point_check = 0

            point_lon = float(splitline[0])
            point_lat = float(splitline[1])
            route_info = splitline[2]

            PTCdistance = math.sqrt((point_lon-current_location_lon)**2 + (point_lat-current_location_lat)**2)
            if (PTCdistance<=0.00001):
                print(route_info)
                point_check = 1
                current_point+=1
            
    port.close()

# t1 = threading.Thread(target=gpsTracking)
# t1.start()



MODEL_PATH = "./run/surface/deeplab/model_best.pth.tar"
ORIGINAL_HEIGHT = 480
ORIGINAL_WIDTH = 640
MODEL_HEIGHT = 480 #512
MODEL_WIDTH =  640 #1024
NUM_CLASSES = 4  # including background
CUDA = True if torch.cuda.is_available() else False

MODE = 'jpg'  # 'mp4' or 'jpg'
OVERLAPPING = True  # whether to mix segmentation map and original image

CUSTOM_COLOR_MAP = [
    [0, 0, 0],  # background
    [255, 0, 255],  # crosswalk
    [255, 0, 0],  # roadway
    [0, 255, 0],  # sidewalk
]  # To ignore unused classes while predicting
CUSTOM_N_CLASSES = len(CUSTOM_COLOR_MAP)

class ModelWrapper:
    def __init__(self):
        self.composed_transform = transforms.Compose([
            transforms.Resize((MODEL_HEIGHT, MODEL_WIDTH), interpolation=Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

        self.model = self.load_model(MODEL_PATH)

    @staticmethod
    def load_model(model_path):
        model = DeepLabv3_plus(nInputChannels=3, n_classes=NUM_CLASSES, os=16)
        if CUDA:
            model = torch.nn.DataParallel(model, device_ids=[0])
            patch_replication_callback(model)
            model = model.cuda()
        if not osp.isfile(MODEL_PATH):
            raise RuntimeError("=> no checkpoint found at '{}'".format(model_path))
        checkpoint = torch.load(model_path)   #, map_location=torch.device('cpu')
        if CUDA:
            model.module.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch: {}, best_pred: {})"
              .format(model_path, checkpoint['epoch'], checkpoint['best_pred']))
        model.eval()
        return model

    def predict(self, rgb_img: np.array):
        x = self.composed_transform(Image.fromarray(rgb_img))
        x = x.unsqueeze(0)

        if CUDA:
            x = x.cuda()
        with torch.no_grad():
            output = self.model(x)
        pred = output.data.detach().cpu().numpy()
        pred = np.argmax(pred, axis=1).squeeze(0)
        segmap = decode_segmap(pred, dataset='custom', label_colors=CUSTOM_COLOR_MAP, n_classes=CUSTOM_N_CLASSES)
        segmap = np.array(segmap * 255).astype(np.uint8)

        resized = cv2.resize(segmap, (ORIGINAL_WIDTH, ORIGINAL_HEIGHT),
                             interpolation=cv2.INTER_NEAREST)
        return resized

model_wrapper = ModelWrapper()

capture = cv2.VideoCapture(0)


idx = np.array([320])

road_way_count = 0 

g_lon, g_lat = 127.08341510840815, 37.240436069188185      # 도로를 인식한 위치라 가정
current_point = 1
while True:
        ret, frame = capture.read()

        if ret == False:
            continue
        
        segmap = model_wrapper.predict(frame)
        
        if OVERLAPPING:
            h, w, _ = np.array(segmap).shape
            img_resized = cv2.resize(frame, (w, h))
            result = (img_resized * 0.5 + segmap * 0.5).astype(np.uint8)
        else:
            result = segmap

        # boundary = np.where((result[:,:,0] == 255)&(result[:,:,1] == 0)&(result[:,:,2] == 0))
        # print(boundary)
        # print(result.shape)

        detect_array = result[200:265,320:330]
        boolean_indexing = (detect_array[:,:,0] == 255)&(detect_array[:,:,1] == 0)&(detect_array[:,:,2] == 0)
        boolean_set = np.unique(boolean_indexing)
        if(boolean_set.size == 1 & boolean_set[0] == True):
            road_way_count+=1

        print(road_way_count)

        if(road_way_count == 100):
            imu = "South"                       # mqtt로 받아옴  , current_point = 1
            if(imu == "South"):
                direction = g_lon - linestring[current_point+1][0]
                if(direction > 0):
                    next_point_lon = g_lon - 0.00005      # green확인 후 이동
                    next_point_lat = g_lat
                    linestring.insert(current_point+1,[next_point_lon,next_point_lat])
                    print(linestring)




        cv2.imshow('image', result[200:265,320:330])
        cv2.imshow('image2', result)
        key = cv2.waitKey(1)
        if key == 27:
            break