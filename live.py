import cv2
import numpy as np
import torch
from torchvision import transforms
from modules.dataloaders.utils import decode_segmap
from modules.models.deeplab_xception import DeepLabv3_plus
from modules.models.sync_batchnorm.replicate import patch_replication_callback
from PIL import Image
import os.path as osp

MODEL_PATH = "./run/surface/deeplab/model_iou_77.pth.tar"
ORIGINAL_HEIGHT = 480
ORIGINAL_WIDTH = 640
MODEL_HEIGHT = 480 #512
MODEL_WIDTH =  640 #1024
NUM_CLASSES = 7  # including background
CUDA = True if torch.cuda.is_available() else False

MODE = 'jpg'  # 'mp4' or 'jpg'

OVERLAPPING = True  # whether to mix segmentation map and original image

CUSTOM_COLOR_MAP = [
    [0, 0, 0],  # background
    [255, 128, 0],  # bike_lane
    [255, 0, 0],  # caution_zone
    [255, 0, 255],  # crosswalk
    [255, 255, 0],  # guide_block
    [0, 0, 255],  # roadway
    [0, 255, 0],
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

        cv2.imshow('image', result)
            
        key = cv2.waitKey(1)
        if key == 27:
            break