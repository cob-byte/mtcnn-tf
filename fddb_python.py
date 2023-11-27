#coding:utf-8
import sys
import argparse
from training.mtcnn_model import P_Net, R_Net, O_Net
from tools.loader import TestLoader
from detection.detector import Detector
from detection.fcn_detector import FcnDetector
from detection.MtcnnDetector import MtcnnDetector
import cv2
import os

def get_imdb_fddb(data_dir):
    imdb = []
    nfold = 10
    for n in range(nfold):
        file_name = 'FDDB-folds/FDDB-fold-%02d.txt' % (n + 1)
        file_name = os.path.join(data_dir, file_name)
        fid = open(file_name, 'r')
        image_names = []
        for im_name in fid.readlines():
            image_names.append(im_name.strip('\n'))    
        imdb.append(image_names)
    return imdb        

def save_detections_per_image(detector, image_info, output_dir, fold_num):
    for image_name in image_info:
        img = cv2.imread(os.path.join(data_dir, 'originalPics', image_name + '.jpg'))
        all_boxes, _ = detector.detect_one_image(img)

        # Construct the file name without the extension
        file_name = image_name.replace('/', '_')

        fold_output_dir = os.path.join(output_dir, f'{fold_num}')
        os.makedirs(fold_output_dir, exist_ok=True)

        dets_file_name = os.path.join(fold_output_dir, f'{file_name}.txt')
        with open(dets_file_name, 'w') as fid:
            boxes = all_boxes[0]
            if boxes is None:
                fid.write(file_name + '\n')
                fid.write(str(1) + '\n')
                fid.write('%f %f %f %f %f\n' % (0, 0, 0, 0, 0.99))
                continue
            fid.write(file_name + '\n')
            fid.write(str(len(boxes)) + '\n')
            for box in boxes:
                fid.write('%f %f %f %f %f\n' % (
                    float(box[0]), float(box[1]), float(box[2] - box[0] + 1), float(box[3] - box[1] + 1), box[4]))

if __name__ == "__main__":
    data_dir = './fddb/'
    out_dir = './fddb/fddb-output-python/'

    test_mode = "ONet"
    thresh = [0.1, 0.15, 0.5]
    min_face_size = 20
    stride = 2
    detectors = [None, None, None]
    prefix = ['./tmp/model/pnet/pnet', './tmp/model/rnet/rnet', './tmp/model/onet/onet']
    epoch = [30, 30, 30]
    batch_size = [2048, 256, 16]
    model_path = ['%s-%s' % (x, y) for x, y in zip(prefix, epoch)]
    
    # load pnet model
    PNet = FcnDetector(P_Net, model_path[0])
    detectors[0] = PNet
    
    # load rnet model
    if test_mode in ["RNet", "ONet"]:
        RNet = Detector(R_Net, 24, batch_size[1], model_path[1])
        detectors[1] = RNet
    
    # load onet model
    if test_mode == "ONet":
        ONet = Detector(O_Net, 48, batch_size[2], model_path[2])
        detectors[2] = ONet
    
    mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=min_face_size,
                                   stride=stride, threshold=thresh)
    
    for fold_num, fold_images in enumerate(get_imdb_fddb(data_dir), start=1):
        save_detections_per_image(mtcnn_detector, fold_images, out_dir, fold_num)