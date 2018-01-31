# This script is used to demonstrate MobileNet-SSD network using OpenCV deep learning module.
#
# It works with model taken from https://github.com/chuanqi305/MobileNet-SSD/ that
# was trained in Caffe-SSD framework, https://github.com/weiliu89/caffe/tree/ssd.
# Model detects objects from 20 classes.
#
# Also TensorFlow model from TensorFlow object detection model zoo may be used to
# detect objects from 90 classes:
# http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_11_06_2017.tar.gz
# Text graph definition must be taken from opencv_extra:
# https://github.com/opencv/opencv_extra/tree/master/testdata/dnn/ssd_mobilenet_v1_coco.pbtxt
import numpy as np
import argparse
from collections import Counter

try:
    import cv2 as cv
except ImportError:
    raise ImportError('Can\'t find OpenCV Python module. If you\'ve built it from sources without installation, '
                      'configure environemnt variable PYTHONPATH to "opencv_build_dir/lib" directory (with "python3" subdirectory if required)')

inWidth = 300
inHeight = 300
WHRatio = inWidth / float(inHeight)
inScaleFactor = 0.007843
meanVal = 127.5

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Script to run MobileNet-SSD object detection network '
                    'trained either in Caffe or TensorFlow frameworks.')
    parser.add_argument("--video", help="path to video file. If empty, camera's stream will be used")
    parser.add_argument("--image", help="path to image file.")
    parser.add_argument("--prototxt", default="MobileNetSSD_deploy.prototxt.txt",
                                      help='Path to text network file: '
                                           'MobileNetSSD_deploy.prototxt for Caffe model or '
                                           'ssd_mobilenet_v1_coco.pbtxt from opencv_extra for TensorFlow model')
    parser.add_argument("--weights", default="MobileNetSSD_deploy.caffemodel",
                                     help='Path to weights: '
                                          'MobileNetSSD_deploy.caffemodel for Caffe model or '
                                          'frozen_inference_graph.pb from TensorFlow.')
    parser.add_argument("--num_classes", default=20, type=int,
                        help="Number of classes. It's 20 for Caffe model from "
                             "https://github.com/chuanqi305/MobileNet-SSD/ and 90 for "
                             "TensorFlow model from http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_11_06_2017.tar.gz")
    parser.add_argument("--thr", default=0.2, type=float, help="confidence threshold to filter out weak detections")
    args = parser.parse_args()

    if args.num_classes == 20:
        net = cv.dnn.readNetFromCaffe(args.prototxt, args.weights)
        swapRB = False
        classNames = { 0: 'background',
            1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
            5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
            10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
            14: 'motorbike', 15: 'person', 16: 'pottedplant',
            17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor' }
    else:
        assert(args.num_classes == 90)
        net = cv.dnn.readNetFromTensorflow(args.weights, args.prototxt)
        swapRB = True
        classNames = { 0: 'background',
            1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus',
            7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant',
            13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat',
            18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear',
            24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag',
            32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard',
            37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove',
            41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle',
            46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
            51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
            56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
            61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed',
            67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse',
            75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven',
            80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock',
            86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush' }
    
    is_image = False 
    if args.video:
        cap = cv.VideoCapture(args.video)
    elif args.image:
        is_image = True
        original_img = cv.imread(args.image)  # Preserved to be used later to get final localizations
        frame = cv.imread(args.image)
    else:
        cap = cv.VideoCapture(0)
    
    # output is a dictionary that will store the key:value of labels:list of lists of final running avg bottom left point, height, and width of the rectangular box 
    output = {}  
    avg_LeftBottom_pt = (0, 0)
    avg_height = 0
    avg_width = 0
    avg_confidence = 0
    ticker = {}  # Used to keep count of the # iteration on any label (since iterating through frame by frame but the net doesn't always pick up everything)
    PROXIMITY_THRESH = 20  # Number of minimum pixel distance required between two rectangles to be considered the same
    
    while True:
        # Capture frame-by-frame
        if not is_image:
            ret, frame = cap.read()
        blob = cv.dnn.blobFromImage(frame, inScaleFactor, (inWidth, inHeight), (meanVal, meanVal, meanVal), swapRB)
        net.setInput(blob)
        detections = net.forward()

        cols = frame.shape[1]
        rows = frame.shape[0]

        if cols / float(rows) > WHRatio:
            cropSize = (int(rows * WHRatio), rows)
        else:
            cropSize = (cols, int(cols / WHRatio))

        y1 = int((rows - cropSize[1]) / 2)
        y2 = y1 + cropSize[1]
        x1 = int((cols - cropSize[0]) / 2)
        x2 = x1 + cropSize[0]
        frame = frame[y1:y2, x1:x2]

        cols = frame.shape[1]
        rows = frame.shape[0]
        
        labels = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > args.thr:
                class_id = int(detections[0, 0, i, 1])
                labels.append(classNames[class_id])
                
        count = Counter(labels)  # dict of labels and their corresponding number of appearances if above a certain confidence threshold
        for key in count.keys():
            output[key] = [] 
            ticker[key] = ticker.get(key, 0) + 1
                
                
        print("***--------------------------*** \n%r" % Counter(labels))
                
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > args.thr:
                class_id = int(detections[0, 0, i, 1])
                l = classNames[class_id]  # the actual label in string form ('book' instead of 84)
                

                xLeftBottom = int(detections[0, 0, i, 3] * cols)
                yLeftBottom = int(detections[0, 0, i, 4] * rows)
                xRightTop   = int(detections[0, 0, i, 5] * cols)
                yRightTop   = int(detections[0, 0, i, 6] * rows)
                height = abs(int(xLeftBottom)-int(xRightTop))
                width = abs(int(yLeftBottom)-int(yRightTop))
                
                # Problems still: fix the issue with taking the average of multiple classes + putting all this code under the if class_id block
                iteration = ticker[l]
                avg_LeftBottom_pt = ((avg_LeftBottom_pt[0] * iteration + xLeftBottom)/float(iteration+1), (avg_LeftBottom_pt[1] * iteration + yLeftBottom)/float(iteration+1))
                avg_height = (avg_height * iteration + height)/float(iteration+1)
                avg_width = (avg_width * iteration + width)/float(iteration+1)
                avg_confidence = (avg_confidence * iteration + confidence)/float(iteration+1)
                output[l].append([avg_LeftBottom_pt, avg_height, avg_width, avg_confidence])
                #for v in range(count[l].values()):
                #    output[l].append([avg_LeftBottom_pt, avg_height, avg_width])
                
                print("Label: %s\nBottom left point: (%s, %s) \nTop right point: (%s, %s)\nHeight: %s\nWidth: %s\nAverages: %r\n\n\n" % (l, xLeftBottom, yLeftBottom, xRightTop, yRightTop, str(height), str(width), output))
                
                img_width, img_height = (frame.shape[1], frame.shape[0])  # tuple w shape (width, height)
                # cv.rectangle(frame, (100, 100), (img_width-100, img_height-100), (255, 255, 255))  # Test rectangle just to see where things are
                cv.rectangle(frame, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop),
                              (0, 255, 0))
                if class_id in classNames:
                    label = l + ": " + str(confidence)
                    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 1, 2)

                    yLeftBottom = max(yLeftBottom, labelSize[1])
                    
                    # This forms the filled in small white rectangle that is the label (inside of the outer green rectangle)
                    cv.rectangle(frame, (xLeftBottom, yLeftBottom - labelSize[1]),
                                         (xLeftBottom + labelSize[0], yLeftBottom + baseLine),
                                         (255, 255, 255), cv.FILLED)
                    cv.putText(frame, label, (xLeftBottom, yLeftBottom),
                                cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        image = cv.resize(frame, (700, 540))  
        cv.imshow("detections", image)
        if cv.waitKey(1) >= 0:
            break

for key, value in output.items():
    print(key, value)
    for ind, i in enumerate(value[0]):
        if ind == 3:  # Don't want to round confidence score since it is a float between 0 and 1
            break
        if type(i) == tuple:
            output[key][0][ind] = (int(i[0]), int(i[1]))
        else:
            output[key][0][ind] = int(i)
print(output, type(output.values()[0][0][1]))

for label, info in output.items():
    print(label, info)
    cv.rectangle(original_img, info[0][0], (info[0][0][0]+info[0][1], info[0][0][0]+info[0][2]), (0, 255, 0))
    
    l = label + ": " + str(info[0][3])
    labelSize, baseLine = cv.getTextSize(l, cv.FONT_HERSHEY_SIMPLEX, 1, 2)
    
    cv.rectangle(original_img, (info[0][0][0], info[0][1] - labelSize[1]), (info[0][0][0] + labelSize[0], info[0][0][1] + baseLine), (255, 255, 255), cv.FILLED)
    cv.putText(original_img, l, info[0][0], cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

image = cv.resize(original_img, (700, 540))  
cv.imshow("Localizations", image)
cv.waitKey(0)
    
print("***--------------------------------***\nFinal rolling average measurements: %r" % output)




