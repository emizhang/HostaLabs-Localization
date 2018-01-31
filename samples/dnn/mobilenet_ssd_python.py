# TO RUN IT: python mobilenet_ssd_python.py --image 'dorm_room.jpg' --num_classes 90 --prototxt 'ssd_mobilenet_v1_coco.pbtxt' --weights 'frozen_inference_graph.pb'
# If you want to change the image input, just change the string after --image

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
import json
import copy

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
        classNames = {0: 'background',
                      1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
                      5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
                      10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
                      14: 'motorbike', 15: 'person', 16: 'pottedplant',
                      17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor'}
    else:
        assert (args.num_classes == 90)
        net = cv.dnn.readNetFromTensorflow(args.weights, args.prototxt)
        swapRB = True
        classNames = {0: 'background',
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
                      86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}
        
        kitchen_list = ['cat', 'dog', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon'
                        'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza'
                        'donut', 'cake', 'chair', 'potted plant', 'dining table', 'laptop', 'cell phone', 'microwave'
                        'oven', 'toaster', 'sink', 'refrigerator']

    is_image = False
    if args.video:
        cap = cv.VideoCapture(args.video)
    elif args.image:
        is_image = True
        original_img = cv.imread(args.image)  # Preserved to be used later to get final localizations
        frame = cv.imread(args.image)
    else:
        cap = cv.VideoCapture(0)

    # output is a dictionary of dictionaries that will store the class:informatoin of info type mapped to corresponding final bottom left point, height, and width of the rectangular box
    output = {}
    best_LeftBottom_pt = (0, 0)
    best_height = 0
    best_width = 0
    best_confidence = {}  # Dictionary of labels mapped to their best confidence scores
    frame_num = 0
    PROXIMITY_THRESH = 40  # threshold for how many pixels between two rectangular boxes before they are considered to belong to the same label

    while frame_num < 50:
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

        # Plan to handle repeats of the same label:
        # 1. Initialize output dict so that this can be reflected (easy with Counter)
        # 2. Use proximity threshold to determine which rectangles should be considered the same
        if frame_num == 0:
            # Don't need Counter or any of the code in the next block but keeping it for now in case I want to fix the situation of multiple classes later
            labels = []
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                class_id = int(detections[0, 0, i, 1])
                if confidence > args.thr and classNames[class_id] in kitchen_list:
                    labels.append(classNames[class_id])

            count = Counter(
                labels)  # dict of labels and their corresponding number of appearances if above a certain confidence threshold
            # print("***--------------------------*** \n%r" % Counter(labels))
            for key, num_appearances in count.items():
                # if num_appearances > 1:
                #     for i in range(num_appearances):
                #         k = "%s #%d" % (key, i + 1)
                #         output[k] = {}
                #         best_confidence[k] = 0
                # else:
                output[key] = {}
                best_confidence[key] = 0

            # print("INIT OUTPUT: ", output, "INIT BEST CONFIDENCE: ", best_confidence)

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            class_id = int(detections[0, 0, i, 1])
            l = classNames[class_id]  # the actual label in string form ('book' instead of 84)
            if confidence > args.thr and l in kitchen_list:

                xLeftBottom = int(detections[0, 0, i, 3] * cols)
                yLeftBottom = int(detections[0, 0, i, 4] * rows)
                xRightTop = int(detections[0, 0, i, 5] * cols)
                yRightTop = int(detections[0, 0, i, 6] * rows)
                height = abs(int(yLeftBottom) - int(yRightTop))
                width = abs(int(xLeftBottom) - int(xRightTop))

                # img_width, img_height = (frame.shape[1], frame.shape[0])  # tuple w shape (width, height)
                # cv.rectangle(frame, (100, 100), (img_width-100, img_height-100), (255, 255, 255))  # Test rectangle just to see where things are
                cv.rectangle(frame, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop),
                             (0, 255, 0))

                if class_id in classNames:

                    # Compare confidence with the curr best confidence for that class, 0 if none
                    if confidence > best_confidence.get(l, 0):
                        best_LeftBottom_pt = (xLeftBottom, yLeftBottom)
                        best_height = height
                        best_width = width
                        best_confidence[l] = confidence

                        try:
                            output[l]['bottom_left'] = best_LeftBottom_pt
                            output[l]['width'] = best_width
                            output[l]['height'] = best_height
                            output[l]['confidence'] = best_confidence[l]
                        except KeyError:
                            output[
                                l] = {}  # Initialize the label's dictionary to store info if it wasn't recognized at first
                            output[l]['bottom_left'] = best_LeftBottom_pt
                            output[l]['width'] = best_width
                            output[l]['height'] = best_height
                            output[l]['confidence'] = best_confidence[l]

                        print(
                            "Label: %s\nBottom left point: (%s, %s) \nTop right point: (%s, %s)\nHeight: %s\nWidth: %s\nBest: %r\n\n\n" % (
                            l, xLeftBottom, yLeftBottom, xRightTop, yRightTop, str(height), str(width), output))

                    label = l + ": " + str(confidence)
                    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    labelSize = (int(labelSize[0] * 0.5), int(labelSize[1] * 0.5))
                    baseLine = int(baseLine * 0.5)

                    yLeftBottom = max(yLeftBottom, labelSize[1])

                    # # This forms the filled in small white rectangle that is the label (inside of the outer green rectangle)
                    # cv.rectangle(frame, (xLeftBottom, yLeftBottom - labelSize[1]),
                    #              (xLeftBottom + labelSize[0], yLeftBottom + baseLine),
                    #              (255, 255, 255),
                    #              cv.FILLED)  # larger pictures mult labelSize[1]*2, labelSize[0]*3, baseLine*2
                    # cv.putText(frame, label, (xLeftBottom, yLeftBottom),
                    #            cv.FONT_HERSHEY_SIMPLEX, 0.25, (0, 0, 0), 1)  # larger pictures set font=3, thick=3

        frame_num += 1
        # image = cv.resize(frame, (720, 540))
        # cv.imshow("detections", image)
        # if cv.waitKey(1) >= 0:
        #     break

# Need to crop the original image so that it matches up with the image from frame-by-frame localization step above
cols = original_img.shape[1]
rows = original_img.shape[0]

if cols / float(rows) > WHRatio:
    cropSize = (int(rows * WHRatio), rows)
else:
    cropSize = (cols, int(cols / WHRatio))

y1 = int((rows - cropSize[1]) / 2)
y2 = y1 + cropSize[1]
x1 = int((cols - cropSize[0]) / 2)
x2 = x1 + cropSize[0]
original_img = original_img[y1:y2, x1:x2]

# Making a copy of output dictionary so that I can edit output's entries without having to worry about issues when looping through it
temp_dict = copy.deepcopy(output)
# # Want to reproduce the image except with only the best (highest confidence) rectangles and their corresponding classes remaining.
for label, info in temp_dict.items():
    # print(label, info, type(info['confidence']))

    # Converting everything to integers instead of floats
    for data_type, data_number in info.items():
        if data_type == 'confidence':  # Don't want to round confidence score to int since it is a float between 0 and 1
            output[label][
                data_type] = data_number.item()  # numpy.float32 is incompatible with Python's JSON module when converting, so .item() -> float
            # print(output[label][data_type], type(output[label][data_type]))
        elif type(data_number) == tuple:
            output[label][data_type] = (int(data_number[0]), int(data_number[1]))
        else:
            output[label][data_type] = int(data_number)

    # if len(info) > 1:
    #    best_conf = 0
    #    best_info_ind = 0
    #    for ind, a in enumerate(info):
    #        if a[3] > best_conf:
    #            best_info_ind = ind
    #            best_conf = a[3]
    #
    #    best_info = info[best_info_ind]
    #    output[label][0] = best_info


    # bottomleftpt = info['bottom_left']
    # width = info['width']
    # height = info['height']
    # conf = info['confidence']
    # print(bottomleftpt, width, height, conf)

    # l = label + ": " + str(conf)
    # labelSize, baseLine = cv.getTextSize(l, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    # labelSize = (int(labelSize[0] * 0.5), int(labelSize[1] * 0.5))
    # baseLine = int(baseLine * 0.5)

    # cv.rectangle(original_img, (bottomleftpt[0], bottomleftpt[1] - labelSize[1]),
    #              (bottomleftpt[0] + labelSize[0], bottomleftpt[1] + baseLine), (255, 255, 255),
    #              cv.FILLED)  # larger pictures mult labelSize[1]*2, labelSize[0]*3, baseLine*2
    # cv.putText(original_img, l, bottomleftpt, cv.FONT_HERSHEY_SIMPLEX, 0.25, (0, 0, 0),
    #            1)  # larger pictures set font=3, thick=3

    # cv.rectangle(original_img, bottomleftpt, (bottomleftpt[0] + width, bottomleftpt[1] + height), (0, 255, 0),
    #              1)  # larger pics set thick=4

with open('img_data.json', 'w') as outfile:
    json.dump(output, outfile)

# img_width, img_height = (original_img.shape[1], original_img.shape[0])
# print("width/height", img_width, img_height)

# image = cv.resize(original_img, (720, 540))
# cv.imshow("Localizations", image)
# cv.waitKey(0)

# print("***--------------------------------***\nFinal rolling average measurements: %r" % output)
# print("LIST OF IDENTIFIED OBJECTS: %r" % list(output.keys()))




