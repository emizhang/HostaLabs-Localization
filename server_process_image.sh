#! /bin/bash

#go to directory 
cd ~/HostaLabs-Localization/samples/dnn/

#saves image from server as "new_name.png in same directory as this file"
wget http://ec2-52-91-230-86.compute-1.amazonaws.com/get_original_image_recent -O new_image.png

#runs python script to get output
python mobilenet_ssd_python.py --image 'new_image.png' --num_classes 90 --prototxt 'ssd_mobilenet_v1_coco.pbtxt' --weights 'frozen_inference_graph.pb'

#output will be ~/HostaLabs-Localization/samples/dnn/img_data.json (also posted to server at http://ec2-52-91-230-86.compute-1.amazonaws.com/get_original_image_recent_with_data)


