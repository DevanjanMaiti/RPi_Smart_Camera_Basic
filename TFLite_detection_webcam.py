######## Webcam Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Dev Maiti / Evan Juras (Original)
# Date: 1st June 2025
# Description: 
# This program uses a TensorFlow Lite model to perform object detection on a live webcam
# feed. It draws boxes and scores around the objects of interest in each frame from the
# webcam. To improve FPS, the webcam object runs in a separate thread from the main program.
# This script will work with either a Picamera or regular USB webcam.
#
# This code is based off the TensorFlow Lite image classification example at:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py
#
# I added my own method of drawing boxes and labels using OpenCV.

# Import packages
import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util


##from gpiozero import MotionSensor
##from time import sleep
import subprocess
import shutil

from PIL import Image
import imagehash
##import wiringpi
##import requests
##import picamera

##pir = MotionSensor(17)


# Define VideoStream class to handle streaming of video from webcam in separate processing thread
# Source - Adrian Rosebrock, PyImageSearch: https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/
class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(640,480),framerate=30):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
            
        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

	# Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
	# Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
	# Return the most recent frame
        return self.frame

    def stop(self):
	# Indicate that the camera and thread should be stopped
        self.stopped = True

# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    required=True)
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.5)
parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                    default='1280x720')
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')

args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
use_TPU = args.edgetpu

# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

# If using Edge TPU, assign filename for Edge TPU model
if use_TPU:
    # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'       

# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
if labels[0] == '???':
    del(labels[0])

# Load the Tensorflow Lite model.
# If using Edge TPU, use special load_delegate argument
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()

#Define and start Google Drive rclone sync thread
def rclone_sync():
    while True:
        time.sleep(60)
        print("uploading images from recent incidents ....")
        
        img_dir = '/home/pi/proj/cam_vids'
        subprocess.run(['rclone', 'sync', img_dir, 'Cam_Drive:Cam_Drive'])
        
        print("upload complete!")

rclone_sync_thread = Thread(target=rclone_sync)
rclone_sync_thread.start()

# Initialize video stream
videostream = VideoStream(resolution=(imW,imH),framerate=30).start()
time.sleep(1)

first_loop = 1

#for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
while True:

    gdrive_clr_loop_cnt = 0
    
    if first_loop:
        print("camera starting, cleaning up remote storage ....")
    else:
        print("camera paused till remote storage cleanup ....")
        
    first_loop = 0
        
    subprocess.run(['rclone', 'delete', 'Cam_Drive:Cam_Drive'])
    os.system('rm -rf /home/pi/proj/cam_vids/*')
    
    time.sleep(1)
    hash_prev = 0
    img_hash_cutoff = 5
    light_threshold = 50

    print("camera running, looking around!!")

    while gdrive_clr_loop_cnt < 10000: ## total images = 10000
        
        
        # Start timer (for calculating frame rate)
        t1 = cv2.getTickCount()
            
        # Grab frame from video stream
        frame1 = videostream.read()

        # Acquire frame and resize to expected shape [1xHxWx3]
        frame = frame1.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Darkness Detection
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        avg_brightness = np.mean(frame_gray)
        if avg_brightness < light_threshold:
            print("too dark to process, camera paused ....")
            break

        frame_resized = cv2.resize(frame_rgb, (width, height))
        input_data = np.expand_dims(frame_resized, axis=0)

        # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
        if floating_model:
            input_data = (np.float32(input_data) - input_mean) / input_std

        # Perform the actual detection by running the model with the image as input
        interpreter.set_tensor(input_details[0]['index'],input_data)
        interpreter.invoke()

        # Retrieve detection results
        boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
        classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
        scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects
        #num = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)

        # Loop over all detections and draw detection box if confidence is above minimum threshold
        for i in range(len(scores)):
            if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

                # Draw label
                object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
                if (object_name == 'person' or object_name == 'car'):
       
                    print(object_name, "detected!!")
                        
                    time_local = time.localtime()
                    time_str = time.asctime(time_local)
                    img_fpath = '/home/pi/proj/cam_vids/%s.jpg' % time_str
                    cv2.imwrite(img_fpath, frame)

                    ## Image hash based filtering to avoid storing duplicates images; eg: when a car is parked in front of camera
                    hash_new = imagehash.average_hash(Image.open(img_fpath))
                    hash_new_int = hash_new.__hash__()

                    hash_diff = abs(hash_prev - hash_new_int)
                    if hash_diff < img_hash_cutoff:
                        print("similar images, event ignored ....")
                        os.remove(img_fpath)
                    hash_prev = hash_new_int

                    gdrive_clr_loop_cnt = gdrive_clr_loop_cnt + 1
                    break

            
            # Calculate framerate
            t2 = cv2.getTickCount()
            time1 = (t2-t1)/freq
            frame_rate_calc= 1/time1
##            print("frame rate =", frame_rate_calc) # avoiding for now since printing same frame rate every time doesn't make sense
            
        time.sleep(1) # adding a gap to avoid detecting too many similar images

        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break

# Clean up
cv2.destroyAllWindows()
videostream.stop()
rclone_sync_thread.stop()
