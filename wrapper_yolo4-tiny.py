#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from std_msgs.msg import String, Int16
from sensor_msgs.msg import CompressedImage
import darknet_images
import darknet
import time
import argparse 
from pathlib import Path
import os
#from cv_bridge import CvBridge, CvBridgeError

#---------------------- Instractions ----------------------
# set your current diractory to darknet path 

interval = 5
global img, count

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
def parser():
    parser = argparse.ArgumentParser(description="YOLO Object Detection")
    parser.add_argument("--camera_topic", default='/kinect2/qhd/image_color_rect/compressed', type=str,
                        help="Compress Image massage topic")
    parser.add_argument("--frame_rate", default=5, type=int,
                        help="Frame rate in Hz")
    parser.add_argument("--batch_size", default=1, type=int,
                        help="number of images to be processed at the same time")
    parser.add_argument("--weights", default="data/yolov4-tiny.weights",
                        help="yolo weights path")
    parser.add_argument("--dont_show", action='store_true',
                        help="windown inference display. For headless systems")
    parser.add_argument("--ext_output", action='store_true',
                        help="display bbox coordinates of detected objects")
    parser.add_argument("--save_labels", action='store_true',
                        help="save detections bbox for each image in yolo format")
    parser.add_argument("--config_file", default=str(Path(__file__).parent)+"/cfg/yolov4-tiny.cfg",
                        help="path to config file")
    parser.add_argument("--data_file", default=str(Path(__file__).parent)+"/cfg/coco.data",
                        help="path to data file")
    parser.add_argument("--thresh", type=float, default=.25,
                        help="remove detections with lower confidence")
    return parser.parse_args()

def img_lisner(msg):
    global img
    img = msg

def image_detection(image, network, class_names, class_colors, thresh):
    # Darknet doesn't accept numpy images.
    # Create one with image we reuse for each detect
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height),
                               interpolation=cv2.INTER_LINEAR)

    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
    darknet.free_image(darknet_image)
    image = darknet.draw_boxes(detections, image_resized, class_colors)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), detections

def video_prossecing(network,class_names, class_colors):
    global img
    img_cv2 = np.fromstring(img.data, np.uint8)
    frame = cv2.imdecode(img_cv2, cv2.IMREAD_COLOR)
    # cv2.imshow('',frame)
    # cv2.waitKey(3)
    image, detections = image_detection(frame, network, class_names, class_colors, args.thresh)
    darknet.print_detections(detections, args.ext_output)
    return image, detections
    # cv2.imshow('Inference', image)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     cv2.destroyAllWindows()
    #     return

def createCompresseImage(cv2_img):
    msg = CompressedImage()
    msg.header.stamp = rospy.Time.now()
    msg.format = "jpeg"
    msg.data = np.array(cv2.imencode('.jpg', cv2_img)[1]).tostring()
    return msg

def main(args):
    global interval
    network, class_names, class_colors = darknet.load_network(args.config_file, args.data_file,
                                            args.weights,batch_size=args.batch_size)
    rate = rospy.Rate(args.frame_rate)
    pub = rospy.Publisher('/yolo4_result/image/compressed',CompressedImage,queue_size=1)
    rospy.wait_for_message(args.camera_topic, CompressedImage)
    while not rospy.is_shutdown():
        prev_time = time.time()
        # try:
        image, detections = video_prossecing(network,class_names, class_colors)
        print("FPS: {}".format(int(1/(time.time()-prev_time))))
        compress_img = createCompresseImage(image)
        pub.publish(compress_img)
        # except:
        #     pass
        # rate.sleep()

if __name__=="__main__":
    args = parser()   
    rospy.init_node("wrapper_yolo4_tiny")
    rospy.Subscriber(args.camera_topic, CompressedImage, img_lisner, queue_size=1)
    main(args)
