#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage
import darknet
import argparse
from pathlib import Path
import os
from obj_detector.msg import Detection_msg

# ---------------------- Discription ----------------------\
# Subscribe to camera topic and run Yolo4-tiny to detect items. Then
# the node publish two topics:
# image with boundaries box
# detection massage with the field: [header, class id, score, pose [x_center, y_center, size_x, size_y]]
# Yolo is changing the image resolution, therefore there is transformation
# back to armdillo camera resolution.
# The yolo parameters are define in parser function, the user can define
# them while excuting this script. Another way to change them is by
# change the defult value at the parser function.


global img, count, new_shape

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
    parser.add_argument("--weights", default="data/yolov4.weights",
                        help="yolo weights path")
    parser.add_argument("--dont_show", action='store_true',
                        help="windown inference display. For headless systems")
    parser.add_argument("--ext_output", action='store_true',
                        help="display bbox coordinates of detected objects")
    parser.add_argument("--save_labels", action='store_true',
                        help="save detections bbox for each image in yolo format")
    parser.add_argument("--config_file", default=str(Path(__file__).parent)+"/cfg/yolov4.cfg",
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


def video_prossecing(network, class_names, class_colors):
    global img, new_shape
    img_cv2 = np.fromstring(img.data, np.uint8)
    frame = cv2.imdecode(img_cv2, cv2.IMREAD_COLOR)
    new_shape = frame.shape
    image, detections = image_detection(frame, network, class_names, class_colors, args.thresh)
    return image, detections


def createCompresseImage(cv2_img):
    msg = CompressedImage()
    msg.header.stamp = rospy.Time.now()
    msg.format = "jpeg"
    msg.data = np.array(cv2.imencode('.jpg', cv2_img)[1]).tostring()
    return msg


def detection_publish(detections, publisher):
    for detection in detections:
        obj_msg = Detection_msg()
        obj_msg.header.stamp = rospy.Time.now()
        obj_msg.header.frame_id = "kinect2_link"
        obj_msg.class_id = detection[0]
        obj_msg.score = float(detection[1])
        obj_msg.pose.x_center = detection[2][0]
        obj_msg.pose.y_center = detection[2][1]
        obj_msg.pose.size_x = detection[2][2]
        obj_msg.pose.size_y = detection[2][3]
        publisher.publish(obj_msg)


def detection_tf(old_shape, new_shape, detections):
    # Transformation of detections from Yolo resolution to armadillo.
    height_factor = new_shape[0] / old_shape[0]
    width_factor = new_shape[1] / old_shape[1]
    detect_tf = lambda detect: [detect[2][0]*width_factor, detect[2][1]*height_factor,
                                detect[2][2]*width_factor, detect[2][3]*height_factor]
    for i in range(len(detections)):
        detections[i][2] = detect_tf(detections[i])
    return detections


def print_img(detections):
    # This function is for testing the TF from yolo to armadillo camera.
    global img
    img_cv2 = np.fromstring(img.data, np.uint8)
    frame = cv2.imdecode(img_cv2, cv2.IMREAD_COLOR)
    for detect in detections:
        x = int(detect[2][0])
        y = int(detect[2][1])
        cv2.circle(frame, (x,y), radius=2, color=(0, 0, 255), thickness=-5)
    cv2.imshow('img',frame)
    cv2.waitKey(1)


def tuple2list(t):
    return list(map(tuple2list, t)) if isinstance(t, (tuple, list)) else t


def main(args):
    global new_shape
    print(os.path.exists(args.weights))
    if not os.path.exists(args.weights):
        print("Invalid weight path {}".format(os.path.abspath(args.weights)))
        download_link = "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights"
        git_hub = "https://github.com/AlexeyAB/darknet"
        print("1. download weights for tiny-4 from this link: \n"
              "{} \n\n 2. Allocate the weights in {}/data/???.weights \n\n You can also visit darknet"
              " git for more instructions: {}".format(download_link, str(Path(__file__).parent), git_hub))
        return
    network, class_names, class_colors = darknet.load_network(args.config_file,
                                                              args.data_file,
                                                              args.weights,
                                                              batch_size=args.batch_size)

    rate = rospy.Rate(args.frame_rate)
    pub = rospy.Publisher('/yolo4_result/compressed', CompressedImage, queue_size=1)
    detection_publisher = rospy.Publisher('/yolo4_result/detections',
                                          Detection_msg, queue_size=10)
    rospy.wait_for_message(args.camera_topic, CompressedImage)

    while not rospy.is_shutdown():
        image, detections = video_prossecing(network, class_names, class_colors)
        compress_img = createCompresseImage(image)
        detections = tuple2list(detections)
        detections = detection_tf(image.shape, new_shape, detections)
        # print_img(detections) # For testing the TF.
        pub.publish(compress_img)
        detection_publish(detections, detection_publisher)
        rate.sleep()


if __name__ == "__main__":
    args = parser()
    rospy.init_node("wrapper_yolo4_tiny", anonymous=True)
    rospy.Subscriber(args.camera_topic, CompressedImage, img_lisner, queue_size=1)
    main(args)
