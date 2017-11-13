import os, sys
import rospy
import tensorflow as tf
import cv2
import numpy as np

from PIL import Image
from io import BytesIO

from collections import defaultdict
from io import StringIO

from utils import label_map_util
from utils import visualization_utils as vis_util

from styx_msgs.msg import TrafficLight

import time

dirname, filename = os.path.split(os.path.abspath(sys.argv[0]))

MODEL_NAME = 'traffic_light_graph_17111201'         #Retrained ssd_mobilenet_v1_coco

PATH_TO_CKPT = os.path.join(dirname, "light_classification/model/" + MODEL_NAME + "/frozen_inference_graph.pb")
PATH_TO_LABELS = os.path.join(dirname, "light_classification/model/" + MODEL_NAME + "/object-detection.pbtxt")
NUM_CLASSES = 3


class TLClassifier(object):

    def __init__(self):

        #Default Prediction @ startup
        self.light_prediction = TrafficLight.UNKNOWN

        #Load Label Map
        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)

        #Load Classifier
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        #Start tf session
        self.sess = tf.Session(graph=self.detection_graph)

        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')


        rospy.on_shutdown(self.cleanup)

    def cleanup(self):
        self.sess.close()

    def load_image_into_numpy_array(self, image):
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        References:
        https://stackoverflow.com/questions/40273109/convert-python-opencv-mat-image-to-tensorflow-image-data

        """

        #1= width, 0=height
        if (image.shape[1] > image.shape[0]):
            delta = image.shape[1] - image.shape[0]
            d2 = int(delta / 2)
            image = image[0:image.shape[0], d2:(image.shape[1]-d2)]

        image = cv2.resize(image,dsize=(450,450), interpolation = cv2.INTER_CUBIC)

        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image, axis=0)
      
        # Actual detection.
        (boxes, scores, classes, num) = self.sess.run(
          [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
          feed_dict={self.image_tensor: image_np_expanded})

        boxes = np.squeeze(boxes)
        classes = np.squeeze(classes).astype(np.int32)
        scores = np.squeeze(scores)

        rospy.logwarn("Light Classifier Check")

        detection_threshold = 0.40

        self.light_prediction = TrafficLight.UNKNOWN

        for i in range(boxes.shape[0]):
            if scores[i] > detection_threshold:
                object_name = self.category_index[classes[i]]['name']

                rospy.logwarn("Light Classifier Object:%s  Name=%s    Score=%s ", i, object_name, scores[i])
  
                if (object_name ==  'greenlight'):
                    self.light_prediction = TrafficLight.GREEN
                if (object_name == 'redlight'):
                    self.light_prediction = TrafficLight.RED
                if (object_name ==  'yellowlight'):
                    self.light_prediction = TrafficLight.YELLOW               

        if (self.light_prediction != TrafficLight.UNKNOWN):
            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
              image,
              boxes,
              classes,
              scores,
              self.category_index,
              use_normalized_coordinates=True,
              line_thickness=8)

            time.sleep(0.1)
            self.image_tl_boxes = image
            cv2.imwrite('temp.png', self.image_tl_boxes)



        return self.light_prediction
