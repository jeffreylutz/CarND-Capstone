import rospy
import tensorflow as tf
import cv2
import numpy as np
import os, sys


from styx_msgs.msg import TrafficLight

dirname, filename = os.path.split(os.path.abspath(sys.argv[0]))



MODEL_PATH = os.path.join(dirname, "light_classification/model/retrained_graph.pb")
LABELS_PATH = os.path.join(dirname, "light_classification/model/retrained_labels.txt")

class TLClassifier(object):

    def __init__(self):
        #TODO load classifier

        self.imgnum = 0        
        # Loads label file, strips off carriage return
        self.label_lines = [line.rstrip() for line 
                           in tf.gfile.GFile(LABELS_PATH)]

        # Unpersists graph from file
        with tf.gfile.FastGFile(MODEL_PATH, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')

        #Start tf session
        self.sess = tf.Session()

        # Feed the image_data as input to the graph and get first prediction
        #self.softmax_tensor = self.sess.graph.get_tensor_by_name('final_result:0')

        rospy.on_shutdown(self.cleanup)

    def cleanup(self):
        self.sess.close()

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        References:
        https://stackoverflow.com/questions/40273109/convert-python-opencv-mat-image-to-tensorflow-image-data

        """
        #TODO implement light color prediction

        self.imgnum += 1

        self.softmax_tensor = self.sess.graph.get_tensor_by_name('final_result:0')

        #Convert

        #1= width, 0=height
        if (image.shape[1] > image.shape[0]):
            delta = image.shape[1] - image.shape[0]
            d2 = int(delta / 2)
            image = image[0:image.shape[0], d2:(image.shape[1]-d2)]

        img2= cv2.resize(image,dsize=(299,299), interpolation = cv2.INTER_CUBIC)
        #img2= cv2.resize(image,dsize=(224,224), interpolation = cv2.INTER_CUBIC)  #MobileNet needs 224,224

        #Numpy array
        np_image_data = np.asarray(img2)
        
        np_image_data = cv2.normalize(np_image_data.astype('float'), None, -0.5, 0.5, cv2.NORM_MINMAX)
        
        np_final = np.expand_dims(np_image_data,axis=0)

        #cv2.imwrite((str(self.imgnum) + 'temp.png'), img2)

        #now feeding it into the session:
        #[... initialization of session and loading of graph etc]

        #USE Mul:0 as input for inception
        predictions = self.sess.run(self.softmax_tensor,
                                   {'Mul:0': np_final})

        #USE input:0 for MobileNet
        #predictions = self.sess.run(self.softmax_tensor,
        #                           {'input:0': np_final})

        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
            
        human_string = self.label_lines[top_k[0]]
        score = predictions[0][top_k[0]]

        light_prediction = TrafficLight.UNKNOWN
        pred = "UNKNOWN"

        if (score > 0.64):
            if (human_string == 'redlight'):
                light_prediction = TrafficLight.RED
                pred = "RED LIGHT"
            if (human_string == 'greenlight'):
                light_prediction = TrafficLight.GREEN
                pred = "GREEN LIGHT"


        rospy.logwarn("Light Classifier: %s  Score=%s   %s", human_string, score, pred)

        return light_prediction
