#Nobuyuki Tomatsu
#2019/06/02 Implement traffic light state classification.

#from styx_msgs.msg import TrafficLight
import numpy as np
import tensorflow as tf
import cv2

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        self.resized_width=32
        self.resized_height=32
        self.sess=tf.Session()
        self.saver=tf.train.import_meta_graph("./classifier_model/tl_classifier.ckpt.meta")
        self.saver.restore(self.sess, "./classifier_model/tl_classifier.ckpt")
        self.x_input=self.sess.graph.get_tensor_by_name("x_input:0")
        self.prediction=self.sess.graph.get_tensor_by_name("prediction:0")
        self.data=np.zeros([1,self.resized_height,self.resized_width,3])

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        image_hsv=cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        self.data[0,:,:,:]=cv2.resize(image_hsv,(self.resized_height,self.resized_width))
        #tl_state_pred=self.prediction.eval(feed_dict={self.x_input: self.data})
        tl_state_pred=self.sess.run("prediction:0",feed_dict={"x_input:0":self.data})[0]
        #RED=0 YELLOW=1 GREEN=2
        return tl_state_pred