#Nobuyuki Tomatsu
#2019/06/02 Implement traffic light state classification.

from styx_msgs.msg import TrafficLight
import os
import sys
import numpy as np
import tensorflow as tf
import random
import cv2
import rospy
import yaml
from collections import defaultdict

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
	config_string = rospy.get_param("/traffic_light_config")
	self.config = yaml.safe_load(config_string)
	self.is_real = self.config['is_site']
	#print(self.is_real)

	CLASSIFIER_BASE = os.path.dirname(os.path.realpath(__file__))
	if self.is_real:
		GRAPH = 'frozen_inference_graph_real.pb'
	else:
		#GRAPH = 'frozen_inference_graph.pb'
		GRAPH = 'optimized_inference_graph_sim.pb'
	self.PATH_TO_GRAPH = CLASSIFIER_BASE + '/' + GRAPH
	self.PATH_TO_LABELS = r'udacity_label_map.pbtxt'
	self.NUM_CLASSES = 13
	self.tl_state_pred = 0

	self.detection_graph = self.load_graph(self.PATH_TO_GRAPH)

    def load_graph(self, graph_file):
	graph = tf.Graph()
	with graph.as_default():
		od_graph_def = tf.GraphDef()
		with tf.gfile.GFile(graph_file, 'rb') as fid:
			serialized_graph = fid.read()
			od_graph_def.ParseFromString(serialized_graph)
			tf.import_graph_def(od_graph_def, name='')
	return graph

    def load_image_into_numpy_array(self, image):
	im_height = np.size(image, 0)
	im_width = np.size(image, 1)
	return np.array(image).reshape((im_height, im_width, 3)).astype(np.uint8)       

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        with self.detection_graph.as_default():
		with tf.Session(graph=self.detection_graph) as sess:
			image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        		detect_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        		detect_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        		detect_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        		num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

			image_np = self.load_image_into_numpy_array(image)
            		image_expanded = np.expand_dims(image_np, axis=0)

			(boxes, scores, classes, num) = sess.run([detect_boxes, detect_scores, detect_classes, num_detections], feed_dict={image_tensor: image_expanded})
			"""
			if scores[0][0] > 0.05:			
				if classes[0][0] == 1 or classes[0][0] == 4:
					self.tl_state_pred = TrafficLight.GREEN
				elif classes[0][0] == 2:
					self.tl_state_pred = TrafficLight.RED
				elif classes[0][0] == 3:
					self.tl_state_pred = TrafficLight.YELLOW
			else:
				self.tl_state_pred = TrafficLight.UNKNOWN
			"""
			if classes[0][0] == 1 or classes[0][0] == 4:
				self.tl_state_pred = TrafficLight.GREEN
			elif classes[0][0] == 2:
				self.tl_state_pred = TrafficLight.RED
			elif classes[0][0] == 3:
				self.tl_state_pred = TrafficLight.YELLOW
			
        return self.tl_state_pred
