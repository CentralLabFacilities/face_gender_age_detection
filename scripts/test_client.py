#!/usr/bin/env python

import sys
import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
import os

from gender_and_age.srv import *
from gender_and_age.msg import *

def add_two_ints_client():
    print('add_two_ints_client - start')
    rospy.wait_for_service('gender_and_age')
    try:
	req = []
	bridge = CvBridge()
	
	images_men = []
    	for filename in os.listdir('/home/susan/Downloads/age_and_gender/gender_training_4/men/'):
        	img = cv2.imread(os.path.join('/home/susan/Downloads/age_and_gender/gender_training_4/men/',filename))
        	if img is not None:
            		req.append(bridge.cv2_to_imgmsg(img, encoding="bgr8"))
	
	"""
    	for filename in os.listdir('/home/susan/Downloads/age_and_gender/gender_training_4/women/'):
        	img = cv2.imread(os.path.join('/home/susan/Downloads/age_and_gender/gender_training_4/women/',filename))
        	if img is not None:
			req.append(bridge.cv2_to_imgmsg(img, encoding="bgr8"))
      	
	"""


	
	#cv_image_1 = cv2.imread('/home/susan/Downloads/age_and_gender/gender_training_4/women/ffbfb90e5a743fd15637eb3168357411cc72c89d.jpg',1)
	#cv_image_2 = cv2.imread('/home/susan/Downloads/age_and_gender/gender_training_4/women/ffe06aed82e414506e7e21d68cefdd0ffc248b00.jpg',1)

	#image_message_1 = bridge.cv2_to_imgmsg(cv_image_1, encoding="bgr8")
	#image_message_2 = bridge.cv2_to_imgmsg(cv_image_2, encoding="bgr8")

	
	#req.append(image_message_1)
	#req.append(image_message_2)
	

        add_two_ints = rospy.ServiceProxy('gender_and_age', GenderAndAgeService)
        print "%s " %add_two_ints(req)
	
    except rospy.ServiceException, e:
        print "Service call failed: %s"%e



if __name__ == "__main__":
    add_two_ints_client()
