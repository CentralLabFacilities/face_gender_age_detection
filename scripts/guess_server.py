#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time
from data import inputs
import numpy as np
import tensorflow as tf
from model import select_model, get_checkpoint
from utils import *
import os
import json
import csv

import cv2
from cv_bridge import CvBridge, CvBridgeError

from gender_and_age.srv import GenderAndAgeService
from gender_and_age.msg import *
import rospy

RESIZE_FINAL = 227
GENDER_LIST =['M','F']
AGE_LIST = ['(0, 2)','(4, 6)','(8, 12)','(15, 20)','(25, 32)','(38, 43)','(48, 53)','(60, 100)']
MAX_BATCH_SZ = 128
pre_trained_age_checkpoint = '/home/susan/Downloads/age_and_gender/22801/'
pre_trained_gender_checkpoint = '/home/susan/Downloads/age_and_gender/21936/'



def classify_one_multi_crop(sess, label_list, softmax_output, images, img2):
    try:

	#Format for the images Tensor
	img2= cv2.resize(img2,dsize=(227,227), interpolation = cv2.INTER_CUBIC)
	#Numpy array
	np_image_data = np.asarray(img2)
	
	np_image_data=cv2.normalize(np_image_data.astype('float'), None, -0.5, .5, cv2.NORM_MINMAX)

	np_final = np.expand_dims(np_image_data,axis=0)	

	#now feeding it into the session:
	#[... initialization of session and loading of graph etc]
	batch_results = sess.run(softmax_output,
                           feed_dict={images: np_final})


        output = batch_results[0]
        batch_sz = batch_results.shape[0]
    
        for i in range(1, batch_sz):
            output = output + batch_results[i]
        
        output /= batch_sz
        best = np.argmax(output)
        best_choice = (label_list[best], output[best])
        print('Guess @ 1 %s, prob = %.2f' % best_choice)

	
    
        nlabels = len(label_list)
        if nlabels > 2:
            output[best] = 0
            second_best = np.argmax(output)
            print('Guess @ 2 %s, prob = %.2f' % (label_list[second_best], output[second_best]))

    except Exception as e:
        print(e)
        #print('Failed to run image %s ' % image_file)
    return label_list[best], output[best]




#def main(argv=None):
#    print('main')
    

def gender_and_age_detection(req):  # pylint: disable=unused-argument
    print('gender_and_age_detection')
    gender_list = []
    age_list = []
    response = []
    cv_images = []
    bridge = CvBridge()
    for object in req.objects:
	cv_images.append(bridge.imgmsg_to_cv2(object, desired_encoding="bgr8"))
    config = tf.ConfigProto(allow_soft_placement=True)

    
    with tf.Session(config=config) as sess_gender:

        label_list = GENDER_LIST
        nlabels = len(label_list)

        print('Executing on /gpu:0')
        model_fn = select_model('inception')
    
        with tf.device('/gpu:0'):
            
            images = tf.placeholder(tf.float32, [None, RESIZE_FINAL, RESIZE_FINAL, 3])
            logits = model_fn(nlabels, images, 1, False)
            init = tf.global_variables_initializer()

            checkpoint_path = pre_trained_gender_checkpoint
            model_checkpoint_path, global_step_gender = get_checkpoint(checkpoint_path, None, 'checkpoint')
 
            saver = tf.train.Saver()
            saver.restore(sess_gender, model_checkpoint_path)
                        
            softmax_output = tf.nn.softmax(logits)

            for cv_image in cv_images:
		if cv_image is not None:
			gender, prob = classify_one_multi_crop(sess_gender, label_list, softmax_output, images,cv_image)
			gender_list.append(GenderProbability(gender, prob))
            		
                    
   
    tf.reset_default_graph()

    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess_age:

        label_list = AGE_LIST
        nlabels = len(label_list)

        print('Executing on /cpu:0')
        model_fn = select_model('inception')

        with tf.device('/cpu:0'):
            
            images = tf.placeholder(tf.float32, [None, RESIZE_FINAL, RESIZE_FINAL, 3])
	    logits = model_fn(nlabels, images, 1, False)
	    
            init = tf.global_variables_initializer()
            checkpoint_path = pre_trained_age_checkpoint
	    
            model_checkpoint_path, global_step = get_checkpoint(checkpoint_path, None, 'checkpoint')
            
            saver = tf.train.Saver()
            saver.restore(sess_age, model_checkpoint_path)
                        
            softmax_output = tf.nn.softmax(logits)
                
            for cv_image in cv_images:
		if cv_image is not None:
            		age, prob = classify_one_multi_crop(sess_age, label_list, softmax_output, images,cv_image)
			age_list.append(AgeProbability(age,prob))
    

    if len(gender_list) == len(age_list):
	gender_age_list = []
	for x in range(0, len(gender_list)):
	    gender_age_list.append(GenderAndAge(gender_list[x], age_list[x]))
    	response = GenderAndAgeList(gender_age_list)
	return response
        
if __name__ == '__main__':
    rospy.init_node('guess_server')
    srv = rospy.Service('gender_and_age', GenderAndAgeService, gender_and_age_detection)
    rospy.spin()
