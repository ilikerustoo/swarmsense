#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import sys
import os

# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# print("##################################################")
# print(os.path.dirname(os.path.abspath(__file__)))
# print((__file__))


class KeyPointClassifier(object):
    def __init__(
        self,
        # model_path='model/keypoint_classifier/keypoint_classifier.tflite',
        # model_path='/home/ilikerustoo/ros2_ws/src/crazyswarm2/crazyflie_examples/crazyflie_examples/model/keypoint_classifier/keypoint_classifer.tflite',
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'keypoint_classifier.tflite'),
        num_threads=1,
    ):
        # print("##################################################")
        # print(os.path.abspath(model_path))
        self.interpreter = tf.lite.Interpreter(model_path=model_path,
                                               num_threads=num_threads)

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def __call__(
        self,
        landmark_list,
    ):
        input_details_tensor_index = self.input_details[0]['index']
        self.interpreter.set_tensor(
            input_details_tensor_index,
            np.array([landmark_list], dtype=np.float32))
        self.interpreter.invoke()

        output_details_tensor_index = self.output_details[0]['index']

        result = self.interpreter.get_tensor(output_details_tensor_index)

        result_index = np.argmax(np.squeeze(result))

        return result_index
