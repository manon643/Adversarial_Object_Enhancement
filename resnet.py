import tensorflow as tf
import numpy as np
import utils

from cnn_modules import CNN

class ResNet50(CNN):
    """ This class contains components of ResNet50 architecture """

    def resnet_block(self, input_feature_map, number_bottleneck_channels,
    number_input_channels, number_output_channels, is_training, stride=[1, 1, 1, 1]):
        """ Run a ResNet block """
        out_1 = CNN._conv2d(self, input_feature_map, [1, 1, number_input_channels, number_bottleneck_channels],
        [number_bottleneck_channels], [1, 1, 1, 1], 'bottleneck_down', is_training)
        out_2 = CNN._conv2d(self, out_1, [3, 3, number_bottleneck_channels, number_bottleneck_channels],
        [number_bottleneck_channels], stride, 'conv3x3', is_training)
        out_3 = CNN._conv2d(self, out_2, [1, 1, number_bottleneck_channels, number_output_channels],
        [number_output_channels], [1, 1, 1, 1], 'bottleneck_up', is_training, False)

        # Shortcut connection
        if number_input_channels != number_output_channels:
            identity_mapping = CNN._conv2d(self, input_feature_map, [1, 1, number_input_channels, number_output_channels],
        [number_output_channels], stride, 'identity_mapping', is_training, False)
            return tf.nn.relu(tf.add(identity_mapping, out_3))
        else:
            return tf.nn.relu(tf.add(input_feature_map, out_3))

    def resnet_module(self, input_data, number_blocks, number_bottleneck_channels, number_input_channels,
                    number_output_channels, is_training, stride=[1, 2, 2, 1]):
        """ Run a ResNet module consisting of residual blocks """
        for index, block in enumerate(range(number_blocks)):
            if index == 0:
                with tf.variable_scope('module' + str(index)):
                    out = self.resnet_block(input_data, number_bottleneck_channels, number_input_channels,
                    number_output_channels, is_training, stride=stride)
            else:
                with tf.variable_scope('module' + str(index)):
                    out = self.resnet_block(out, number_bottleneck_channels, number_output_channels,
                    number_output_channels, is_training, stride=[1, 1, 1, 1])
        return out

    def construct_backbone_architecture(self, x_train, is_training, endpoints):
        """ This function construct the backbone architecture to get detection feature maps """
        with tf.variable_scope("FirstStageFeatureExtractor") as scope:
            out_1 = CNN._conv2d(self, x_train, [7, 7, 3, 64], [64], [1, 2, 2, 1], 'conv3x3', is_training)
            out_1_pool = tf.nn.max_pool(out_1, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')
        with tf.variable_scope("ResNetBlock1"):
            out_2 = self.resnet_module(out_1_pool, 3, 64, 64, 256, is_training, [1, 1, 1, 1])
            endpoints['block2'] = out_2
        with tf.variable_scope("ResNetBlock2"):
            out_3 = self.resnet_module(out_2, 4, 128, 256, 512, is_training)
            endpoints['block3'] = out_3
        with tf.variable_scope("ResNetBlock3"):
            out_4 = self.resnet_module(out_3, 6, 256, 512, 1024, is_training)
            endpoints['block4'] = out_4
        with tf.variable_scope("ResNetBlock4"):
            out_5 = self.resnet_module(out_4, 3, 512, 1024, 2048, is_training)
            endpoints['block5'] = out_5
        return endpoints
