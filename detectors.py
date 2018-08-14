import tensorflow as tf
import numpy as np
import utils
import pdb
import math
import glob
import argparse

class SSDResNet():
    """ This class contains the components of the resnet Architecture """
    feature_layers = ['block3', 'block4', 'block5']

    def __init__(self):
        """ Constructor for the SSD-resnet Model """
        self.number_classes = 9 # +1 for background class
        self.number_iterations = 1000
        self.anchor_sizes = [(15., 30.),
                      (45., 60.),
                      (75., 90.)]
        self.anchor_ratios = [[2, .5, 3., 1./3.],
                        [2, .5, 3., 1./3.],
                        [2, .5, 3., 1./3.]]
        self.feat_shapes = [[28, 28],[14, 14],[7, 7]]
        self.anchor_steps = [8, 16.5, 33]
        self.img_shape = [224, 224]
        self.batch_size = 24
        self.number_iterations_dataset = 1000
        self.buffer_size = 1000
        self.positive_threshold = 0.5
        self.negative_threshold = 0.5
        self.select_threshold = 0.5
        self.negatives_ratio = 3
        self.learning_rate = 1e-3
        self.label_map = {'max': 0}

    def variable_summaries(self, var):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
              stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

    def weight_variable(self, shape, filter_name):
        """ Define the Weights and Initialize Them and Attach to the Summary """
        weights = tf.get_variable(filter_name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())
        self.variable_summaries(weights)
        return weights

    def bias_variable(self, shape, bias_name):
        """ Define the Biases and Initialize Them and Attach to the Summary """
        bias = tf.get_variable(bias_name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())
        self.variable_summaries(bias)
        return bias

    def _batch_norm(self, input, filter_id, is_training):
        """ Apply Batch Normalization After Convolution and Before Activation """
        input_norm = tf.contrib.layers.batch_norm(input, decay = 0.95, center=True, scale=True, is_training=is_training)
        return input_norm

    def _conv2d(self, input_data, shape, bias_shape, stride, filter_id, is_training, padding='SAME'):
        """ Perform 2D convolution on the input data and apply RELU """
        weights = self.weight_variable(shape, 'weights' + filter_id)
        bias = self.bias_variable(bias_shape, 'bias' + filter_id)
        output_conv = tf.nn.conv2d(input_data, weights, strides=stride, padding='SAME')
        output_conv_norm = self._batch_norm(output_conv + bias, filter_id, is_training)
        return tf.nn.relu(output_conv_norm)

    def _fcl(self, input_data, shape, bias_shape, filter_id, classification_layer=False):
        """ Run a Fully Connected Layer and ReLU if necessary """
        weights = self.weight_variable(shape, 'weights'+  filter_id)
        bias = self.bias_variable(bias_shape, 'bias' + filter_id)

        if classification_layer:
            return tf.matmul(input_data, weights) + bias
        else:
            out_fc_layer = tf.reshape(input_data, [-1, shape[0]])
            return tf.nn.relu(tf.matmul(out_fc_layer, weights) + bias)

    def resnet_block(self, input_feature_map, number_bottleneck_channels,
    number_input_channels, number_output_channels, is_training, stride=[1, 1, 1, 1]):
        """ Run a ResNet block """
        out_1 = self._conv2d(input_feature_map, [1, 1, number_input_channels, number_bottleneck_channels],
        [number_bottleneck_channels], [1, 1, 1, 1], 'bottleneck_down', is_training)
        out_2 = self._conv2d(out_1, [3, 3, number_bottleneck_channels, number_bottleneck_channels],
        [number_bottleneck_channels], stride, 'conv3x3', is_training)
        out_3 = self._conv2d(out_2, [1, 1, number_bottleneck_channels, number_output_channels],
        [number_output_channels], [1, 1, 1, 1], 'bottleneck_up', is_training)
        identity_mapping = self._conv2d(input_feature_map, [1, 1, number_input_channels, number_output_channels],
        [number_output_channels], stride, 'identity_mapping', is_training)
        return tf.add(identity_mapping, out_3)

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

    def ssd_anchor_box_encoder(self, index, dtype=np.float32, offset=0.5):
        """ Compute SSD anchor boxes in the domain of feature maps of interest to perform
            detections.
        """
        # Compute the position grid: simple way.
        y, x = np.mgrid[0:self.feat_shapes[index][0], 0:self.feat_shapes[index][1]]
        y = (y.astype(dtype) + offset) * self.anchor_steps[index] / self.img_shape[0]
        x = (x.astype(dtype) + offset) * self.anchor_steps[index] / self.img_shape[1]

        # Expand dims to support easy broadcasting.
        y = np.expand_dims(y, axis=-1)
        x = np.expand_dims(x, axis=-1)

        # Compute relative height and width.
        num_anchors = len(self.anchor_sizes[index]) * (len(self.anchor_ratios[index]) + 1)
        h = np.zeros((num_anchors, ), dtype=dtype)
        w = np.zeros((num_anchors, ), dtype=dtype)

        # Add first anchor boxes with ratio=1.
        anchor_counter = 0
        for temp_index in range(len(self.anchor_sizes[index])):
            anchor_index = temp_index*(anchor_counter*(len(self.anchor_ratios[index])+1))
            h[anchor_index] = self.anchor_sizes[index][temp_index] / self.img_shape[0]
            w[anchor_index] = self.anchor_sizes[index][temp_index] / self.img_shape[1]
            for i, r in enumerate(self.anchor_ratios[index]):
                h[anchor_index+i+1] = self.anchor_sizes[index][temp_index] / self.img_shape[0] / math.sqrt(float(r))
                w[anchor_index+i+1] = self.anchor_sizes[index][temp_index] / self.img_shape[1] * math.sqrt(float(r))
            anchor_counter += 1

        return y, x, h, w

    def detection_layer(self, inputs, index):
        """ Predict bounding box locations and classes in each head """
        net = inputs
        num_anchors = len(self.anchor_sizes[index]) * (len(self.anchor_ratios[index]) + 1)

        # Location prediction - Returns nxnx(4xnum_anchors) tensor
        num_loc_pred = num_anchors * 4
        filter_loc = tf.get_variable('conv_loc', [3, 3, net.get_shape()[3].value, num_loc_pred],
                            initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
        loc_predictions = tf.nn.conv2d(net, filter_loc, padding="SAME", strides=[1, 1, 1, 1])
        loc_predictions = utils.channel_to_last(loc_predictions)
        loc_predictions = tf.reshape(loc_predictions, utils.tensor_shape(loc_predictions, 4)[:-1]+[num_anchors, 4])

        # Class prediction - Return nxnx(number_classes) tensor
        num_class_pred = num_anchors * self.number_classes
        filter_class = tf.get_variable('conv_class', [3, 3, net.get_shape()[3].value, num_class_pred],
                            initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
        class_predictions = tf.nn.conv2d(net, filter_class, padding="SAME", strides=[1, 1, 1, 1])
        class_predictions = utils.channel_to_last(class_predictions)
        class_predictions = tf.reshape(class_predictions, utils.tensor_shape(class_predictions, 4)[:-1]+[num_anchors, self.number_classes])

        return class_predictions, loc_predictions

    def loss_function(self, gt_localizations, gt_classes, overall_predictions, overall_anchors, ratio_negatives=3):
        """ Define the loss function for SSD - Classification + Localization """
        overall_loss = 0
        positive_loss = 0
        negative_loss = 0
        loc_loss = 0
        for index, (predictions, anchors) in enumerate(zip(overall_predictions, overall_anchors)):
            target_labels_all = []
            target_localizations_all = []
            target_scores_all = []
            for batch_index in range(self.batch_size):
                target_tensor = utils.ssd_bboxes_encode_layer(gt_classes[batch_index], gt_localizations[batch_index], anchors, self.number_classes)
                target_labels_all.append(target_tensor[0])
                target_localizations_all.append(target_tensor[1])
                target_scores_all.append(target_tensor[2])
            target_labels = tf.stack(target_labels_all)
            target_localizations = tf.stack(target_localizations_all)
            target_scores = tf.stack(target_scores_all)

            # Determine the Positive and Negative Samples
            pos_samples = tf.cast(target_scores > self.positive_threshold, tf.uint16)
            num_pos_samples = tf.reduce_sum(pos_samples)
            neg_samples = tf.cast(target_scores < self.negative_threshold, tf.float32)
            num_neg_samples = tf.reduce_sum(neg_samples)

            target_labels_flattened = tf.reshape(target_labels, [-1])
            predictions_flattened = tf.reshape(predictions[0], [-1, self.number_classes])
            pos_samples_flattened = tf.cast(tf.reshape(pos_samples, [-1]), tf.float32)
            neg_samples_flattened = tf.cast(tf.reshape(neg_samples, [-1]), tf.float32)

            # Construct the loss function
            with tf.name_scope('cross_entropy_pos{}'.format(index)):
                loss_pos = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=predictions_flattened, labels=target_labels_flattened)
                positives_only_loss = tf.reduce_sum(loss_pos * pos_samples_flattened)
                positives_only_loss = tf.Print(positives_only_loss, [positives_only_loss], "-> Positives Loss")
                loss_classification_pos = tf.div(positives_only_loss, self.batch_size)

            with tf.name_scope('cross_entropy_neg{}'.format(index)):
                loss_neg = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=predictions_flattened, labels=tf.cast(pos_samples_flattened, tf.int64))
                num_hard_negatives = tf.cast(num_pos_samples * ratio_negatives, tf.int32) + self.batch_size
                _, indices_hnm = tf.nn.top_k(loss_neg * neg_samples_flattened, num_hard_negatives, name='hard_negative_mining')
                negatives_only_loss = tf.reduce_sum(tf.gather(loss_neg, indices_hnm))
                loss_classification_neg = tf.div(negatives_only_loss, self.batch_size)
                loss_classification_neg = tf.Print(loss_classification_neg, [loss_classification_neg], "-> Negatives Loss")

            with tf.name_scope('localization{}'.format(index)):
                weights = tf.expand_dims(1.0 * tf.to_float(tf.reshape(pos_samples, [self.batch_size, self.feat_shapes[index][0],
                self.feat_shapes[index][1], len(self.anchor_sizes[index]) * (len(self.anchor_ratios[index]) + 1)])), axis=-1)
                loss = tf.abs(predictions[1] - target_localizations)
                loss_localization = tf.div(tf.reduce_sum(loss * weights), self.batch_size)
                loss_localization = tf.Print(loss_localization, [loss_localization], "-> Localization Loss")

            overall_loss += ((loss_classification_pos + loss_classification_neg + loss_localization) / (tf.cast(num_pos_samples, tf.float32) + 1e-8))
            positive_loss += (loss_classification_pos / (tf.cast(num_pos_samples, tf.float32) + 1e-8))
            negative_loss += (loss_classification_neg / (tf.cast(num_pos_samples, tf.float32) + 1e-8))
            loc_loss += (loss_localization / (tf.cast(num_pos_samples, tf.float32) + 1e-8))

        return overall_loss, positive_loss, negative_loss, loc_loss

    def construct_backbone_architecture(self, x_train, is_training, endpoints):
        """ This function construct the backbone architecture to get detection feature maps """
        with tf.variable_scope("FirstStageFeatureExtractor") as scope:
            out_1 = self._conv2d(x_train, [7, 7, 3, 64], [64], [1, 2, 2, 1], 'conv3x3', is_training)
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

    def detection_layers(self, endpoints):
        """ This function perform detections on the desired feature maps """
        overall_predictions = []
        overall_anchors = []
        for index, layer in enumerate(self.feature_layers):
            with tf.variable_scope("PredictionModule{}".format(index)):
                overall_anchors.append(self.ssd_anchor_box_encoder(index))
                overall_predictions.append(self.detection_layer(endpoints[layer], index))

        return overall_predictions, overall_anchors
