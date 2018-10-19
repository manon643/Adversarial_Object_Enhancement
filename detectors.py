import tensorflow as tf
import numpy as np
import utils
import math

from resnet import ResNet50

class SSDResNet(ResNet50):
    """ This class contains the components of the SSD with the ResNet backbone architecture """
    feature_layers = ['block3', 'block4', 'block5']

    def __init__(self):
        """ Constructor for the SSD-resnet Model """
        self.number_classes = 21 #9 # +1 for background class
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
        self.batch_size = 2#24
        self.number_iterations_dataset = 1000
        self.buffer_size = 1000
        self.positive_threshold = 0.5
        self.negative_threshold = 0.5
        self.select_threshold = 0.5
        self.negatives_ratio = 3
        self.learning_rate = 1e-3
        self.label_map = {'max': 0}

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

    def loss_function(self, gt_localizations, gt_classes, overall_predictions, overall_anchors, ratio_negatives=3, names=None):
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
            target_labels_flattened = tf.Print(target_labels_flattened, [names[0]], message="NAME:")
            predictions_flattened = tf.reshape(predictions[0], [-1, self.number_classes])
            pos_samples_flattened = tf.cast(tf.reshape(pos_samples, [-1]), tf.float32)
            neg_samples_flattened = tf.cast(tf.reshape(neg_samples, [-1]), tf.float32)

            # Construct the loss function
            with tf.name_scope('cross_entropy_pos{}'.format(index)):
                loss_pos = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=predictions_flattened, labels=target_labels_flattened)
                positives_only_loss = tf.reduce_sum(loss_pos * pos_samples_flattened)
                positives_only_loss = tf.Print(positives_only_loss, [index, positives_only_loss], "-> Positives Loss")
                loss_classification_pos = tf.div(positives_only_loss, self.batch_size)

            with tf.name_scope('cross_entropy_neg{}'.format(index)):
                loss_neg = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=predictions_flattened, labels=tf.cast(pos_samples_flattened, tf.int64))
                num_hard_negatives = tf.cast(num_pos_samples * ratio_negatives, tf.int32) + self.batch_size
                _, indices_hnm = tf.nn.top_k(loss_neg * neg_samples_flattened, num_hard_negatives, name='hard_negative_mining')
                negatives_only_loss = tf.reduce_sum(tf.gather(loss_neg, indices_hnm))
                loss_classification_neg = tf.div(negatives_only_loss, self.batch_size)
                loss_classification_neg = tf.Print(loss_classification_neg, [index, loss_classification_neg], "-> Negatives Loss")

            with tf.name_scope('localization{}'.format(index)):
                weights = tf.expand_dims(1.0 * tf.to_float(tf.reshape(pos_samples, [self.batch_size, self.feat_shapes[index][0],
                self.feat_shapes[index][1], len(self.anchor_sizes[index]) * (len(self.anchor_ratios[index]) + 1)])), axis=-1)
                loss = tf.abs(predictions[1] - target_localizations)
                loss_localization = tf.div(tf.reduce_sum(loss * weights), self.batch_size)
                loss_localization = tf.Print(loss_localization, [index, loss_localization], "-> Localization Loss")

            overall_loss += ((loss_classification_pos + loss_classification_neg + loss_localization) / (tf.cast(num_pos_samples, tf.float32) + 1e-8))
            positive_loss += (loss_classification_pos / (tf.cast(num_pos_samples, tf.float32) + 1e-8))
            negative_loss += (loss_classification_neg / (tf.cast(num_pos_samples, tf.float32) + 1e-8))
            loc_loss += (loss_localization / (tf.cast(num_pos_samples, tf.float32) + 1e-8))
        return overall_loss, positive_loss, negative_loss, loc_loss

    def detection_layers(self, endpoints):
        """ This function perform detections on the desired feature maps """
        overall_predictions = []
        overall_anchors = []
        for index, layer in enumerate(self.feature_layers):
            with tf.variable_scope("PredictionModule{}".format(index)):
                overall_anchors.append(self.ssd_anchor_box_encoder(index))
                overall_predictions.append(self.detection_layer(endpoints[layer], index))
        return overall_predictions, overall_anchors
