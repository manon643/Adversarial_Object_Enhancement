import tensorflow as tf
#import tensorlayer as tl
import numpy as np
from utils import pixel_shuffler

class SRGAN:

    def __init__(self, n_classes):
        self.n_classes = n_classes
        self._delta = 1.0
        return

    def downsample(self, hr_img, new_size):
        return tf.image.resize_images(hr_img, new_size)

    def generator(self, lr_img, is_training=False, reuse=False):
        """ Generator in Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network
            feature maps (n) and stride (s) feature maps (n) and stride (s)

            Input:
            lr_img : [batch_size, lr_size, lr_size, 3]
         """
        w_init = tf.random_normal_initializer(stddev=0.02)
        b_init = None  # tf.constant_initializer(value=0.0)
        g_init = tf.random_normal_initializer(1., 0.02)
        with tf.variable_scope("generator", reuse=reuse) as vs:
            # tl.layers.set_name_reuse(reuse) # remove for TL 1.8.0+
            n = lr_img
            n = tf.layers.conv2d(n, 64, (3, 3), (1, 1), activation=tf.nn.relu, padding='SAME', kernel_initializer=w_init, name='n64s1/c')
            temp = n

            # B residual blocks
            for i in range(16):
                net = tf.layers.conv2d(n, 64, (3, 3), (1, 1), activation=None, padding='SAME', kernel_initializer=w_init, bias_initializer=b_init,
                               name='n64s1/c1/%s' % i)
                net = tf.layers.batch_normalization(net, training=is_training, gamma_initializer=g_init, name='n64s1/b1/%s' % i)
                net = tf.nn.relu(net)
                net = tf.layers.conv2d(net, 64, (3, 3), (1, 1), activation=None, padding='SAME', kernel_initializer=w_init, bias_initializer=b_init,        name='n64s1/c2/%s' % i)
                net = tf.layers.batch_normalization(net, training=is_training, gamma_initializer=g_init, name='n64s1/b2/%s' % i)
                net = tf.nn.relu(net)
                net += n
                n = net

            n = tf.layers.conv2d(n, 64, (3, 3), (1, 1), activation=None, padding='SAME', kernel_initializer=w_init, bias_initializer=b_init,
                    name='n64s1/c/m')
            n = tf.layers.batch_normalization(n, training=is_training, gamma_initializer=g_init, name='n64s1/b/m')
            n = tf.nn.relu(n)
            n += temp
            # B residual blacks end

            n = tf.layers.conv2d(n, 256, (3, 3), (1, 1), activation=None, padding='SAME', kernel_initializer=w_init, name='n256s1/1')
            n = pixel_shuffler(n, scale=2, channels=256, activation=tf.nn.relu, name='pixelshufflerx2/1')

            n = tf.layers.conv2d(n, 256, (3, 3), (1, 1), activation=None, padding='SAME', kernel_initializer=w_init, name='n256s1/2')
            n = pixel_shuffler(n, scale=2, channels=256, activation=tf.nn.relu, name='pixelshufflerx2/2')

            sr_img = tf.layers.conv2d(n, 3, (1, 1), (1, 1), activation=tf.nn.tanh, padding='SAME', kernel_initializer=w_init, name='out')

            return sr_img

    def discriminator(self, input_images, is_training=True, reuse=False):
        w_init = tf.random_normal_initializer(stddev=0.02)
        b_init = None  # tf.constant_initializer(value=0.0)
        gamma_init = tf.random_normal_initializer(1., 0.02)
        df_dim = 64
        lrelu = lambda x: tf.nn.leaky_relu(x, 0.2)
        with tf.variable_scope("discriminator", reuse=reuse):
            #tllayers.set_name_reuse(reuse)
            net_in = input_images
            net = tf.layers.conv2d(net_in, df_dim, (4, 4), (2, 2), activation=lrelu, padding='SAME', kernel_initializer=w_init, name='h0/c')

            net = tf.layers.conv2d(net, df_dim * 2, (4, 4), (2, 2), activation=None, padding='SAME', kernel_initializer=w_init,
                 bias_initializer=b_init, name='h1/c')
            net = tf.layers.batch_normalization(net, training=True, gamma_initializer=gamma_init, name='h1/bn')
            net = lrelu(net)
            net = tf.layers.dropout(net, rate=0.2, training=is_training)
            net = tf.layers.conv2d(net, df_dim * 4, (4, 4), (2, 2), activation=None, padding='SAME', kernel_initializer=w_init,
                 bias_initializer=b_init, name='h2/c')
            net = tf.layers.batch_normalization(net, training=True, gamma_initializer=gamma_init, name='h2/bn')
            net = lrelu(net)
            net = tf.layers.dropout(net, rate=0.2, training=is_training)
            net = tf.layers.conv2d(net, df_dim * 8, (4, 4), (2, 2), activation=None, padding='SAME', kernel_initializer=w_init,
                 bias_initializer=b_init, name='h3/c')
            net = tf.layers.batch_normalization(net, training=True, gamma_initializer=gamma_init, name='h3/bn')
            net = lrelu(net)
            net = tf.layers.dropout(net, rate=0.2, training=is_training)
            net = tf.layers.conv2d(net, df_dim * 16, (4, 4), (2, 2), activation=None, padding='SAME', kernel_initializer=w_init,
                 bias_initializer=b_init, name='h4/c')
            net = tf.layers.batch_normalization(net, training=True, gamma_initializer=gamma_init, name='h4/bn')
            net = lrelu(net)
            net = tf.layers.dropout(net, rate=0.2, training=is_training)
            net = tf.layers.conv2d(net, df_dim * 32, (4, 4), (2, 2), activation=None, padding='SAME', kernel_initializer=w_init,
                 bias_initializer=b_init, name='h5/c')
            net = tf.layers.batch_normalization(net, training=True, gamma_initializer=gamma_init, name='h5/bn')
            net = lrelu(net)
            net = tf.layers.dropout(net, rate=0.2, training=is_training)
            net = tf.layers.conv2d(net, df_dim * 16, (1, 1), (1, 1), activation=None, padding='SAME', kernel_initializer=w_init,
                 bias_initializer=b_init, name='h6/c')
            net = tf.layers.batch_normalization(net, training=True, gamma_initializer=gamma_init, name='h6/bn')
            net = lrelu(net)
            net = tf.layers.dropout(net, rate=0.2, training=is_training)
            net = tf.layers.conv2d(net, df_dim * 8, (1, 1), (1, 1), activation=None, padding='SAME', kernel_initializer=w_init,
                 bias_initializer=b_init, name='h7/c')
            net = tf.layers.batch_normalization(net, training=True, gamma_initializer=gamma_init, name='h7/bn')
            net_h7 = net
            net = tf.layers.dropout(net, rate=0.2, training=is_training)

            net = tf.layers.conv2d(net, df_dim * 2, (1, 1), (1, 1), activation=None, padding='SAME', kernel_initializer=w_init, bias_initializer=b_init,
                    name='res/c')
            net = tf.layers.batch_normalization(net, training=True, gamma_initializer=gamma_init, name='res/bn')
            net = lrelu(net)
            net = tf.layers.conv2d(net, df_dim * 2, (3, 3), (1, 1), activation=None, padding='SAME', kernel_initializer=w_init, bias_initializer=b_init,
                    name='res/c2')
            net = tf.layers.batch_normalization(net, training=True, gamma_initializer=gamma_init, name='res/bn2')
            net = lrelu(net)
            net = tf.layers.conv2d(net, df_dim * 8, (3, 3), (1, 1), activation=None, padding='SAME', kernel_initializer=w_init, bias_initializer=b_init,
                    name='res/c3')
            net = tf.layers.batch_normalization(net, training=True, gamma_initializer=gamma_init, name='res/bn3')

            ### Branch 1
            net_h8 = net_h7 + net
            net_h8 = lrelu(net_h8)

            net_ho = tf.layers.flatten(net_h8, name='ho/flatten')
            net_ho = tf.layers.dense(net_ho, units=1, activation=None, kernel_initializer=w_init, name='ho/dense')
            logits = net_ho

            ### Branch 2
            net_b2 = tf.layers.flatten(net_h8, name='b2/flatten')
            net_b2 = tf.layers.dense(net_b2, units=self.n_classes+4, activation=tf.sigmoid, kernel_initializer=w_init, name='b2/dense')
            ### Branch 3
            #net_b3 = tf.layers.flatten(net_h8, name='b3/flatten')
            #net_b3 = tf.layers.dense(net_b3, units=4, activation=tf.sigmoid, kernel_initializer=w_init, name='b3/dense')
            # Summary
            b2_dense_weights = tf.get_default_graph().get_tensor_by_name('discriminator/b2/dense/kernel:0')
            tf.summary.histogram("b2_dense_w", b2_dense_weights)
            #b3_dense_weights = tf.get_default_graph().get_tensor_by_name('discriminator/b3/dense/kernel:0')
            #tf.summary.histogram("b3_dense_w", b3_dense_weights)
            # Splitting
            img_class = net_b2[:, 4:]
            img_bbox  = net_b2[:, :4]

        return logits, img_class, img_bbox

    def _smooth_l1(self, prediction_tensor, target_tensor, weights=1.0):
        """Compute loss function.
        Args:
          prediction_tensor: A float tensor of shape [batch_size, 4] representing the (encoded) predicted locations of objects.
          target_tensor: A float tensor of shape [batch_size, 4] representing the targets
        Returns:
          loss: a float tensor of shape [batch_size, num_anchors] tensor
            representing the value of the loss function.
        """
        return tf.reduce_sum(tf.losses.huber_loss(
            target_tensor,
            prediction_tensor,
            delta=self._delta,
            weights=weights,
            loss_collection=None,
            reduction=tf.losses.Reduction.NONE
        ), axis=1)

    def localization_loss(self, predictions, targets, gt_classes):
        # regress the log width and height
        #new_targets_list = [targets[:, 0], targets[:, 1], tf.log(targets[:, 2] - targets[:, 0]), tf.log(targets[:, 3] - targets[:, 1])]
        pos_samples = tf.expand_dims(tf.to_float(tf.math.greater(gt_classes, 0)), axis=-1)
        return tf.reduce_mean(self._smooth_l1(predictions, targets, weights=pos_samples))
