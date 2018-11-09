import tensorflow as tf
import tensorlayer as tl
import numpy as np


class SRGAN:

    def __init__(self):
        return

    def downsample(self, hr_img, new_size):
        return tf.image.resize_images(hr_img, new_size)

    def generator(self, lr_img, is_train=False, reuse=False):
        """ Generator in Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network
            feature maps (n) and stride (s) feature maps (n) and stride (s)

            Input:
            lr_img : [batch_size, lr_size, lr_size, 3]
            """
        w_init = tf.random_normal_initializer(stddev=0.02)
        b_init = None  # tf.constant_initializer(value=0.0)
        g_init = tf.random_normal_initializer(1., 0.02)
        with tf.variable_scope("SRGAN_g", reuse=reuse) as vs:
            # tl.layers.set_name_reuse(reuse) # remove for TL 1.8.0+
            n = tl.InputLayer(lr_img, name='in')
            n = tl.Conv2d(n, 64, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, name='n64s1/c')
            temp = n

            # B residual blocks
            for i in range(16):
                nn = tl.Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                               name='n64s1/c1/%s' % i)
                nn = tl.BatchNormLayer(nn, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='n64s1/b1/%s' % i)
                nn = tl.Conv2d(nn, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                               name='n64s1/c2/%s' % i)
                nn = tl.BatchNormLayer(nn, is_train=is_train, gamma_init=g_init, name='n64s1/b2/%s' % i)
                nn = tl.ElementwiseLayer([n, nn], tf.add, name='b_residual_add/%s' % i)
                n = nn

            n = tl.Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                          name='n64s1/c/m')
            n = tl.BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n64s1/b/m')
            n = tl.ElementwiseLayer([n, temp], tf.add, name='add3')
            # B residual blacks end

            n = tl.Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='n256s1/1')
            n = tl.SubpixelConv2d(n, scale=2, n_out_channel=None, act=tf.nn.relu, name='pixelshufflerx2/1')

            n = tl.Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='n256s1/2')
            n = tl.SubpixelConv2d(n, scale=2, n_out_channel=None, act=tf.nn.relu, name='pixelshufflerx2/2')

            sr_img = tl.Conv2d(n, 3, (1, 1), (1, 1), act=tf.nn.tanh, padding='SAME', W_init=w_init, name='out')

        return sr_img

    def discriminator(self, input_images, is_train=True, reuse=False):
        w_init = tf.random_normal_initializer(stddev=0.02)
        b_init = None  # tf.constant_initializer(value=0.0)
        gamma_init = tf.random_normal_initializer(1., 0.02)
        df_dim = 64
        lrelu = lambda x: tl.act.lrelu(x, 0.2)
        with tf.variable_scope("SRGAN_d", reuse=reuse):
            tl.layers.set_name_reuse(reuse)
            net_in = tl.InputLayer(input_images, name='input/images')
            net_h0 = tl.Conv2d(net_in, df_dim, (4, 4), (2, 2), act=lrelu, padding='SAME', W_init=w_init, name='h0/c')

            net_h1 = tl.Conv2d(net_h0, df_dim * 2, (4, 4), (2, 2), act=None, padding='SAME', W_init=w_init,
                               b_init=b_init, name='h1/c')
            net_h1 = tl.BatchNormLayer(net_h1, act=lrelu, is_train=is_train, gamma_init=gamma_init, name='h1/bn')
            net_h2 = tl.Conv2d(net_h1, df_dim * 4, (4, 4), (2, 2), act=None, padding='SAME', W_init=w_init,
                               b_init=b_init, name='h2/c')
            net_h2 = tl.BatchNormLayer(net_h2, act=lrelu, is_train=is_train, gamma_init=gamma_init, name='h2/bn')
            net_h3 = tl.Conv2d(net_h2, df_dim * 8, (4, 4), (2, 2), act=None, padding='SAME', W_init=w_init,
                               b_init=b_init, name='h3/c')
            net_h3 = tl.BatchNormLayer(net_h3, act=lrelu, is_train=is_train, gamma_init=gamma_init, name='h3/bn')
            net_h4 = tl.Conv2d(net_h3, df_dim * 16, (4, 4), (2, 2), act=None, padding='SAME', W_init=w_init,
                               b_init=b_init, name='h4/c')
            net_h4 = tl.BatchNormLayer(net_h4, act=lrelu, is_train=is_train, gamma_init=gamma_init, name='h4/bn')
            net_h5 = tl.Conv2d(net_h4, df_dim * 32, (4, 4), (2, 2), act=None, padding='SAME', W_init=w_init,
                               b_init=b_init, name='h5/c')
            net_h5 = tl.BatchNormLayer(net_h5, act=lrelu, is_train=is_train, gamma_init=gamma_init, name='h5/bn')
            net_h6 = tl.Conv2d(net_h5, df_dim * 16, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init,
                               b_init=b_init, name='h6/c')
            net_h6 = tl.BatchNormLayer(net_h6, act=lrelu, is_train=is_train, gamma_init=gamma_init, name='h6/bn')
            net_h7 = tl.Conv2d(net_h6, df_dim * 8, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init,
                               b_init=b_init, name='h7/c')
            net_h7 = tl.BatchNormLayer(net_h7, is_train=is_train, gamma_init=gamma_init, name='h7/bn')

            net = tl.Conv2d(net_h7, df_dim * 2, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                            name='res/c')
            net = tl.BatchNormLayer(net, act=lrelu, is_train=is_train, gamma_init=gamma_init, name='res/bn')
            net = tl.Conv2d(net, df_dim * 2, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                            name='res/c2')
            net = tl.BatchNormLayer(net, act=lrelu, is_train=is_train, gamma_init=gamma_init, name='res/bn2')
            net = tl.Conv2d(net, df_dim * 8, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                            name='res/c3')
            net = tl.BatchNormLayer(net, is_train=is_train, gamma_init=gamma_init, name='res/bn3')

            ### Branch 1
            net_h8 = tl.ElementwiseLayer([net_h7, net], combine_fn=tf.add, name='res/add')
            net_h8.outputs = tl.act.lrelu(net_h8.outputs, 0.2)

            net_ho = tl.FlattenLayer(net_h8, name='ho/flatten')
            net_ho = tl.DenseLayer(net_ho, n_units=1, act=tf.identity, W_init=w_init, name='ho/dense')
            logits = net_ho.outputs

            ### Branch 2
            net_b2 = tl.FlattenLayer(net, name='b2/flatten')
            net_b2 = tl.DenseLayer(net_b2, n_units=5, act=tf.sigmoid, W_init=w_init, name='b2/dense')

            # Splitting
            img_class = net_b2.outputs[:, :, :, 0]
            img_bbox = net_b2.outputs[:, :, :, 1:]

        return logits, img_class, img_bbox
