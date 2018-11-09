import tensorflow as tf
import numpy as np
import utils
import pdb
import math
import glob
import argparse
import os

from detectors import SSDResNet
from resnet import ResNet50
from performance_evaluation import compute_metrics

import ROIPooler
import SRGAN

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


def get_parser():
    """ This function returns a parser object """
    aparser = argparse.ArgumentParser()
    aparser.add_argument('--test_frequency', type=int, default=10,
                         help='After every provided number of iterations the model will be test')
    aparser.add_argument('--train_dir', type=str,
                         help='Provide the training directory to the text file with file names and labels in it')
    aparser.add_argument('--test_dir', type=str,
                         help='Provide the test directory to the text file with file names and labels in it')
    aparser.add_argument('--summary_dir', type=str,
                         help='Provide the directory to save operation summaries')
    aparser.add_argument('--ckpt_dir', type=str, default="checkpoints/",
                         help='Provide the checkpoint directory where the network parameters will be stored')
    aparser.add_argument('--run', type=str,
                         help='Name of the run')

    return aparser


def main():
    img_shape = (224, 224)
    new_shape = (30, 30)
    batch_size = (10)
    learning_rate = 3 * 1e-4
    label_map = {'max': 0}
    number_iterations = 100

    # Parse the command line args
    args = get_parser().parse_args()

    # Construct the Graph
    # x_train = tf.placeholder(tf.float32, [batch_size, img_shape[0], img_shape[1], 3])
    # gt_classes = [tf.placeholder(tf.int64, [None]) for _ in range(batch_size)]
    # gt_bboxes = [tf.placeholder(tf.float32, [None, 4]) for _ in range(batch_size)]
    # is_training = tf.placeholder(tf.bool)
    # global_step = tf.train.global_step()

    # Construct the Graph
    gt_rois = tf.placeholder(tf.float32, [None, new_shape[0], new_shape[1], 3])
    gt_rois_class = tf.placeholder(tf.int64, [None])
    gt_rois_bboxes = tf.placeholder(tf.float32, [None, 4])
    is_training = tf.placeholder(tf.bool)
    global_step = tf.train.global_step()

    # Heat map TODO
    # heat_map_model = HeatMapModel()
    # heat_map = heat_map_model.output(x_train)
    # heat_map_loss = heat_map_model.loss(heat_map, gt_bboxes)
    # tf.summary.image("Heat Map", heat_map, max_outputs=3)
    # tf.summary.scalar("Heat_map_loss", heat_map_loss)

    # ROI Pooling TODO
    # roi_pooler = ROIPooler()
    # pr_rois, loc_pr_rois = roi_pooler.extract(x_train, heat_map)
    # gt_rois, loc_gt_rois, gt_rois_class = roi_pooler.manual_pooling(x_train, gt_classes, gt_bboxes)
    # neg_rois, loc_neg_rois = roi_pooler.neg_extract(x_train, loc_gt_rois, number = 3)

    # rois = tf.cond(is_training, gt_rois, pr_rois)

    # SRGAN TODO
    srgan = SRGAN()
    lr_rois = srgan.downsample(gt_rois) #tf.cond(is_training, rois, tf.concat([rois, neg_rois], axis=0))
    hr_rois = gt_rois
    sr_rois = srgan.generator(lr_rois, is_training=is_training)

    tf.summary.image("Small images", lr_rois, max_output=3)
    tf.summary.image("Ground truth", gt_rois, max_output = 3)
    tf.summary.image("Super resolution", sr_rois, max_output=3)

    real_logits, hr_pr_class, hr_pr_bbox = srgan.discriminate(hr_rois, is_training=is_training)
    fake_logits, sr_pr_class, sr_pr_bbox = srgan.discriminate(sr_rois, is_training=is_training)

    with tf.namescope("loss_function"):
        # loss_mse = tf.losses.mse_error(hr_rois, sr_rois) #Should we keep this? Not adversarial
        dis_loss_real = tf.losses.sigmoid_cross_entropy(tf.ones_like(real_logits), real_logits)
        dis_loss_fake = tf.losses.sigmoid_cross_entropy(tf.zeros_like(fake_logits), fake_logits)
        gen_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(fake_logits), fake_logits)

        # Classifying loss
        cls_loss = tf.losses.sigmoid_cross_entropy(gt_rois_class, hr_pr_class)
        # Localisation loss TODO
        #loc_loss = localization_loss(hr_pr_bbox, gt_rois_bboxes, gt_rois_class)

        dis_loss = cls_loss + dis_loss_fake + dis_loss_real #+ loc_loss

        total_loss = gen_loss + dis_loss #+ heat_map_loss

    # Define Optimizer

    with tf.name_scope('optimizers'):
        # control op dependencies for batch norm and trainable variables
        hmvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                   scope='heat_map')
        dvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                  scope='discriminator')
        gvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                  scope='generator')

        update_ops_hm = tf.get_collection(tf.GraphKeys.UPDATE_OPS,
                                          scope='heat_map')

        update_ops_gen = tf.get_collection(tf.GraphKeys.UPDATE_OPS,
                                           scope='generator')
        update_ops_dis = tf.get_collection(tf.GraphKeys.UPDATE_OPS,
                                           scope='discriminator')

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                           beta1=0.5)

        with tf.control_dependencies(update_ops_gen):
            gen_op = optimizer.minimize(gen_loss, var_list=gvars,
                                        global_step=global_step)

        with tf.control_dependencies(update_ops_dis):
            dis_op = optimizer.minimize(dis_loss, var_list=dvars)

        #with tf.control_dependencies(update_ops_hm):
        #    hm_op = optimizer.minimize(heat_map_loss, var_list=hmvars)

    tf.summary.scalar('total_loss', total_loss)
    tf.summary.scalar('positives_loss', cls_loss)
    tf.summary.scalar('generator_loss', gen_loss)
    tf.summary.scalar('discriminator_loss', dis_loss)
    #tf.summary.scalar('localization_loss', loc_loss)

    # Decode predictions to the image domain TODO
    #eval_scores, eval_bboxes = utils.decode_predictions(overall_predictions, overall_anchors, net.number_classes,
    #                                                    tf.constant([0, 0, 1, 1], tf.float32), net.select_threshold)

    # Overlay the bounding boxes on the images
    #tf_image_overlaid_detected = utils.overlay_bboxes_eval(eval_scores, eval_bboxes, x_train)
    #tf_image_overlaid_gt = utils.overlay_bboxes_ground_truth(gt_classes, gt_bboxes, x_train, net.batch_size)
    #tf.summary.image("Detected Bounding Boxes", tf_image_overlaid_detected, max_outputs=3)
    #tf.summary.image("Ground Truth Bounding Boxes", tf_image_overlaid_gt, max_outputs=3)
    merged = tf.summary.merge_all()

    # Execute the graph
    img_names = glob.glob('{}/{}'.format(args.train_dir, '*.jpeg'))
    img_names = np.array(img_names)
    np.random.shuffle(img_names)
    with tf.train.MonitoredTrainingSession(checkpoint_dir=args.ckpt_dir, summary_dir=args.summary_dir) as sess:
        train_writer = tf.summary.FileWriter(os.path.join(args.summary_dir, args.run, "train"), sess.graph)
        test_writer = tf.summary.FileWriter(args.summary_dir, sess.graph)
        for epoch_id in range(0, number_iterations):
            for iteration_id in range(0, len(img_names), batch_size):
                try:
                    img_tensor, gt_bbox_tensor, gt_class_tensor = utils.batch_reader(img_names, iteration_id,
                                                                                     label_map, img_shape,
                                                                                     batch_size)
                    roi_pooler = ROIPooler()
                    rois_tensor, rois_class_tensor, rois_bboxes_tensor = roi_pooler.np_manual_pooling(img_tensor,
                                                                                                      gt_class_tensor,
                                                                                                      gt_bbox_tensor)
                    feed_dict = {is_training: True, gt_rois: rois_tensor, gt_rois_class: rois_class_tensor,
                                 gt_rois_bboxes: rois_bboxes_tensor}
                    #Updating
                     _, gen_loss_value = sess.run([gen_op, gen_loss], feed_dict=feed_dict)
                    summary, _, dis_loss_value, loss_value = sess.run([merged, dis_op, dis_loss], feed_dict=feed_dict)
                    train_writer.add_summary(summary, epoch_id * len(img_names) + iteration_id / batch_size)
                    #precision, recall = compute_metrics(bboxes_pr, scores_pr, gt_bbox_tensor, gt_class_tensor)
                    print("Loss at iteration {} {} : {} - gen:{} - dis:{}".format(epoch_id, iteration_id / batch_size, loss_value, gen_loss_value, dis_loss_value))
                    #print('Training Accuracy at epoch {} iteration {} at %50 IOU : {},  Recall at %50 IOU : {}'.format(
                    #    epoch_id, iteration_id, precision[50], recall[50]))
                except Exception as error:
                    print(error)
                    continue
            # feed_dict.update({is_training: False})
            # summary, loss_value, scores_pr, bboxes_pr = sess.run([merged, total_loss, eval_scores, eval_bboxes],
            #                                                      feed_dict=feed_dict)
            # test_writer.add_summary(summary, epoch_id * len(img_names) + iteration_id / net.batch_size)
            # precision, recall = compute_metrics(bboxes_pr, scores_pr, gt_bbox_tensor, gt_class_tensor)
            # print(
            #     'Validation Accuracy at epoch {} iteration {} at %50 IOU : {},  Recall at %50 IOU : {}'.format(epoch_id,
            #                                                                                                    iteration_id,
            #                                                                                                    precision[
            #                                                                                                        50],
            #                                                                                                    recall[
            #                                                                                                        50]))


if __name__ == '__main__':
    main()
