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


#from tensorflow.python import debug as tf_debug

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def get_parser():
    """ This function returns a parser object """
    aparser = argparse.ArgumentParser()
    aparser.add_argument('--test_frequency', type=int, default=10,
                         help='After every provided number of iterations the model will be test')
    aparser.add_argument('--train_dir', type=str,
                         help='Provide the training directory to the text file with file names and labels in it')
    aparser.add_argument('--test_dir', type=str,
                     help='Provide the test directory to the text file with file names and labels in it')
    aparser.add_argument('--train_summary_dir', type=str,
                     help='Provide the directory to save train operation summaries')
    aparser.add_argument('--test_summary_dir', type=str,
                     help='Provide the directory to save test operation summaries')
    aparser.add_argument('--ckpt_dir', type=str,
                     help='Provide the checkpoint directory where the network parameters will be stored')

    return aparser

def main():

    # Parse the command line args
    args = get_parser().parse_args()

    # Construct the Graph
    net = SSDResNet()
    endpoints = {}
    x_train = tf.placeholder(tf.float32, [net.batch_size, net.img_shape[0], net.img_shape[1], 3])
    x_names = tf.placeholder(tf.string, [net.batch_size])
    is_training = tf.placeholder(tf.bool)
    
    endpoints = net.construct_backbone_architecture(x_train, is_training, endpoints)
    overall_predictions, overall_anchors = net.detection_layers(endpoints)

    # Construct the Loss Function and Define Optimizer
    gt_classes = [tf.placeholder(tf.int64, [None]) for _ in range(net.batch_size)]
    gt_bboxes = [tf.placeholder(tf.float32, [None, 4]) for _ in range(net.batch_size)]
    total_loss, positives_loss, negatives_loss, localization_loss = net.loss_function(gt_bboxes, gt_classes, overall_predictions, overall_anchors, net.negatives_ratio, x_names)
    optimizer = tf.train.AdamOptimizer(net.learning_rate)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(total_loss)
    tf.summary.scalar('total_loss', total_loss)
    tf.summary.scalar('positives_loss', positives_loss)
    tf.summary.scalar('negatives_loss', negatives_loss)
    tf.summary.scalar('localization_loss', localization_loss)

    # Decode predictions to the image domain
    eval_scores, eval_bboxes = utils.decode_predictions(overall_predictions, overall_anchors, net.number_classes, tf.constant([0, 0, 1, 1], tf.float32), net.select_threshold)

    # Overlay the bounding boxes on the images
    tf_image_overlaid_detected = utils.overlay_bboxes_eval(eval_scores, eval_bboxes, x_train)
    tf_image_overlaid_gt = utils.overlay_bboxes_ground_truth(gt_classes, gt_bboxes, x_train, net.batch_size)
    tf.summary.image("Detected Bounding Boxes", tf_image_overlaid_detected, max_outputs = 20)
    tf.summary.image("Ground Truth Bounding Boxes", tf_image_overlaid_gt, max_outputs = 20)
    merged = tf.summary.merge_all()

    # Execute the graph
    img_names = glob.glob(os.path.join(args.train_dir, '*.jpeg'))
    #DEBUG
    print("Loaded {} images from {}".format(len(img_names),os.path.join(args.train_dir, '*.jpeg')))
    img_names = np.array(img_names)
    np.random.shuffle(img_names)
    with tf.Session() as sess:
        #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        train_writer = tf.summary.FileWriter(args.train_summary_dir, sess.graph)
        test_writer = tf.summary.FileWriter(args.test_summary_dir, sess.graph)
        sess.run(tf.global_variables_initializer())
        for epoch_id in range(0, net.number_iterations):
            for iteration_id in range(0, len(img_names), net.batch_size):
                try:
                    print("ITERATION ID", iteration_id)
                    batch_names = img_names[iteration_id:iteration_id+net.batch_size]
                    img_tensor, gt_bbox_tensor, gt_class_tensor = utils.batch_reader(img_names, iteration_id, net.label_map, net.img_shape, net.batch_size)
                    #print("ok")
                    
                    feed_dict = {is_training: True, x_train: img_tensor, x_names:batch_names}
                    for batch_ind in range(net.batch_size):
                        feed_dict.update({gt_bboxes[batch_ind]: gt_bbox_tensor[batch_ind], gt_classes[batch_ind]: gt_class_tensor[batch_ind]})
                    #print("ok2")
                    summary, _, loss_value, bboxes_pr, scores_pr = sess.run([merged, train_op, total_loss, eval_bboxes, eval_scores], feed_dict=feed_dict)
                    #print("ok3")
                    train_writer.add_summary(summary, epoch_id * len(img_names) + iteration_id/net.batch_size)
                    #print("ok4")
                    precision, recall = compute_metrics(bboxes_pr, scores_pr, gt_bbox_tensor, gt_class_tensor)
                    print("Loss at iteration {} {} : {}".format(epoch_id, iteration_id/net.batch_size, loss_value))
                    print('Training Accuracy at epoch {} iteration {} at %50 IOU : {},  Recall at %50 IOU : {}'.format(epoch_id, iteration_id, precision[50], recall[50]))
                except Exception as error:
                    print(error)
                    continue 

            feed_dict.update({is_training: False})
            summary, loss_value, scores_pr, bboxes_pr = sess.run([merged, total_loss, eval_scores, eval_bboxes], feed_dict=feed_dict)
            test_writer.add_summary(summary, epoch_id * len(img_names) + iteration_id/net.batch_size)
            precision, recall = compute_metrics(bboxes_pr, scores_pr, gt_bbox_tensor, gt_class_tensor)
            print('Validation Accuracy at epoch {} iteration {} at %50 IOU : {},  Recall at %50 IOU : {}'.format(epoch_id, iteration_id, precision[50], recall[50]))

if __name__ == '__main__':
    main()
