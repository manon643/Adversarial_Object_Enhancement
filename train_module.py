import tensorflow as tf
import numpy as np
import utils
import pdb
import math
import glob
import argparse
import os
from PIL import Image

from detectors import SSDResNet
from resnet import ResNet50
from performance_evaluation import compute_metrics_module as compute_metrics

from ROIPooler import ROIPooler
from SRGAN import SRGAN

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'


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
    aparser.add_argument('--save_test', action="store_true",
                         help='If activate stores images to see progress and training')

    return aparser


def main():
    img_shape = (224, 224)
    new_shape = (256, 256)
    batch_size = (24)
    number_iterations = 1000
    lr_init = 1e-4
    beta_1 = 0.9
    lr_decay = 0.9
    decay_every = 2
    label_map = {'max': 0}

    # Parse the command line args
    args = get_parser().parse_args()

    # Construct the Graph
    # x_train = tf.placeholder(tf.float32, [batch_size, img_shape[0], img_shape[1], 3])
    # gt_classes = [tf.placeholder(tf.int64, [None]) for _ in range(batch_size)]
    # gt_bboxes = [tf.placeholder(tf.float32, [None, 4]) for _ in range(batch_size)]
    # is_training = tf.placeholder(tf.bool)
    # global_step = tf.train.global_step()

    ###TRAINING GRAPH

    # Construct the Graph
    gt_rois = tf.placeholder(tf.float32, [None, new_shape[0], new_shape[1], 3], name="gt_rois")
    gt_rois_class = tf.placeholder(tf.int64, [None], name="gt_classes")
    gt_rois_bboxes = tf.placeholder(tf.float32, [None, 4], name="gt_bboxes")
    is_training = tf.placeholder(tf.bool, name="is_training")
    global_step = tf.train.create_global_step()
    lr_value = tf.Variable(lr_init, trainable=False, name="lr")

    # For testing
    lr_rois_ph = tf.placeholder(tf.float32, [None, new_shape[0]//4, new_shape[1]//4, 3], name="lr_rois")
    
    # Heat map TODO
    # heat_map_model = HeatMapModel()
    # heat_map = heat_map_model.output(x_train)
    # heat_map_loss = heat_map_model.loss(heat_map, gt_bboxes)
    # tf.summary.image("Heat Map", heat_map, max_outputss=3)
    # tf.summary.scalar("Heat_map_loss", heat_map_loss)

    # ROI Pooling TODO
    # roi_pooler = ROIPooler()
    # pr_rois, loc_pr_rois = roi_pooler.extract(x_train, heat_map)
    # gt_rois, loc_gt_rois, gt_rois_class = roi_pooler.manual_pooling(x_train, gt_classes, gt_bboxes)
    # neg_rois, loc_neg_rois = roi_pooler.neg_extract(x_train, loc_gt_rois, number = 3)

    # rois = tf.cond(is_training, gt_rois, pr_rois)

    # SRGAN
    srgan = SRGAN(21)
    down_rois = srgan.downsample(gt_rois, (new_shape[0]//4, new_shape[1]//4)) 
    is_training = tf.Print(is_training, [is_training], "IS_TRAINING=")
    lr_rois = tf.cond(is_training, lambda:down_rois, lambda:lr_rois_ph)
    hr_rois = gt_rois
    sr_rois_train = srgan.generator(lr_rois, is_training=True)
    sr_rois_test = srgan.generator(lr_rois, is_training=False, reuse=True)
    sr_rois = tf.cond(is_training, lambda:sr_rois_train, lambda:sr_rois_test)
    print("lr_rois SHAPE:", lr_rois.shape)
    print("hr_rois SHAPE:", hr_rois.shape)
    print("sr_rois SHAPE:", sr_rois.shape)

    fake_logits, sr_pr_class_train, sr_pr_bbox_train = srgan.discriminator(sr_rois, is_training=True)
    real_logits, hr_pr_class_train, hr_pr_bbox_train = srgan.discriminator(hr_rois, is_training=True, reuse=True)
    _, sr_pr_class_test, sr_pr_bbox_test = srgan.discriminator(sr_rois, is_training=False, reuse=True)
    _, hr_pr_class_test, hr_pr_bbox_test = srgan.discriminator(hr_rois, is_training=False, reuse=True)
    
    hr_pr_class = tf.cond(is_training, lambda:hr_pr_class_train, lambda:hr_pr_class_test)
    sr_pr_class = tf.cond(is_training, lambda:sr_pr_class_train, lambda:sr_pr_class_test)
    hr_pr_bbox = tf.cond(is_training, lambda:hr_pr_bbox_train, lambda:hr_pr_bbox_test)
    sr_pr_bbox = tf.cond(is_training, lambda:sr_pr_bbox_train, lambda:sr_pr_bbox_test)

    with tf.name_scope("loss_function"):
        # loss_mse = tf.losses.mse_error(hr_rois, sr_rois) #Should we keep this? Not adversarial
        dis_loss_real = tf.losses.sigmoid_cross_entropy(tf.ones_like(real_logits), real_logits)
        dis_loss_fake = tf.losses.sigmoid_cross_entropy(tf.zeros_like(fake_logits), fake_logits)
        dis_loss_adv = dis_loss_real + dis_loss_fake
        gen_loss_adv = tf.losses.sigmoid_cross_entropy(tf.ones_like(fake_logits), fake_logits)

        mse_loss = tf.losses.mean_squared_error(labels=gt_rois, predictions=sr_rois)
        # Classifying loss
        cls_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=gt_rois_class, logits=hr_pr_class))
        # Localisation loss TODO
        loc_loss = srgan.localization_loss(sr_pr_bbox, gt_rois_bboxes)
        loc_loss2 = srgan.localization_loss(hr_pr_bbox, gt_rois_bboxes)

        gen_loss = 1e-3*gen_loss_adv + mse_loss
        dis_loss = dis_loss_adv + 100*loc_loss + 0.1*cls_loss

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

        optimizer_gen = tf.train.AdamOptimizer(learning_rate=lr_value,
                                           beta1=beta_1)
        optimizer_gen_init = tf.train.AdamOptimizer(learning_rate=lr_value,
                                           beta1=beta_1)
        optimizer_dis = tf.train.AdamOptimizer(learning_rate=lr_value,
                                           beta1=beta_1)

        with tf.control_dependencies(update_ops_gen):
            gen_op_mse = optimizer_gen_init.minimize(mse_loss, var_list=gvars,
                                            global_step=global_step)
 
        with tf.control_dependencies(update_ops_gen):
            gen_op = optimizer_gen.minimize(gen_loss, var_list=gvars,
                                        global_step=global_step)

        with tf.control_dependencies(update_ops_dis):
            dis_op = optimizer_dis.minimize(dis_loss, var_list=dvars)

        #with tf.control_dependencies(update_ops_hm):
        #    hm_op = optimizer.minimize(heat_map_loss, var_list=hmvars)

    tf.summary.scalar('total_loss', total_loss)
    tf.summary.scalar('positives_loss', cls_loss)
    tf.summary.scalar('generator_loss', gen_loss)
    tf.summary.scalar('mse_loss', mse_loss)
    tf.summary.scalar('discriminator_loss', dis_loss)
    tf.summary.scalar('dis_loss_adv', dis_loss_adv)
    tf.summary.scalar('dis_loss_loc', loc_loss)
    tf.summary.scalar('dis_loss_loc2', loc_loss2)
    
    tf_image_overlaid_detected = utils.overlay_bboxes(sr_pr_bbox, sr_rois)
    tf_image_overlaid_detected_gt = utils.overlay_bboxes(hr_pr_bbox, hr_rois)
    #tf_image_overlaid_detected = utils.adapt_overlay_bboxes_pred(sr_pr_bbox, sr_rois)
    tf_image_overlaid_gt = utils.overlay_bboxes(gt_rois_bboxes, gt_rois)
    tf.summary.image("Detected Bounding Boxes", tf_image_overlaid_detected, max_outputs = 3)
    tf.summary.image("Detected HR BBoxes", tf_image_overlaid_detected_gt, max_outputs = 3)
    tf.summary.image("Ground Truth Bounding Boxes", tf_image_overlaid_gt, max_outputs = 3)
    tf.summary.image("Small images", lr_rois, max_outputs=3)
    tf.summary.image("Z_Bicubic", tf.image.resize_images(lr_rois, new_shape, tf.image.ResizeMethod.BICUBIC), max_outputs=3)

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
    print("Found {} images in {}".format(len(img_names),args.train_dir))
    valid_names = glob.glob('{}/{}'.format(args.test_dir, '*.jpeg'))
    print("Found {} images in {}".format(len(valid_names), args.test_dir))
    test_img = np.random.choice(img_names)
    img_names.remove(test_img)
    train_img = np.random.choice(img_names)
    folder = "poster_images"
    img_names = np.array(img_names)
    #with tf.train.MonitoredTrainingSession(checkpoint_dir=args.ckpt_dir, summary_dir=args.summary_dir) as sess:
    with tf.train.MonitoredTrainingSession(checkpoint_dir=os.path.join(args.ckpt_dir, args.run), save_checkpoint_secs=2*60) as sess:
        train_writer = tf.summary.FileWriter(os.path.join(args.summary_dir, args.run, "train"), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(args.summary_dir, args.run, "test"), sess.graph)
        # Pretrain
        if args.save_test: 
            train_img_tensor, gt_bbox_tensor, gt_class_tensor = utils.batch_reader_list([train_img], 0,
                                                                                       label_map, (224, 224),
                                                                                       1)
            roi_pooler = ROIPooler(train_img_tensor, (new_shape[0], new_shape[1]))
            train_rois_tensor, train_class, train_bbox = roi_pooler.np_manual_pooling(train_img_tensor,
                                                                        gt_class_tensor,
                                                                        gt_bbox_tensor, random=True)
            print(label_map, train_class[0])
            feed_dict_zeros_train = {gt_rois: train_rois_tensor, gt_rois_class: train_class, is_training:True,
                                     gt_rois_bboxes: train_bbox , lr_rois_ph:np.zeros((1,new_shape[0]//4,new_shape[1]//4,3))}
            #Saving full img
            new_name_train = os.path.join(folder, "training_"+train_img.split("/")[-1])[:-5]
            img = Image.fromarray(train_img_tensor[0])
            img.save(new_name_train+".jpeg")
            
            #Saving roi
            roi_img = Image.fromarray(np.uint8(train_rois_tensor[0]*127.5+127.5))
            roi_img.save(new_name_train+"_roi.jpeg")
            
            #Saving sr and lr
            sr_img, lr_img = sess.run([sr_rois, lr_rois], feed_dict = feed_dict_zeros_train) 
            lr_img_ = Image.fromarray(np.uint8(lr_img[0]*127.5+127.5))
            lr_img_.save(new_name_train+"_lr.jpeg")
            sr_img_ = Image.fromarray(np.uint8(sr_img[0]*127.5+127.5))
            sr_img_.save(new_name_train+"_sr.jpeg")

            #Saving test image
            test_img_tensor, gt_bbox_tensor, gt_class_tensor = utils.batch_reader_list([test_img], 0,
                                                                                       label_map, (224, 224),
                                                                                       1)
            roi_pooler_small = ROIPooler(test_img_tensor, (new_shape[0]//4, new_shape[1]//4))
            test_rois_tensor, test_class, _ = roi_pooler_small.np_manual_pooling(test_img_tensor,
                                                                        gt_class_tensor,
                                                                        gt_bbox_tensor, random=False)
            feed_dict_zeros_test = {gt_rois: np.zeros((1,new_shape[0],new_shape[1],3)), gt_rois_class: np.zeros((1)), is_training:False,
                               gt_rois_bboxes: np.zeros((1,4)), lr_rois_ph:test_rois_tensor}
            #Saving full img
            new_name_test = os.path.join(folder, "testing_"+test_img.split("/")[-1])[:-5]
            img = Image.fromarray(test_img_tensor[0])
            img.save(new_name_test+".jpeg")
            
            #Saving roi
            lr_img = Image.fromarray(np.uint8(test_rois_tensor[0]*127.5+127.5))
            lr_img.save(new_name_test+"_lr.jpeg")
            
            #Saving sr
            sr_img = sess.run(sr_rois, feed_dict = feed_dict_zeros_test) 
            sr_img_ = Image.fromarray(np.uint8(sr_img[0]*127.5+127.5))
            sr_img_.save(new_name_test+"_sr.jpeg")
            
        pre_train_epoch = 0#2#10#83#00#number_iterations# 2 #if step else 0
        for epoch_id in range(0, pre_train_epoch):
            np.random.shuffle(img_names)
            for iteration_id in range(0, len(img_names)-batch_size, batch_size):
                #try:
                img_tensor, gt_bbox_tensor, gt_class_tensor = utils.batch_reader_list(img_names, iteration_id,
                                                                                      label_map, img_shape,
                                                                                      batch_size)
                roi_pooler = ROIPooler(img_tensor, new_shape)
                rois_tensor, rois_class_tensor, rois_bboxes_tensor = roi_pooler.np_manual_pooling(img_tensor,
                                                                                                  gt_class_tensor,
                                                                                                  gt_bbox_tensor)
                feed_dict = {gt_rois: rois_tensor, gt_rois_class: rois_class_tensor, is_training:True,
                             gt_rois_bboxes: rois_bboxes_tensor, lr_rois_ph:np.zeros_like(rois_tensor[:,::4,::4,:])}
                #Updating
                summary, _, mse_loss_value, step = sess.run([merged, gen_op_mse, mse_loss, global_step], feed_dict=feed_dict)
                train_writer.add_summary(summary, step)
                print("Loss at pretrain iteration {} {} : {}".format(epoch_id, iteration_id / batch_size, mse_loss_value))
                if epoch_id%1==0:
                    img_tensor, gt_bbox_tensor, gt_class_tensor = utils.batch_reader_list(valid_names, 0,
                                                                                          label_map, img_shape,
                                                                                          50)
                    roi_pooler = ROIPooler(img_tensor, (new_shape[0]//4, new_shape[1]//4))
                    rois_tensor, rois_class_tensor, rois_bboxes_tensor = roi_pooler.np_manual_pooling(img_tensor,
                                                                                                      gt_class_tensor,
                                                                                                      gt_bbox_tensor)
                    print(rois_tensor.shape)
                    feed_dict = {gt_rois: np.zeros((50,new_shape[0],new_shape[1],3)), gt_rois_class: np.zeros((50)), is_training:False,
                               gt_rois_bboxes: np.zeros((50,4)), lr_rois_ph:rois_tensor}
                    #Updating
                    summary, step = sess.run([merged, global_step], feed_dict=feed_dict)
                    test_writer.add_summary(summary, step)
                    precision, recall = compute_metrics(bboxes_pr, scores_pr, rois_bboxes_tensor, rois_class_tensor)
                    print('Validation Accuracy at epoch {} iteration {} at %50 IOU : {},  Recall at %50 IOU : {}'.format(epoch_id, iteration_id, precision[50], recall[50]))
            if args.save_test:
                #Saving sr
                sr_img = sess.run(sr_rois, feed_dict = feed_dict_zeros_test) 
                sr_img_ = Image.fromarray(np.uint8(sr_img[0]*127.5+127.5))
                sr_img_.save(new_name_test+"_{}_sr.jpeg".format(epoch_id))

                #Saving sr and lr
                sr_img = sess.run(sr_rois, feed_dict = feed_dict_zeros_train) 
                sr_img_ = Image.fromarray(np.uint8(sr_img[0]*127.5+127.5))
                sr_img_.save(new_name_train+"_{}_sr.jpeg".format(epoch_id))
        #Training
        for epoch_id in range(0, number_iterations):
            np.random.shuffle(img_names)
        #    if epoch_id != 0 and (epoch_id % decay_every == 0):
        #        new_lr_decay = lr_decay**(epoch_id // decay_every)
        #        sess.run(tf.assign(lr_value, lr_init * new_lr_decay))
        #        log = " ** new learning rate: %f (for GAN)" % (lr_init * new_lr_decay)
        #        print(log)
            for iteration_id in range(0, len(img_names)-batch_size, batch_size):
                #try:
                img_tensor, gt_bbox_tensor, gt_class_tensor = utils.batch_reader_list(img_names, iteration_id,
                                                                                      label_map, img_shape,
                                                                                      batch_size)
                roi_pooler = ROIPooler(img_tensor, new_shape)
                rois_tensor, rois_class_tensor, rois_bboxes_tensor = roi_pooler.np_manual_pooling(img_tensor,
                                                                                                  gt_class_tensor,
                                                                                                  gt_bbox_tensor)
                #print(rois_tensor.shape, rois_class_tensor.shape, rois_bboxes_tensor.shape)
                feed_dict = {gt_rois: rois_tensor, gt_rois_class: rois_class_tensor,is_training:True,
                             gt_rois_bboxes: rois_bboxes_tensor, lr_rois_ph:np.zeros_like(rois_tensor[:,::4,::4,:])}
                #Updating
                for k in range(1):
                    _, dis_loss_value = sess.run([dis_op, dis_loss], feed_dict=feed_dict)
                summary, _, gen_loss_value, loss_value, step = sess.run([merged, gen_op, gen_loss, total_loss, global_step], feed_dict=feed_dict)
                train_writer.add_summary(summary, step)
                bboxes_pr, scores_pr = sess.run([sr_pr_bbox, sr_pr_class], feed_dict=feed_dict)
                precision, recall = compute_metrics(bboxes_pr, scores_pr, rois_bboxes_tensor, rois_class_tensor)
                print("Loss at iteration {} {} : {} - gen:{} - dis:{}".format(epoch_id, iteration_id / batch_size, loss_value, gen_loss_value, dis_loss_value))
                print('Training Accuracy at epoch {} iteration {} at %50 IOU : {},  Recall at %50 IOU : {}'.format(
                    epoch_id, iteration_id, precision[50], recall[50]))
                if epoch_id%2==0:
                    img_tensor, gt_bbox_tensor, gt_class_tensor = utils.batch_reader_list(valid_names, 0,
                                                                                          label_map, img_shape,
                                                                                          50)
                    roi_pooler = ROIPooler(img_tensor, (new_shape[0]//4, new_shape[1]//4))
                    rois_tensor, rois_class_tensor, rois_bboxes_tensor = roi_pooler.np_manual_pooling(img_tensor,
                                                                                                      gt_class_tensor,
                                                                                                      gt_bbox_tensor)
                    print(rois_tensor.shape)
                    feed_dict = {gt_rois: np.zeros((50,new_shape[0],new_shape[1],3)), gt_rois_class: np.zeros((50)), is_training:False,
                               gt_rois_bboxes: np.zeros((50,4)), lr_rois_ph:rois_tensor}
                    #Updating
                    summary, bboxes_pr, scores_pr, step = sess.run([merged, sr_pr_bbox, sr_pr_class, global_step], feed_dict=feed_dict)
                    print(scores_pr.shape)
                    test_writer.add_summary(summary, step)
                    precision, recall = compute_metrics(bboxes_pr, scores_pr, rois_bboxes_tensor, rois_class_tensor)
                    print('Validation Accuracy at epoch {} iteration {} at %50 IOU : {},  Recall at %50 IOU : {}'.format(epoch_id, iteration_id, precision[50], recall[50]))

                #except Exception as error:
                #    print(error)
                #    continue
            if args.save_test:
                #Saving sr
                sr_img = sess.run(sr_rois, feed_dict = feed_dict_zeros_test) 
                sr_img_ = Image.fromarray(np.uint8(sr_img[0]*127.5+127.5))
                sr_img_.save(new_name_test+"_{}_sr_train.jpeg".format(epoch_id))
                #Saving sr and lr
                sr_img = sess.run(sr_rois, feed_dict = feed_dict_zeros_train) 
                sr_img_ = Image.fromarray(np.uint8(sr_img[0]*127.5+127.5))
                sr_img_.save(new_name_train+"_{}_sr_train.jpeg".format(epoch_id))
            

if __name__ == '__main__':
    main()
