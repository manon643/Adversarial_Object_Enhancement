import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
from skimage.transform import resize



class ROIPooler:
    anchor_size = [(35, 35), (40, 35), (35, 40)]
    new_size = 40

    def __init__(self, imgs):
        self.mask = np.zeros_like(imgs)
        self.number_pos = np.zeros(imgs.shape[0])
        return

    def set_anchor_sizes_from_data(self):
        raise NotImplementedError

    def np_manual_pooling(self, imgs, classes, bboxes):
        """
        :param imgs: batch_size, h, w, c
        :param classes: list of [None]
        :param bboxes: list of [None, 4]
        :return:
        pooled_images, [new_batch_size, new_size, new_size, c]
        new_classes, [new_batch_size]
        new_bbox, [new_batch_size, 4]
        """
        print(imgs.shape, len(classes), len(bboxes))
        pooled_imgs, new_classes, new_bboxes = [], [], []
        batch_size = imgs.shape[0]
        for id_img in range(batch_size):
            print(len(classes[id_img]), len(bboxes[id_img]))
            img = imgs[id_img]
            for j, obj_class in enumerate(classes[id_img]):
                obj_bbox = bboxes[id_img][j]
                centroid_x = int((obj_bbox[0] + obj_bbox[2]) / 2 * img.shape[1])
                centroid_y = int((obj_bbox[1] + obj_bbox[3]) / 2 * img.shape[2])

                for anchor_w, anchor_h in self.anchor_size:
                    min_x = max(0, centroid_x - anchor_w // 2)
                    min_y = max(0, centroid_y - anchor_h // 2)
                    max_x = min(img.shape[1], centroid_x + anchor_w // 2)
                    max_y = min(img.shape[2], centroid_y + anchor_h // 2)
                    cropped_img = img[min_x:max_x, min_y:max_y, :]
                    self.mask[id_img, min_x:max_x, min_y:max_y, :] = 1
                    #print(cropped_img.type)                             
                    img_chip = resize(cropped_img, (self.new_size, self.new_size), anti_aliasing=True)
                    pooled_imgs.append(img_chip)

                    new_classes.append(obj_class)

                    new_min_x = max(0, obj_bbox[0] * img.shape[1] - min_x) // anchor_w
                    new_min_y = max(0, obj_bbox[1] * img.shape[2] - min_y) // anchor_h
                    new_max_x = min(1, obj_bbox[2] * img.shape[1] - min_x) // anchor_w
                    new_max_y = min(1, obj_bbox[3] * img.shape[2] - min_y) // anchor_h

                    new_bboxes.append([new_min_x, new_min_y, new_max_x, new_max_y])

                    self.number_pos[id_img] += 1
        print(np.sum(self.number_pos))
        print(len(pooled_imgs), len(new_classes), len(new_bboxes))
        print(pooled_imgs[0].shape)
        return np.stack(pooled_imgs), np.array(new_classes), np.stack(new_bboxes)

    def neg_pooling(self, imgs, n=2, overlap_cond=0.5):
        neg_pooled_imgs = []
        for id_img, img in enumerate(imgs):
            i = 0
            flat_mask = self.mask[id_img].reshape((-1))
            while i < (n * self.number_pos[id_img]):
                anchor_w, anchor_h = np.random.choice(self.anchor_size)
                centroid_id = np.random.choice(flat_mask.shape[0],
                                               p=flat_mask / flat_mask.sum())
                centroid_x = centroid_id // self.mask[id_img].shape[0]
                centroid_y = centroid_id % self.mask[id_img].shape[0]

                min_x = max(0, centroid_x - anchor_w // 2)
                min_y = max(0, centroid_y - anchor_h // 2)
                max_x = min(img.shape[1], centroid_x + anchor_w // 2)
                max_y = min(img.shape[2], centroid_y + anchor_h // 2)
                overlap = self.mask[id_img, min_x:max_x, min_y:max_y, :].sum() / (anchor_w * anchor_h)
                if overlap > overlap_cond:
                    continue

                cropped_img = img[min_x:max_x, min_y:max_y, :]
                neg_pooled_imgs.append(cropped_img)
                i += 1
        return neg_pooled_imgs
