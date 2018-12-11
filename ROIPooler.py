import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
from skimage.transform import resize



class ROIPooler:
    #anchor_size = [(35, 35), (40, 35), (35, 40)]

    def __init__(self, new_shape):
        self.new_shape = new_shape
        return

    def set_anchor_sizes_from_data(self):
        raise NotImplementedError

    def np_manual_pooling(self, imgs, classes, bboxes, random=True):
        indexes = list(range(len(imgs)))
        pos_ind = [i for i in indexes if np.random.random()<=1.0/3.0]
        neg_ind = [i for i in indexes if i not in pos_ind]
        #return self.pos_pooling(imgs, classes, bboxes, pos_ind, random=random)
        pos_samples, pos_classes, pos_bboxes = self.pos_pooling(imgs, classes, bboxes, pos_ind, random=random)
        self.compute_mask_neg(imgs, bboxes, neg_ind)
        #return self.neg_pooling(imgs, neg_ind)
        neg_samples, neg_classes, neg_bboxes = self.neg_pooling(imgs, neg_ind)
       #print(neg_samples.shape, pos_samples.shape)
        samples = np.concatenate([pos_samples, neg_samples], axis=0)
        classes = np.concatenate([pos_classes, neg_classes], axis=0)
        bboxes = np.concatenate([pos_bboxes, neg_bboxes], axis=0)
        np.random.shuffle(indexes)
        return samples[indexes], classes[indexes], bboxes[indexes]

    def compute_mask_neg(self, imgs, bboxes, neg_pos):
        self.mask = []
        for id_img in neg_pos:
            img = imgs[id_img]
            mask_ = np.zeros_like(img)
            for j in range(len(bboxes[id_img])):
                obj_bbox = bboxes[id_img][j]
                max_x = int(obj_bbox[3] * img.shape[1])
                max_y = int(obj_bbox[2] * img.shape[0])
                min_x = int(obj_bbox[1] * img.shape[1])
                min_y = int(obj_bbox[0] * img.shape[0])
                mask_[min_y:max_y, min_x:max_x, :] = 1
            self.mask.append(mask_)

    def pos_pooling(self, imgs, classes, bboxes, pos_ind, random=True):
        """
        :param imgs: batch_size, h, w, c
        :param classes: list of [None]
        :param bboxes: list of [None, 4]
        :return:
        pooled_images, [new_batch_size, new_size, new_size, c]
        new_classes, [new_batch_size]
        new_bbox, [new_batch_size, 4]
        """
        #print(imgs.shape, len(classes), len(bboxes))
        pooled_imgs, new_classes, new_bboxes = [], [], []
        for id_img in pos_ind:
            img = imgs[id_img]
            if random:
                j = np.random.choice(len(classes[id_img]))
            else:
                min_area = 1
                j = 0
                for i, obj_bbox in enumerate(bboxes[id_img]):
                    area = (obj_bbox[2]-obj_bbox[0])*(obj_bbox[3]-obj_bbox[1])
                    if area < min_area:
                        min_area, j = area, i
            obj_class = classes[id_img][j]
            obj_bbox = bboxes[id_img][j]
            centroid_x = int((obj_bbox[1] + obj_bbox[3]) / 2 * img.shape[1])
            centroid_y = int((obj_bbox[0] + obj_bbox[2]) / 2 * img.shape[0])
            offset_x = int(np.random.normal()*10)
            offset_y = int(np.random.normal()*10)
            if centroid_x+offset_x < img.shape[1] and centroid_x+offset_x>=0:
                centroid_x += offset_x
            else:
                offset_x = 0
            if centroid_y+offset_y < img.shape[0] and centroid_y+offset_y>=0:
                centroid_y += offset_y
            else:
                offset_y = 0
            anchor_w, anchor_h = self.new_shape
            min_x = max(0, centroid_x - anchor_h // 2)
            min_y = max(0, centroid_y - anchor_w // 2)
            max_x = min(img.shape[1], centroid_x + anchor_h // 2)
            max_y = min(img.shape[0], centroid_y + anchor_w // 2)
            cropped_img = img[min_y:max_y, min_x:max_x, :]
            try:
               #print(np.min(cropped_img), np.max(cropped_img))
                if cropped_img.shape[:2] != self.new_shape:
                   #print("reshape")
                    img_chip = resize(cropped_img, self.new_shape, anti_aliasing=True)
                else:
                    img_chip = cropped_img/255
               #print(np.min(img_chip), np.max(img_chip))
            except:
               #print(centroid_y, centroid_x)
               #print(img.shape)
               #print(min_y, max_y, min_x, max_x)
                print(cropped_img.shape)
            pooled_imgs.append(img_chip)

            new_classes.append(obj_class)

            new_min_y = max(0, (obj_bbox[0] * img.shape[0] - min_y) / anchor_w)
            new_min_x = max(0, (obj_bbox[1] * img.shape[1] - min_x) / anchor_h)
            new_max_y = min(1, (obj_bbox[2] * img.shape[0] - min_y) / anchor_w)
            new_max_x = min(1, (obj_bbox[3] * img.shape[1] - min_x) / anchor_h)

            new_bboxes.append([new_min_y, new_min_x, new_max_y, new_max_x])
        if pos_ind:
            return 2*np.stack(pooled_imgs)-1, np.array(new_classes), np.stack(new_bboxes)
        return [], [], []

    def neg_pooling(self, imgs, neg_ind, overlap_cond=0.7):
        neg_pooled_imgs = []
        for id_mask, id_img in enumerate(neg_ind):
            img = imgs[id_img]
            i = 0
            while True:
                anchor_w, anchor_h = self.new_shape
                centroid_x = np.random.randint(anchor_h//2, img.shape[1]-anchor_h//2)
                centroid_y = np.random.randint(anchor_w//2, img.shape[0]-anchor_w//2)
               #print(centroid_y, centroid_x)
               #print(img.shape)
                min_x = max(0, centroid_x - anchor_h // 2)
                min_y = max(0, centroid_y - anchor_w // 2)
                max_x = min(img.shape[1], centroid_x + anchor_h // 2)
                max_y = min(img.shape[0], centroid_y + anchor_w // 2)
                cropped_img = img[min_y:max_y, min_x:max_x, :]
               #print(cropped_img.shape)
                overlap = self.mask[id_mask][min_x:max_x, min_y:max_y, :].sum() / (anchor_w * anchor_h)
                if overlap > overlap_cond:
                    i += 1
                    if i%10 == 0:
                       overlap_cond *= 0.9
                    continue

                img_chip = resize(cropped_img, self.new_shape, anti_aliasing=True)
                neg_pooled_imgs.append(img_chip)
                break
        return 2*np.stack(neg_pooled_imgs)-1, np.array([0 for _ in neg_ind]), np.array([[0, 0, 1, 1] for _ in neg_ind])
