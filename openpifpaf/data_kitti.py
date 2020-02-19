
import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
import logging
import utils

import json



def labelloader(path):
    with open(path) as foo_file:
        data = foo_file.read()
        json_load = json.loads(data)
        datas = []
        segments_x = []
        segments_y = []
        for key in json_load.keys():
            if 'ignore' in json_load[key]:
                segments_x.append([json_load[key]['ignore'][0],json_load[key]['ignore'][2],
                                   json_load[key]['ignore'][2],json_load[key]['ignore'][0]])
                segments_y.append([json_load[key]['ignore'][1],json_load[key]['ignore'][1],
                                   json_load[key]['ignore'][3],json_load[key]['ignore'][3]])
            else:
                json_load[key]['trackID'] = int(key)
                datas.append(json_load[key])
    tt = np.array(segments_x)
    tt[tt<0] = 0
    segments_x = list(tt)
    return datas,segments_x ,segments_y



IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def default_loader(path):
    return Image.open(path).convert('RGB')


def collate_images_anns_meta(batch):
    images = torch.utils.data.dataloader.default_collate([b[0] for b in batch])
    images2 = torch.utils.data.dataloader.default_collate([b[1] for b in batch])
    anns = torch.utils.data.dataloader.default_collate([b[2] for b in batch])
    anns2 = torch.utils.data.dataloader.default_collate([b[3] for b in batch])
    anns3 = torch.utils.data.dataloader.default_collate([b[4] for b in batch])

    metas = [b[5] for b in batch]
    return images, images2, anns, anns2, anns3, metas


def collate_images_targets_meta(batch):
    images = torch.utils.data.dataloader.default_collate([b[0] for b in batch])
    images2 = torch.utils.data.dataloader.default_collate([b[1] for b in batch])
    targets = torch.utils.data.dataloader.default_collate([b[2] for b in batch])
    targets2 = torch.utils.data.dataloader.default_collate([b[3] for b in batch])
    targets3 = torch.utils.data.dataloader.default_collate([b[4] for b in batch])
    metas = [b[5] for b in batch]
    return images, images2, targets, targets2, targets3, metas


class myImageFloder(data.Dataset):
    def __init__(self, left, right, left_label, right_label, n_images=None, all_images=False, loader=default_loader,
                 dataloader=labelloader
                 , preprocess=None, image_transform=None, target_transforms=None):

        self.left = left
        self.right = right
        self.label_L = left_label
        self.label_R = right_label
        self.loader = loader
        self.dataloader = dataloader
        if not all_images:
            self.filter_for_keypoint_annotations()
            self.left = [left[i] for i in self.index]
            self.right = [right[i] for i in self.index]
            self.label_L = [left_label[i] for i in self.index]
            self.label_R = [right_label[i] for i in self.index]
        if n_images:
            self.left = self.left[:n_images]
            self.right = self.right[:n_images]
            self.label_L = self.label_L[:n_images]
            self.label_R = self.label_R[:n_images]
        self.preprocess = preprocess
        self.image_transform = image_transform
        self.target_transforms = target_transforms

        self.log = logging.getLogger(self.__class__.__name__)

    def filter_for_keypoint_annotations(self):
        print('filter for keypoint annotations ...')

        def has_keypoint_annotation(image_id):
            with open(image_id) as foo_file:
                data = foo_file.read()
                anns = json.loads(data)
            for ann in anns.keys():
                if 'keypoints' not in anns[ann]:
                    continue
                if any(v > 0.0 for v in anns[ann]['keypoints'][2::3]):
                    return True
            return False

        self.index = [image_id for image_id in range(len(self.label_L))
                      if has_keypoint_annotation(self.label_L[image_id])]
        print('... done.')

    def __getitem__(self, index):
        left = self.left[index]
        right = self.right[index]
        label_L = self.label_L[index]
        label_R = self.label_R[index]

        left_img = self.loader(left)
        right_img = self.loader(right)
        dataL, segments_x_l, segments_y_l = self.dataloader(label_L)
        dataR, segments_x_r, segments_y_r = self.dataloader(label_R)

        ### mask invalid area
        if len(segments_x_l) != 0:
            for region_x, region_y in zip(segments_x_l, segments_y_l):
                if len(region_y) or len(region_x) == 0:
                    continue
                rr, cc = polygon(region_y, region_x, left_img.shape)
                left_img[rr, cc, :] = 0
            for region_x, region_y in zip(segments_x_r, segments_y_r):
                if len(region_y) or len(region_x) == 0:
                    continue
                rr, cc = polygon(region_y, region_x, right_img.shape)
                right_img[rr, cc, :] = 0

        meta = {
            'file_name': left,
        }

        # preprocess image and annotations
        image, image2, anns_o, anns2, preprocess_meta = self.preprocess(left_img, right_img, dataL, dataR)
        meta.update(preprocess_meta)

        # transform image
        original_size = image.size
        image = self.image_transform(image)
        image2 = self.image_transform(image2)

        # mask valid
        valid_area = meta['valid_area']
        utils.mask_valid_image(image, valid_area)
        utils.mask_valid_image(image2, valid_area)

        self.log.debug(meta)
        if self.target_transforms is None:
            return image, image2, anns_o, anns2, meta

        # transform targets
        nums = 0
        targets = []
        targets2 = []
        targets3 = []

        for t in self.target_transforms:
            nums += 1
            if nums != 3:
                targets.append(t(anns2, original_size))
                targets2.append(t(anns2, original_size))
            else:
                targets3.append(t(anns_o, anns2, original_size))
        return image, image2, targets, targets2, targets3, meta

    def __len__(self):
        return len(self.left)
