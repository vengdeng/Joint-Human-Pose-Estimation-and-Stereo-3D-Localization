import logging
import os
import torch.utils.data
import torchvision
from PIL import Image
import numpy as np
from . import transforms, utils
import copy
import imgaug as ia
from imgaug import augmenters as iaa


def collate_images_anns_meta(batch):
    images = torch.utils.data.dataloader.default_collate([b[0] for b in batch])
    images2 = torch.utils.data.dataloader.default_collate([b[1] for b in batch])
    anns = torch.utils.data.dataloader.default_collate([b[2] for b in batch])
    anns2 = torch.utils.data.dataloader.default_collate([b[3] for b in batch])
    anns3 = torch.utils.data.dataloader.default_collate([b[4] for b in batch])

    metas = [b[5] for b in batch]
    return images,images2, anns,anns2,anns3, metas


def collate_images_targets_meta(batch):
    images = torch.utils.data.dataloader.default_collate([b[0] for b in batch])
    images2 = torch.utils.data.dataloader.default_collate([b[1] for b in batch])
    targets = torch.utils.data.dataloader.default_collate([b[2] for b in batch])
    targets2 = torch.utils.data.dataloader.default_collate([b[3] for b in batch])
    targets3 = torch.utils.data.dataloader.default_collate([b[4] for b in batch])
    metas = [b[5] for b in batch]
    return images,images2, targets,targets2,targets3, metas


class CocoKeypoints(torch.utils.data.Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.

    Based on `torchvision.dataset.CocoDetection`.

    Caches preprocessing.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, root, annFile, augmentation=True,
                 n_images=None, all_images=False, preprocess=None, image_transform=None, target_transforms=None):
        from pycocotools.coco import COCO
        self.root = root
        self.coco = COCO(annFile)
        self.cat_ids = self.coco.getCatIds(catNms=['person'])
        if not all_images:
            self.ids = self.coco.getImgIds(catIds=self.cat_ids)
            self.filter_for_keypoint_annotations()
        else:
            self.ids = self.coco.getImgIds()
        if n_images:
            self.ids = self.ids[:n_images]
        print('Images: {}'.format(len(self.ids)))

        ### preprocess
        self.preprocess = preprocess
        self.image_transform = image_transform
        self.target_transforms = target_transforms

        self.augmentation = augmentation
        if self.augmentation:
            """
                    data augmentation use imgaug library 
                    Flip, rotate, contrastadjust, illumination adjust

                    """
            ia.seed(42)
            ## generate the data for detection
            self.seq = iaa.Sequential([
                ## x level translate
                # iaa.Sometimes(0.5,iaa.Fliplr(0.5, name='Fliplr')),  # horizontal
                # iaa.Sometimes(0.5,iaa.Flipud(0.5, name='Flipud')),  # vertical
                iaa.Sometimes(1, iaa.Affine(scale={"x": (1, 1), "y": (1, 1)},
                                            translate_percent={"x": (0.02, 0.2), "y": (-0, 0)},
                                            name='Affine'))],
                random_order=True)


        self.log = logging.getLogger(self.__class__.__name__)

    def filter_for_keypoint_annotations(self):
        print('filter for keypoint annotations ...')

        def has_keypoint_annotation(image_id):
            ann_ids = self.coco.getAnnIds(imgIds=image_id, catIds=self.cat_ids)
            anns = self.coco.loadAnns(ann_ids)
            for ann in anns:
                if 'keypoints' not in ann:
                    continue
                if any(v > 0.0 for v in ann['keypoints'][2::3]):
                    return True
            return False

        self.ids = [image_id for image_id in self.ids
                    if has_keypoint_annotation(image_id)]
        print('... done.')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        image_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=image_id, catIds=self.cat_ids)
        anns = copy.deepcopy(self.coco.loadAnns(ann_ids))
        image_info = self.coco.loadImgs(image_id)[0]
        self.log.debug(image_info)
        with open(os.path.join(self.root, image_info['file_name']), 'rb') as f:
            image = Image.open(f).convert('RGB')

        pad1 = 0
        pad = ((0, 0), (pad1, pad1), (0, 0))
        # Add padding
        image = np.pad(image, pad, 'constant', constant_values=0)

        meta = {
            'image_id': image_id,
            'file_name': image_info['file_name'],
        }

        #### labels
        num_people = len(anns)
        labels = np.zeros((num_people, 17, 3))
        for i in range(num_people):
            #             labels[i][0] = anns[i]['bbox'][0] - anns[i]['bbox'][2]/2
            #             labels[i][1] = anns[i]['bbox'][1] -anns[i]['bbox'][3]/2
            #             labels[i][2] = anns[i]['bbox'][2]/2+anns[i]['bbox'][0]
            #             labels[i][3] = anns[i]['bbox'][3]/2+anns[i]['bbox'][1]
            anns[i]['bbox'][0] += pad1
            anns[i]['keypoints'][::3] = list(np.array(anns[i]['keypoints'][::3]) + pad1)
            labels[i] = np.array(anns[i]['keypoints']).reshape(17, 3)
        anns_o = copy.deepcopy(anns)
        anns2 = anns.copy()
        ## keypoints augmentation
        keypoints = []
        boxes = []
        for i in range(num_people):
            boxes.append(
                ia.BoundingBox(x1=anns2[i]['bbox'][0], y1=anns2[i]['bbox'][1],
                               x2=anns2[i]['bbox'][2] + anns2[i]['bbox'][0],
                               y2=anns2[i]['bbox'][3] + anns2[i]['bbox'][1])
            )
            for t in range(17):
                keypoints.append(
                    ia.Keypoint(x=labels[i][t][0], y=labels[i][t][1]))

        kps = ia.KeypointsOnImage(keypoints, shape=image.shape)
        bbs = ia.BoundingBoxesOnImage(boxes, shape=image.shape)
        if self.augmentation:
            image2, lbl_a, keypoints_aug, image = self.augment_image(image.astype(np.uint8), bbs, kps)
        # Transform to usable bounding box
        for i in range(num_people):
            ## keypoints
            idxx = i * 17
            ### the copy function doesnt work
            ####
            anns2[i]['keypoints'][::3] = [keypoints_aug.keypoints[idxx].x,
                                          keypoints_aug.keypoints[idxx + 1].x,
                                          keypoints_aug.keypoints[idxx + 2].x,
                                          keypoints_aug.keypoints[idxx + 3].x,
                                          keypoints_aug.keypoints[idxx + 4].x,
                                          keypoints_aug.keypoints[idxx + 5].x,
                                          keypoints_aug.keypoints[idxx + 6].x,
                                          keypoints_aug.keypoints[idxx + 7].x,
                                          keypoints_aug.keypoints[idxx + 8].x,
                                          keypoints_aug.keypoints[idxx + 9].x,
                                          keypoints_aug.keypoints[idxx + 10].x,
                                          keypoints_aug.keypoints[idxx + 11].x,
                                          keypoints_aug.keypoints[idxx + 12].x,
                                          keypoints_aug.keypoints[idxx + 13].x,
                                          keypoints_aug.keypoints[idxx + 14].x,
                                          keypoints_aug.keypoints[idxx + 15].x,
                                          keypoints_aug.keypoints[idxx + 16].x]
            anns2[i]['keypoints'][1::3] = [keypoints_aug.keypoints[idxx].y,
                                           keypoints_aug.keypoints[idxx + 1].y,
                                           keypoints_aug.keypoints[idxx + 2].y,
                                           keypoints_aug.keypoints[idxx + 3].y,
                                           keypoints_aug.keypoints[idxx + 4].y,
                                           keypoints_aug.keypoints[idxx + 5].y,
                                           keypoints_aug.keypoints[idxx + 6].y,
                                           keypoints_aug.keypoints[idxx + 7].y,
                                           keypoints_aug.keypoints[idxx + 8].y,
                                           keypoints_aug.keypoints[idxx + 9].y,
                                           keypoints_aug.keypoints[idxx + 10].y,
                                           keypoints_aug.keypoints[idxx + 11].y,
                                           keypoints_aug.keypoints[idxx + 12].y,
                                           keypoints_aug.keypoints[idxx + 13].y,
                                           keypoints_aug.keypoints[idxx + 14].y,
                                           keypoints_aug.keypoints[idxx + 15].y,
                                           keypoints_aug.keypoints[idxx + 16].y]
            ## x_l,y_l,w,h

            anns2[i]['bbox'][0] = lbl_a.bounding_boxes[i].x1
            anns2[i]['bbox'][1] = lbl_a.bounding_boxes[i].y1
            anns2[i]['bbox'][2] = lbl_a.bounding_boxes[i].x2 - lbl_a.bounding_boxes[i].x1
            anns2[i]['bbox'][3] = lbl_a.bounding_boxes[i].y2 - lbl_a.bounding_boxes[i].y1
        # if there are not target transforms, done here
        # if 'flickr_url' in image_info:
        #     _, flickr_file_name = image_info['flickr_url'].rsplit('/', maxsplit=1)
        #     flickr_id, _ = flickr_file_name.split('_', maxsplit=1)
        #     meta['flickr_full_page'] = 'http://flickr.com/photo.gne?id={}'.format(flickr_id)

        # preprocess image and annotations
        image = Image.fromarray(np.uint8(image))
        image2 = Image.fromarray(np.uint8(image2))

        image, image2, anns_o, anns2, preprocess_meta = self.preprocess(image, image2, anns_o, anns2)
        meta.update(preprocess_meta)

        # transform image
        original_size = image.size
        image = self.image_transform(image)
        image2 = self.image_transform(image2)
        print(image2)
        # mask valid
        #valid_area = meta['valid_area']
        #utils.mask_valid_image(image, valid_area)
        #utils.mask_valid_image(image2, valid_area)

        self.log.debug(meta)
        if self.target_transforms is None:
            return image, image2, anns_o, anns2, meta

        # transform targets
        nums = 0
        targets = []
        targets2 = []
        targets3 = []
        # print(self.target_transforms)
        for t in self.target_transforms:

            nums+=1
            if nums !=3:
                targets.append(t(anns_o, original_size))
                targets2.append(t(anns2, original_size))
            else:
                targets3.append(t(anns_o,anns2,original_size))
        return image, image2, targets, targets2,targets3, meta

    def __len__(self):
        return len(self.ids)

    def augment_image(self, img, lbl, kps):
        ## augment image to simulate the matching situation
        seq_det = self.seq.to_deterministic()  # seq_det is now a fixes sequence, so lbl and deck get treated the same as img
        # it will be a new one after each call of seq.to_deterministic()
        img1 = seq_det.augment_image(img)
        new = seq_det.augment_keypoints([kps])[0]
        new2 = seq_det.augment_bounding_boxes([lbl])[0]
        img2 = copy.deepcopy(img)
        return img1, new2, new, img2


class ImageList(torch.utils.data.Dataset):
    def __init__(self, image_paths, image_transform=None, orimage_transform=None):
        self.image_paths = image_paths
        self.image_transform = image_transform or transforms.image_transform
        self.image_transform1 = orimage_transform

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        with open(image_path, 'rb') as f:
            image = Image.open(f).convert('RGB')
        if self.image_transform1 is not None:
            original_image = self.image_transform1(image)
        original_image = torchvision.transforms.functional.to_tensor(original_image)

        image = self.image_transform(image)

        return image_path, original_image, image

    def __len__(self):
        return len(self.image_paths)
