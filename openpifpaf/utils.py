import math
import numpy as np

from .data import COCO_KEYPOINTS, HFLIP


def horizontal_swap_coco(keypoints):
    target = np.zeros(keypoints.shape)

    for source_i, xyv in enumerate(keypoints):
        source_name = COCO_KEYPOINTS[source_i]
        target_name = HFLIP.get(source_name)
        if target_name:
            target_i = COCO_KEYPOINTS.index(target_name)
        else:
            target_i = source_i
        target[target_i] = xyv

    return target


def mask_valid_image(image, valid_area):
    image[:, :int(valid_area[1]), :] = 0
    image[:, :, :int(valid_area[0])] = 0
    max_i = int(math.ceil(valid_area[1] + valid_area[3]))
    max_j = int(math.ceil(valid_area[0] + valid_area[2]))
    if max_i < image.shape[1]:
        image[:, max_i:, :] = 0
    if max_j < image.shape[2]:
        image[:, :, max_j:] = 0


def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]