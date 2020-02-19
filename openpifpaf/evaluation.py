import os
import torch, json
from scipy import stats
import numpy as np
from .utils import jaccard

class Evaluator():
    def __init__(self, result_path, GT_path, pose_quality=False, distance=False, difficulty_L=False):
        self.r_path = result_path
        self.g_path = GT_path
        json_files = [pos_json for pos_json in os.listdir(result_path) if
                      pos_json.endswith('.txt') and pos_json.startswith('0')]
        json_files.sort()
        self.file_name = json_files
        self.r_path = result_path
        ## initial the
        self.easy = []
        self.difficult = []
        self.moderate = []
        self.miss = 0
        self.pose_quality = pose_quality
        self.num_05 = 0
        self.num1 = 0
        self.num2 = 0
        self.num_total = 2446
        self.num_detect = 0
        self.distance = distance
        if difficulty_L:
            self.alecal = self.ale_loz
        else:
            self.alecal = self.ale
        if pose_quality:
            self.quality = {'close': {'correct': 0, 'error': 0}, 'middle': {'correct': 0, 'error': 0},
                            'far': {'correct': 0, 'error': 0}}

    def run(self, iou_th=0.2, z_th=1.5):
        for index_f in range(len(self.file_name)):
            boxes = []
            val_gt = self.g_path + self.file_name[index_f].split('.')[0] + '.txt'
            labelL = self.labelread(val_gt)
            ## Groudtruth loading
            Label_l = []

            for label in labelL:
                ## filter non person groundtruth
                if label['type'] == 'Pedestrian' or label['type'] == 'Person_sitting':
                    Label_l.append(label)
            ### GT box for matching with predicted keypoints
            if len(Label_l) != 0:
                for label_l in Label_l:
                    boxes.append([label_l['bbox']['left'], label_l['bbox']['top'], label_l['bbox']['right'],
                                  label_l['bbox']['bottom']])
                boxes = torch.FloatTensor(boxes).view(len(Label_l), -1)
            with open(self.r_path + self.file_name[index_f]) as foo_file:
                data = foo_file.read()
                json_load = json.loads(data)

                dict_result = {}
                ## build dict to remove duplicate
                self.index_list = {}
                for key in json_load.keys():
                    pairs = np.array(json_load[str(key)])
                    pairs[0] = np.array(pairs[0])
                    pairs[1] = np.array(pairs[1])
                    depths = self.depth_calculation(pairs[1], pairs[0])
                    if depths == -1:
                        self.miss += 1
                        continue
                    dist, Z1, x_depth = depths
                    keypoints = pairs[1]
                    box = self.keypoint_box(keypoints)
                    IOU = jaccard(box, boxes)
                    if IOU.max() > iou_th:
                        self.num_detect += 1
                        index = IOU.argmax()
                        diff = Z1[:, 2].max() - Z1[:, 2].min()
                        ## 3d pose quality checking
                        if self.pose_quality:
                            self.pose_quality(dist, diff)
                        if self.distance:
                            diffs = np.abs(np.sqrt(
                                dist ** 2 + Label_l[index]['location']['x'] ** 2 + Label_l[index]['location'][
                                    'y'] ** 2) -
                                           np.sqrt(Label_l[index]['location']['z'] ** 2 + Label_l[index]['location'][
                                               'x'] ** 2 + Label_l[index]['location']['y'] ** 2))
                        else:
                            diffs = np.abs(dist - Label_l[index]['location']['z'])
                            ## different keypoints matched the same groundtruth box
                            ## reserve the smallest error pair
                        if int(index) in self.index_list.keys():
                            ## if have a better detection, remove the former.
                            self.num_detect -= 1
                            if diffs < self.index_list[int(index)][0]:
                                self.duplicate_remove(index)
                                self.alecal(Label_l, index, diffs)

                        else:
                            self.alecal(Label_l, index, diffs)

                        ###
                        self.alp(diffs)
                    else:
                        self.miss += 1
            ##
            for key in self.index_list.keys():
                if self.index_list[key][1] == 0:
                    self.easy.append(self.index_list[key][0])
                if self.index_list[key][1] == 2:
                    self.difficult.append(self.index_list[key][0])
                if self.index_list[key][1] == 1:
                    self.moderate.append(self.index_list[key][0])

    def duplicate_remove(self, index):
        if self.index_list[int(index)][0] < 0.5:
            self.num_05 -= 1
        if self.index_list[int(index)][0] < 1:
            self.num1 -= 1
        if self.index_list[int(index)][0] < 2:
            self.num2 -= 1

    def keypoint_box(self, keypoints):
        ## increase box size
        ys = keypoints[:, 1][keypoints[:, 1] != 0]
        xs = keypoints[:, 0][keypoints[:, 1] != 0]
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        w = (x_max - x_min) / 5
        h = (y_max - y_min) / 10
        ## increase box size
        box = torch.FloatTensor([x_min - w, y_min - h, x_max + w, y_max + h]).view(1, 4)
        return box

    def pose_quality(self, dist, diff):

        if dist <= 10:
            if diff < 2.5:
                self.quality['close']['correct'] += 1
            else:
                self.quality['close']['error'] += 1
        elif 10 < dist <= 20:
            if diff < 3:
                self.quality['middle']['correct'] += 1
            else:
                self.quality['middle']['error'] += 1
        else:
            if diff < 3.5:
                self.quality['far']['correct'] += 1
            else:
                self.quality['far']['error'] += 1

    def alp(self, diffs):
        ## average location iter
        if diffs < 0.5:
            self.num_05 += 1
        if diffs < 1:
            self.num1 += 1
        if diffs < 2:
            self.num2 += 1

    def ale_loz(self, Label_l, index, diffs):
        #  Easy: Min. bounding box height: 40 Px, Max. occlusion level: Fully visible, Max. truncation: 15 %
        #  Moderate: Min. bounding box height: 25 Px, Max. occlusion level: Partly occluded, Max. truncation: 30 %
        # Hard: Min. bounding box height: 25 Px, Max. occlusion level: Difficult to see, Max. truncation: 50 %
        if Label_l[index]['occluded'] == 0 and Label_l[index]['truncated'] <= 0.15 and (
            Label_l[index]['bbox']['bottom'] - \
                Label_l[index]['bbox']['top']) >= 40:
            self.index_list[int(index)] = [diffs, 0]
        elif Label_l[index]['occluded'] <= 1 and Label_l[index]['truncated'] <= 0.3:
            self.index_list[int(index)] = [diffs, 1]
        else:
            self.index_list[int(index)] = [diffs, 2]

    def ale(self, Label_l, index, diffs):
        #  Easy: Min. bounding box height: 40 Px, Max. occlusion level: Fully visible, Max. truncation: 15 %
        #  Moderate: Min. bounding box height: 25 Px, Max. occlusion level: Partly occluded, Max. truncation: 30 %
        # Hard: Min. bounding box height: 25 Px, Max. occlusion level: Difficult to see, Max. truncation: 50 %
        if Label_l[index]['occluded'] == 0:
            if Label_l[index]['truncated'] <= 0.15 and Label_l[index]['bbox']['bottom'] - \
                    Label_l[index]['bbox']['top'] >= 40:
                self.index_list[int(index)] = [diffs, 0]
            else:
                self.index_list[int(index)] = [diffs, -1]
        elif Label_l[index]['occluded'] == 1:
            if Label_l[index]['truncated'] <= 0.3 and Label_l[index]['bbox']['bottom'] - \
                    Label_l[index]['bbox']['top'] >= 25:
                self.index_list[int(index)] = [diffs, 1]
            else:
                self.index_list[int(index)] = [diffs, -1]
        elif Label_l[index]['occluded'] == 2:
            if Label_l[index]['truncated'] <= 0.5 and Label_l[index]['bbox']['bottom'] - \
                    Label_l[index]['bbox']['top'] >= 25:
                self.index_list[int(index)] = [diffs, 2]
            else:
                self.index_list[int(index)] = [diffs, -1]
        else:
            self.index_list[int(index)] = [diffs, -1]

    def depth_calculation(self, left_keypoints, right_keypoints):
        left_keypoints = np.array(left_keypoints)
        right_keypoints = np.array(right_keypoints)
        dif = left_keypoints[:, 0] - right_keypoints[:, 0]
        ## only calculate those joint v != 0
        non1 = np.array(left_keypoints[:, 1])
        non1[non1 != 0] = 1
        non2 = np.array(right_keypoints[:, 1])
        non2[non2 != 0] = 1
        non = non1 * non2
        x_depth = 0.54 * 721 / (dif) * non
        x_depth[non == 0] = 0
        value = left_keypoints
        value[:, 2] = x_depth
        Z1 = np.copy(value[value[:, 2] != 0])

        if len(Z1) > 0:
            z = np.abs(stats.zscore(Z1[:, 2]))

            dist = np.median(Z1[:, 2][z < 1.5])
        else:
            self.miss += 1
            return -1

        return dist, Z1, x_depth

    def vals_to_dict(self, vals, keys, vals_n=0):
        out = dict()
        for key in keys:
            if isinstance(key, str):
                try:
                    val = float(vals[vals_n])
                except:
                    val = vals[vals_n]
                data = val
                key_name = key
                vals_n += 1
            else:
                data, vals_n = self.vals_to_dict(vals, key[1], vals_n)
                key_name = key[0]
            out[key_name] = data
            if vals_n >= len(vals):
                break
        return out, vals_n

    def labelread(self, file_path):
        bbox = ('bbox', ['left', 'top', 'right', 'bottom'])
        dimensions = ('dimensions', ['height', 'width', 'length'])
        location = ('location', ['x', 'y', 'z'])
        keys = ['type', 'truncated', 'occluded', 'alpha', bbox,
                dimensions, location, 'rotation_y', 'score']
        labels = list()
        for line in open(file_path, 'r'):
            vals = line.split()
            l, _ = self.vals_to_dict(vals, keys)
            labels.append(l)
        return labels