import numpy as np
from .utils import scalar_square_add_single
from scipy import stats
class associate_pair():
    def __init__(self, output_stride=8):
        self.output_stride = output_stride

    def associate_score(self, annotations, aaa,scoreth=0.1):
        v_scale = 0.5
        for iters in range(2):
            for ann in annotations[iters]:
                ann.data[:, 0:2] *= self.output_stride
                if ann.joint_scales is not None:
                    ann.joint_scales *= self.output_stride

            # nms
            annotations0 = self.soft_nms(annotations[iters])

            # threshold results
            keypoint_sets, scores = [], []
            for ann in annotations0:
                score = ann.score()
                if score < scoreth:
                    continue
                kps = ann.data
                kps[:, 2] *= 2.0 / v_scale
                kps[kps[:, 2] < 0] = 0.0

                keypoint_sets.append(kps)
                scores.append(score)
            if iters == 0:
                keypoint_sets_l = np.array(keypoint_sets)
            else:
                keypoint_sets_r = np.array(keypoint_sets)
        pairdict,non_pair = self.max_score_match(keypoint_sets_l, keypoint_sets_r, aaa)
        return pairdict,non_pair

    def vote_bestmatch(self, keypoint1, keycandidate, keypoint_sets_r):
        index_list = [0, 0]
        b_connection_f = []
        for iters, key_r in enumerate(keypoint_sets_r):
            if key_r[key_r[:, 0] > 0][:, 0].mean() > keypoint1[keypoint1[:, 0] > 0][:, 0].mean():
                continue
            if keypoint1[keypoint1[:, 0] > 0].shape[0] < 4:
                continue
            index = np.array([list(range(17))]).T
            n1 = keypoint1[keypoint1[:, 0] > 0]
            n2 = key_r[keypoint1[:, 0] > 0]
            index = index[keypoint1[:, 0] > 0]
            n1 = n1[n2[:, 0] > 0]
            index = index[n2[:, 0] > 0]
            n2 = n2[n2[:, 0] > 0]
            ttt = abs(n1[:, 1] - n2[:, 1])
            ### filter non-level index
            height = keypoint1[keypoint1[:, 0] > 0][:, 1].max() - keypoint1[keypoint1[:, 0] > 0][:, 1].min()
            if height > 110:
                if ttt.mean() > 6:  ## 6
                    continue
            elif height > 60:
                if ttt.mean() > 3:  ## 3
                    continue
            else:
                if ttt.mean() > 2:  ## 2
                    continue
            score_pair = 0
            b_connection = {}
            for i, joint in enumerate(n2):
                score_max = 0
                for connection in keycandidate[index[i][0]]:
                    if len(connection) != 0:
                        score = connection[0] * (np.exp(-1.0 * (
                        ((n1[i][0] - connection[4]) / self.output_stride) ** 2 + 3 * (
                        (n1[i][1] - connection[5]) / self.output_stride) ** 2) ** 0.5 / connection[6]) * \
                                                 np.exp(-1.0 * (
                                                 ((joint[0] - connection[1]) / self.output_stride) ** 2 + 3 * (
                                                 (joint[1] - connection[2]) / self.output_stride) ** 2) ** 0.5 /
                                                        connection[3]))
                    else:
                        b_connection[int(index[i][0])] = []
                    if score > score_max:
                        b_connection[int(index[i][0])] = [score]
                        score_max = score
                score_pair += score_max
            score = score_pair
            if index_list[1] < score:
                index_list = [iters, score]
                b_connection_f = b_connection
        if index_list[1] == 0:
            return [[], keypoint1, 0, 0,None]
        return [keypoint_sets_r[index_list[0]], keypoint1, index_list, b_connection_f,self.depth_calculation(keypoint1,keypoint_sets_r[index_list[0]])]

    def max_score_match(self, keypoint_sets_l, keypoint_sets_r, aaa):
        pairdict = {}
        pair_index = {}
        non_pair = {}
        numss = 0
        for keys in keypoint_sets_l:
            connection_cand = {}
            for j in range(len(keys)):
                ### aaa is the association candidate

                candidate_trans = aaa[j][1:3].T - keys[j][:2] / self.output_stride
                candidate_trans = candidate_trans[aaa[j][4, :].T - keys[j][0] / self.output_stride < 0]
                ### radius = 5
                connection_cand[j] = aaa[j].T[aaa[j][4, :].T - keys[j][0] / 8 < 0][
                    np.linalg.norm(candidate_trans, axis=1) <= 5]
                connection_cand[j][1:3] *= self.output_stride
                connection_cand[j][4:6] *= self.output_stride
            pair = self.vote_bestmatch(keys, connection_cand, keypoint_sets_r)
            if len(pair[0]) != 0:
                if pair[2][0] in pair_index.keys():
                    if pair[2][1] > pair_index[pair[2][0]]:
                        non_pair[numss] = [pair[1]]
                        numss += 1
                        pairdict[pair[2][0]] = pair
                else:
                    pairdict[pair[2][0]] = pair
                    pair_index[pair[2][0]] = pair[2][1]
            else:
                non_pair[numss] = [pair[1]]
                numss += 1
        return pairdict,non_pair
    def soft_nms(self, annotations):
        if not annotations:
            return annotations

        occupied = np.zeros((
            17,
            int(max(np.max(ann.data[:, 1]) for ann in annotations) + 1),
            int(max(np.max(ann.data[:, 0]) for ann in annotations) + 1),
        ))

        annotations = sorted(annotations, key=lambda a: -a.score())
        for ann in annotations:
            joint_scales = (ann.joint_scales
                            if ann.joint_scales is not None
                            else np.ones((ann.data.shape[0]),) * 4.0)
            for xyv, occ, joint_s in zip(ann.data, occupied, joint_scales):
                ij = np.round(xyv[:2]).astype(np.int)
                i = np.clip(ij[0], 0, occ.shape[1] - 1)
                j = np.clip(ij[1], 0, occ.shape[0] - 1)
                v = xyv[2]
                if occ[j, i]:
                    xyv[2] = 0.0

                if v > 0.0:
                    scalar_square_add_single(occ, xyv[0], xyv[1], joint_s, 1)

        annotations = [ann for ann in annotations if np.any(ann.data[:, 2] > 0.0)]
        annotations = sorted(annotations, key=lambda a: -a.score())
        return annotations

    def depth_calculation(self,left_keypoints, right_keypoints):
        left_keypoints = np.array(left_keypoints)
        right_keypoints = np.array(right_keypoints)
        dif = left_keypoints[:, 0] - right_keypoints[:, 0]
        ## only calculate those joint y != 0
        non1 = np.array(left_keypoints[:, 1])
        non1[non1 != 0] = 1
        non2 = np.array(right_keypoints[:, 1])
        non2[non2 != 0] = 1
        non = non1 * non2
        ## kitti depth
        x_depth = 0.54 * 721/ dif * non
        x_depth[non == 0] = 0
        value = left_keypoints
        value[:, 2] = x_depth
        ## Z score outlier remove
        Z1 = np.copy(value[value[:, 2] > 0])
        if len(Z1) > 0:
            z = np.abs(stats.zscore(Z1[:, 2]))
            dist = np.median(Z1[:, 2][z < 1.414])
            if Z1[:, 2].var() > 200:
                return -1
        else:
            return False
        return [dist,x_depth]