import time

import numpy as np
import torch
from .association_pair import associate_pair
class Processor(object):
    def __init__(self, model, decode, *,
                 keypoint_threshold=0.0, instance_threshold=0.1,
                 debug_visualizer=None):
        self.model = model
        self.decode = decode
        self.keypoint_threshold = keypoint_threshold
        self.instance_threshold = instance_threshold
        self.debug_visualizer = debug_visualizer
        self.associate_pair= associate_pair()
    def set_cpu_image(self, cpu_image, processed_image):
        if self.debug_visualizer is not None:
            self.debug_visualizer.set_image(cpu_image, processed_image)

    def fields(self, image_batch):
        start = time.time()
        with torch.no_grad():
            outputs = self.model(image_batch)
        heads_o = outputs
        heads = heads_o[:2]
        # to numpy
        fields = [[field.cpu().detach().numpy() for field in head] for head in heads]
        # index by batch entry
        fields = [
            [[field[i] for field in head] for head in fields]
            for i in range(image_batch.shape[0])]
        outputs[2][0] = torch.sigmoid(outputs[2][0])
        connections = [outputs[2]]
        # to numpy
        fields1 = [[field.cpu().detach().numpy() for field in head] for head in connections]
        # index by batch entry
        fields1 = [
            [[field[i] for field in head] for head in fields1]
            for i in range(image_batch.shape[0]//2)]
        fields.append(fields1)
        print('nn processing time', time.time() - start)
        return fields
    def keypoint_sets(self, image_batch):
        start = time.time()
        fields = self.fields(image_batch)
        annotations,connection_candidate = self.decode(fields)
        pairs = self.associate_pair.associate_score(annotations, connection_candidate,self.instance_threshold)


        print('total processing time', time.time() - start)
        return pairs
