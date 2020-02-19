import argparse
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from .dataloader.KITTI2015_loader import KITTI2015, RandomCrop, ToTensor, Normalize, Pad


from .network import nets
from . import decoder
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='PSMNet')
nets.cli(parser)
decoder.cli(parser, instance_threshold=0.05)
parser.add_argument('--maxdisp', type=int, default=192, help='max diparity')
parser.add_argument('--logdir', default='log/runs', help='log directory')
parser.add_argument('--datadir', default='/data/wenlong-data', help='data directory')
parser.add_argument('--cuda', type=int, default=1, help='gpu number')
parser.add_argument('--batch-size', type=int, default=6, help='batch size')
parser.add_argument('--validate-batch-size', type=int, default=1, help='batch size')
parser.add_argument('--log-per-step', type=int, default=1, help='log per step')
parser.add_argument('--save-per-epoch', type=int, default=1, help='save model per epoch')
parser.add_argument('--model-dir', default='checkpoint', help='directory where save model checkpoint')
parser.add_argument('--model-path', default=None, help='path of model to load')
# parser.add_argument('--start-step', type=int, default=0, help='number of steps at starting')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--num-epochs', type=int, default=300, help='number of training epochs')
parser.add_argument('--num-workers', type=int, default=8, help='num workers in loading data')
parser.add_argument('--figure-width', default=10.0, type=float,
                        help='figure width')
parser.add_argument('--dpi-factor', default=1.0, type=float,
                        help='increase dpi of output image by this factor')
# parser.add_argument('--')

args = parser.parse_args()


# imagenet
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
device_ids = [1]

device = torch.device('cuda:0')
print(device)


def main(args):

    train_transform = T.Compose([Normalize(mean, std), ToTensor()])
    train_dataset = KITTI2015(args.datadir, mode='train', transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=1)

    validate_transform = T.Compose([Normalize(mean, std), ToTensor()])
    validate_dataset = KITTI2015(args.datadir, mode='validate', transform=validate_transform)
    validate_loader = DataLoader(validate_dataset, batch_size=1,shuffle =False, num_workers=1)

    step = 0
    ### pose net

    model1, _ = nets.factory(args)
    model1 = model1.to(device)
    processors = decoder.factory(args, model1)
    withperson_l = []
    withperson_r = []

    for batch,idx in train_loader:
        step += 1

        left_img = batch['left'].to(device)
        right_img = batch['right'].to(device)

        #### attention map
        fields_batch = processors[0].fields(left_img)
        fields_batch_r = processors[0].fields(right_img)

        # unbatch
        for processed_image_cpu_l,processed_image_cpu_r, fields,fields_r in zip(
                left_img,
                right_img,
                fields_batch,
                fields_batch_r):

            for processor in processors:
                keypoint_sets, score_maps = processor.keypoint_sets(fields)
                if len(keypoint_sets) != 0 :
                    withperson_l.append(idx)

                keypoint_sets_r, score_maps_r = processor.keypoint_sets(fields_r)
                if len(keypoint_sets_r) != 0 :
                    withperson_r.append(idx)
    with open('left_idx.txt', 'w') as f:
        for item in withperson_l:
            f.write("%s\n" % item)
    with open('right_idx.txt', 'w') as f:
        for item in withperson_r:
            f.write("%s\n" % item)



    for batch,idx in validate_loader:
        step += 1

        left_img = batch['left'].to(device)
        right_img = batch['right'].to(device)

        #### attention map
        fields_batch = processors[0].fields(left_img)
        fields_batch_r = processors[0].fields(right_img)

        # unbatch
        for processed_image_cpu_l,processed_image_cpu_r, fields,fields_r in zip(
                left_img,
                right_img,
                fields_batch,
                fields_batch_r):

            for processor in processors:
                keypoint_sets, score_maps = processor.keypoint_sets(fields)
                if len(keypoint_sets) != 0 :
                    withperson_l.append(idx)

                keypoint_sets_r, score_maps_r = processor.keypoint_sets(fields_r)
                if len(keypoint_sets_r) != 0 :
                    withperson_r.append(idx)
    with open('left_t_idx.txt', 'w') as f:
        for item in withperson_l:
            f.write("%s\n" % item)
    with open('right_t_idx.txt', 'w') as f:
        for item in withperson_r:
            f.write("%s\n" % item)
    print(withperson_r)

if __name__ == '__main__':
    main(args)