"""Predict poses for given images."""

import argparse
import glob
import json
import os
import torchvision
import numpy as np
import torch

from .decoder.processor2 import Processor
from .decoder.pifpafpsf import PifPaf
from .visualize.img_plot import image_plot
from .network import nets
from . import datasets, decoder


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    nets.cli(parser)
    parser.add_argument('images', nargs='*',
                        help='input images')
    parser.add_argument('--glob',
                        help='glob expression for input images (for many images)')
    parser.add_argument('-o', '--output-directory',
                        help=('Output directory. When using this option, make '
                              'sure input images have distinct file names.'))
    parser.add_argument('--loader-workers', default=2, type=int,
                        help='number of workers for data loading')
    parser.add_argument('--disable-cuda',action='store_true',
                        help='disable CUDA')
    parser.add_argument('--figure-width', default=10.0, type=float,
                        help='figure width')
    parser.add_argument('--dpi-factor', default=1.0, type=float,
                        help='increase dpi of output image by this factor')
    parser.add_argument('--pretrained', default=None,
                        help='load a model from a checkpoint')
    parser.add_argument('--Gid', default=0,
                        help='Select GPU ID, currently only support one gpu')
    args = parser.parse_args()

    # glob
    if args.glob:
        args.images += glob.glob(args.glob)
    if not args.images:
        raise Exception("no image files given")

    # add args.device
    args.device = torch.device('cpu')
    pin_memory = False
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda:'+str(args.Gid))
        pin_memory = True

    # load model
    pretrained = torch.load(args.pretrained)['model']
    model = pretrained.to(args.device)
    pifpaf = PifPaf(8, 0.3, force_complete=True)
    processor = Processor(model, pifpaf,
                          instance_threshold=0,
                          keypoint_threshold=0.05)

    normalize = torchvision.transforms.Normalize(  # pylint: disable=invalid-name
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    transformer = torchvision.transforms.Compose([  # pylint: disable=invalid-name
        torchvision.transforms.ToTensor(),
        normalize,
    ])
    transformer1 = torchvision.transforms.Compose([  # pylint: disable=invalid-name
    ])
    # data
    print(args.images)
    data = datasets.ImageList(args.images,image_transform= transformer,orimage_transform= transformer1)
    data_loader = torch.utils.data.DataLoader(
        data, batch_size=2, shuffle=False,
        pin_memory=pin_memory, num_workers=args.loader_workers)


    for image_i, (image_paths, image_tensors, processed_images_cpu) in enumerate(data_loader):

        processed_images = processed_images_cpu.to(args.device, non_blocking=True)
        # unbatch
        image_path = image_paths[0]
        if args.output_directory is None:
            output_path = image_path
        else:
            file_name = os.path.basename(image_path)
            output_path = os.path.join(args.output_directory, file_name)
        print('image', image_i, image_path, output_path)
        ## save the keypoint pairs
        pairs,non_pair = processor.keypoint_sets(processed_images)
        with open(output_path + '.pifpaf.json', 'w') as f:
            f.write(json.dumps(pairs, cls=NumpyEncoder))
        image_plot(pairs, image_tensors[1].permute(1, 2, 0).detach().numpy(),output_path, True)



if __name__ == '__main__':
    main()
