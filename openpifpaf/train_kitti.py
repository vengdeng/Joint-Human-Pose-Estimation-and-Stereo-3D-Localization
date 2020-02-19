"""Train a pifpaf net."""

import argparse
import datetime
import glob
import torch
import os
from . import data_kitti, encoder, logs, optimize, transforms
from .network import losses, nets, Trainer




def dataloader(filepath):
    left_fold = 'image_2/'
    right_fold = 'image_3/'
    box_l = 'label_2/'
    path = '/data/wenlong-data/kitti_keypoins/kitti_left_kp'
    files = []
    with open("val.txt") as f:
        files_val = [line.split()[0] for line in f]
    for filename in glob.glob(os.path.join(path, '*.txt')):
        files.append(filename)
    box_val_L = files
    path2 = '/data/wenlong-data/kitti_keypoins/kitti_right_kp/'

    # left_dir = join(filepath, 'image_2')
    val_image = []
    val_label = []
    index_val = []
    index_train = []
    num = 0
    #     print(files_val)
    for i in files:
        name = i.split('/')[-1].split('.')[0]
        if str(name) in files_val:
            index_val.append(num)
        else:
            index_train.append(num)
        num += 1
        val_image.append(name + '.png')
        val_label.append(name + '.txt')
    left = [filepath + left_fold + img for img in val_image]
    right = [filepath + right_fold + img for img in val_image]
    box_val_R = [path2 + label for label in val_label]

    # disp_train_R = [filepath+disp_R+img for img in train]

    left_train = [left[i] for i in index_train]
    left_val = [left[i] for i in index_val]
    right_train = [right[i] for i in index_train]
    right_val = [right[i] for i in index_val]
    box_r_val = [box_val_R[i] for i in index_val]
    box_r_train = [box_val_R[i] for i in index_train]
    box_l_val = [box_val_L[i] for i in index_val]
    box_l_train = [box_val_L[i] for i in index_train]

    return left_train, left_val, right_train, right_val, box_l_train, box_l_val, box_r_train, box_r_val

def default_output_file(args):
    out = 'outputs/{}-{}'.format(args.basenet, '-'.join(args.headnets))
    if args.square_edge != 321:
        out += '-edge{}'.format(args.square_edge)
    if args.regression_loss != 'laplace':
        out += '-{}'.format(args.regression_loss)
    if args.r_smooth != 0.0:
        out += '-rsmooth{}'.format(args.r_smooth)
    if args.dilation:
        out += '-dilation{}'.format(args.dilation)
    if args.dilation_end:
        out += '-dilationend{}'.format(args.dilation_end)

    now = datetime.datetime.now().strftime('%y%m%d-%H%M%S')
    out += '-{}.pkl'.format(now)

    return out


def cli():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    logs.cli(parser)
    nets.cli(parser)
    losses.cli(parser)
    encoder.cli(parser)
    optimize.cli(parser)
    parser.add_argument('-o', '--output', default=None,
                        help='output file')
    parser.add_argument('--batch-size', default=8, type=int,
                        help='batch size')
    parser.add_argument('--stride-apply', default=1, type=int,
                        help='apply and reset gradients every n batches')
    parser.add_argument('--epochs', default=75, type=int,
                        help='number of epochs to train')
    parser.add_argument('--loader-workers', default=1, type=int,
                        help='number of workers for data loading')
    parser.add_argument('--pre-n-images', default=10000, type=int,
                        help='number of images to sampe for pretraining')
    parser.add_argument('--n-images', default=None, type=int,
                        help='number of images to sampe')
    parser.add_argument('--freeze-base', default=0, type=int,
                        help='number of epochs to train with frozen base')
    parser.add_argument('--pre-lr', type=float, default=1e-5,
                        help='pre learning rate')
    parser.add_argument('--update-batchnorm-runningstatistics',
                        default=False, action='store_true',
                        help='update batch norm running statistics')
    parser.add_argument('--square-edge', default=401, type=int,
                        help='square edge of input images')
    parser.add_argument('--crop-fraction', default=1.0, type=float,
                        help='crop fraction versus rescale')
    parser.add_argument('--lambdas', default=[10.0, 3.0, 1.0, 10.0, 3.0, 3.0, 10.0, 3.0, 3.0],
                        type=float, nargs='+',
                        help='prefactor for head losses')
    parser.add_argument('--ema', default=1e-3, type=float,
                        help='ema decay constant')
    parser.add_argument('--debug-without-plots', default=False, action='store_true',
                        help='enable debug but dont plot')
    parser.add_argument('--profile', default=None,
                        help='enables profiling. specify path for chrome tracing file')
    parser.add_argument('--disable-cuda', action='store_true',
                        help='disable CUDA')
    parser.add_argument('--pretrained', default=None,
                        help='load a model from a checkpoint')
    args = parser.parse_args()

    if args.output is None:
        args.output = default_output_file(args)

    if args.debug and 'skeleton' not in args.headnets:
        raise Exception('add "skeleton" as last headnet to see debug output')

    if args.debug_without_plots:
        args.debug = True

    # add args.device
    args.device = torch.device('cpu')
    pin_memory = False
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda:1')
        pin_memory = True

    return args, pin_memory


def main():
    args, pin_memory = cli()
    logs.configure(args)
    net_cpu, start_epoch = nets.factory(args)

    for head in net_cpu.head_nets:
        head.apply_class_sigmoid = False

    pretrained = torch.load(args.pretrained)['model']
    net_cpu.base_net = pretrained.base_net
    net_cpu.head_nets[0] = pretrained.head_nets[0]
    net_cpu.head_nets[1] = pretrained.head_nets[1]
    net_cpu.head_nets[2] = pretrained.head_nets[2]
    net = net_cpu.to(device=args.device)

    # if not args.disable_cuda and torch.cuda.device_count() > 1:
    #     print('Using multiple GPUs: {}'.format(torch.cuda.device_count()))
    #     net = torch.nn.DataParallel(net)
    optimizer, lr_scheduler = optimize.factory(args, net.base_net.parameters(), net.head_nets[0].parameters(),
                                               net.head_nets[1].parameters(),net.head_nets[2].parameters())
    loss_list = losses.factory(args)
    print(net_cpu.io_scales())
    target_transforms = encoder.factory(args, net_cpu.io_scales())
    #
    preprocess = transforms.Compose([
        # transforms.HFlip(0.5),
        #     transforms.RescaleRelative(),
        transforms.Crop(401),
        transforms.CenterPad(401),
    ])
    left_train, left_val, right_train, right_val, box_l_train, box_l_val, box_r_train, box_r_val = dataloader(
                                                                                            '/data/wenlong-data/kitti_object/training/')
    train_data = data_kitti.myImageFloder(
        left_train,right_train, box_l_train, box_r_train,
        preprocess=preprocess,
        image_transform=transforms.image_transform_train,
        target_transforms=target_transforms,
        n_images=args.n_images,
    )
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=not args.debug,
         num_workers=args.loader_workers, drop_last=True,
        collate_fn=data_kitti.collate_images_targets_meta)
    preprocess = transforms.Compose([
        # transforms.HFlip(0.5),
        #     transforms.RescaleRelative(),
        transforms.Crop(401),
        transforms.CenterPad(401),
    ])
    val_data = data_kitti.myImageFloder(
        left_val, right_val, box_l_val, box_r_val,
        preprocess=preprocess,
        image_transform=transforms.image_transform_train,
        target_transforms=target_transforms,
        n_images=args.n_images,
    )
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=1, shuffle=False,
        num_workers=args.loader_workers, drop_last=True,
        collate_fn=data_kitti.collate_images_targets_meta)

    encoder_visualizer = None
    if args.debug and not args.debug_without_plots:
        encoder_visualizer = encoder.Visualizer(args.headnets, net_cpu.io_scales())

    if args.freeze_base:
        pre_train_data = data_kitti.myImageFloder(
            preprocess=preprocess,
            image_transform=transforms.image_transform_train,
            target_transforms=target_transforms,
            n_images=args.pre_n_images,
        )
        pre_train_loader = torch.utils.data.DataLoader(
            pre_train_data, batch_size=args.batch_size, shuffle=True,
             num_workers=args.loader_workers, drop_last=True,
            collate_fn=data_kitti.collate_images_targets_meta)

        # freeze base net parameters
        frozen_params = set()
        for n, p in net.named_parameters():
            if not n.startswith('base_net.'):
                continue
            print(n)
            if p.requires_grad is False:
                continue
            p.requires_grad = False
            frozen_params.add(p)
        for n, p in net.named_parameters():
            if not n.startswith('heads_nets.0.'):
                continue
            print(n)
            if p.requires_grad is False:
                continue
            p.requires_grad = False
            frozen_params.add(p)
        for n, p in net.named_parameters():
            if not n.startswith('heads_nets.1.'):
                continue
            print(n)
            if p.requires_grad is False:
                continue
            p.requires_grad = False
        print('froze {} parameters'.format(len(frozen_params)))

        # training
        foptimizer = torch.optim.SGD(
            (p for p in net.parameters() if p.requires_grad),
            lr=args.pre_lr, momentum=0.9, weight_decay=0.0, nesterov=True)
        ftrainer = Trainer(net, loss_list, foptimizer, args.output, args.lambdas,
                           device=args.device, fix_batch_norm=True,
                           encoder_visualizer=encoder_visualizer)
        for i in range(-args.freeze_base, 0):
            ftrainer.train(pre_train_loader, i)
            ftrainer.write_model(i + 1, epoch == i - 1)
        # unfreeze
        for p in frozen_params:
            p.requires_grad = True

    trainer = Trainer(
        net, loss_list, optimizer, args.output,
        lr_scheduler=lr_scheduler,
        device=args.device,
        fix_batch_norm=not args.update_batchnorm_runningstatistics,
        lambdas=args.lambdas,
        stride_apply=args.stride_apply,
        ema_decay=args.ema,
        encoder_visualizer=encoder_visualizer,
        train_profile=args.profile,
    )
    trainer.loop(train_loader, val_loader, args.epochs, start_epoch=start_epoch)


if __name__ == '__main__':
    main()
