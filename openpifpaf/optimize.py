import torch


def cli(parser):
    group = parser.add_argument_group('optimizer')
    group.add_argument('--lr', type=float, default=1e-3,
                       help='learning rate')
    group.add_argument('--momentum', type=float, default=0.9,
                       help='SGD momentum')
    group.add_argument('--no-nesterov', default=True, dest='nesterov', action='store_false',
                       help='do not use Nesterov momentum for SGD update')
    group.add_argument('--weight-decay', type=float, default=0.0,
                       help='SGD weight decay')
    group.add_argument('--adam', action='store_true',
                       help='use Adam optimizer')
    group.add_argument('--amsgrad', action='store_true',
                       help='use Adam optimizer with amsgrad option')

    group_s = parser.add_argument_group('learning rate scheduler')
    group_s.add_argument('--lr-decay', default=[], nargs='+', type=int,
                         help='epochs at which to decay the learning rate')
    group_s.add_argument('--lr-gamma', default=0.1, type=float,
                         help='lr decay factor')

def factory(args, parameters1,parameters2,parameters3,parameters4):
    if args.amsgrad:
        print('Adam optimizer with amsgrad')
        optimizer = torch.optim.Adam(
            [
                {'params': (p for p in parameters1 if p.requires_grad), 'lr': args.lr/20},
                {'params': (p for p in parameters2 if p.requires_grad)}
            ],
            lr=args.lr, weight_decay=args.weight_decay, amsgrad=True, eps=1e-4)
    elif args.adam:
        print('Adam optimizer')
        optimizer = torch.optim.Adam(
            [
                {'params': (p for p in parameters1 if p.requires_grad), 'lr': args.lr/20},
                {'params': (p for p in parameters2 if p.requires_grad)}
            ],
            lr=args.lr, weight_decay=args.weight_decay, eps=1e-4)
    else:
        print('SGD optimizer')
        optimizer = torch.optim.SGD(
            [
            {'params':  (p for p in parameters1 if p.requires_grad),'lr' : args.lr},
            ## balance the trainign data size
            {'params':  (p for p in parameters2 if p.requires_grad),'lr': args.lr/2},
            {'params': (p for p in parameters3 if p.requires_grad), 'lr': args.lr/2},
            {'params': (p for p in parameters4 if p.requires_grad),'lr': args.lr}
            ],lr =args.lr, momentum=args.momentum, weight_decay=args.weight_decay,
            nesterov=args.nesterov)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, args.lr_decay, gamma=args.lr_gamma)

    return optimizer, scheduler
