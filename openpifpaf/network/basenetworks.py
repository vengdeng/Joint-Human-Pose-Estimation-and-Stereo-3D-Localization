import torch


class BaseNetwork(torch.nn.Module):
    """Common base network."""

    def __init__(self, net, shortname, input_output_scale, out_features):
        super(BaseNetwork, self).__init__()

        self.net = net
        self.shortname = shortname
        self.input_output_scale = input_output_scale
        self.out_features = out_features
        self.topology = 'linear'

        # print(list(net.children()))
        print('input output scale', self.input_output_scale)
        print('output features', self.out_features)

    def forward(self, image):  # pylint: disable=arguments-differ
        if isinstance(self.net, torch.nn.ModuleList):
            if self.topology == 'linear':
                intermediate = image
                outputs = []
                for n in self.net:
                    intermediate = n(intermediate)
                    outputs.append(intermediate)

                return outputs

            if self.topology == 'fork':
                intermediate = self.net[0](image)
                return intermediate, self.net[1](intermediate), self.net[2](intermediate)
        return self.net(image)
        # return intermediate, outputs

class DownsampleCat(torch.nn.Module):
    def __init__(self):
        super(DownsampleCat, self).__init__()
        self.pad = torch.nn.ConstantPad2d((0, 1, 0, 1), 0.0)

    def forward(self, x):  # pylint: disable=arguments-differ
        p = self.pad(x)
        o = torch.cat((p[:, :, :-1:2, :-1:2], p[:, :, 1::2, 1::2]), dim=1)
        return o


class ResnetBlocks(object):
    def __init__(self, resnet):
        self.modules = list(resnet.children())
        # print('===============')
        # print(self.modules)

    def input_block(self, use_pool=False, conv_stride=2, pool_stride=2):
        modules = self.modules[:4]

        if not use_pool:
            modules.pop(3)
        else:
            if pool_stride != 2:
                modules[3].stride = torch.nn.modules.utils._pair(pool_stride)  # pylint: disable=protected-access

        if conv_stride != 2:
            modules[0].stride = torch.nn.modules.utils._pair(conv_stride)  # pylint: disable=protected-access

        return torch.nn.Sequential(*modules)

    @staticmethod
    def dilation(block, dilation, stride=1):
        convs = [m for m in block.modules() if isinstance(m, torch.nn.Conv2d)]

        for conv in convs:
            if conv.kernel_size[0] == 1:
                continue

            conv.dilation = torch.nn.modules.utils._pair(dilation)  # pylint: disable=protected-access

            padding = (conv.kernel_size[0] - 1) // 2 * dilation
            conv.padding = torch.nn.modules.utils._pair(padding)  # pylint: disable=protected-access

        # TODO: check these are the right convolutions to adjust
        for conv in convs[:2]:
            conv.stride = torch.nn.modules.utils._pair(stride)  # pylint: disable=protected-access

        return block

    @staticmethod
    def stride(block):
        """Compute the output stride of a block.

        Assume that convolutions are in serious with pools; only one
        convolutions with non-unit stride.
        """
        if isinstance(block, list):
            stride = 1
            for b in block:
                stride *= ResnetBlocks.stride(b)
            return stride

        conv_stride = max(m.stride[0]
                          for m in block.modules()
                          if isinstance(m, torch.nn.Conv2d))

        pool_stride = 1
        pools = [m for m in block.modules() if isinstance(m, torch.nn.MaxPool2d)]
        if pools:
            for p in pools:
                pool_stride *= p.stride

        return conv_stride * pool_stride

    @staticmethod
    def replace_downsample(block):
        print('!!!!!!!!!!')
        first_bottleneck = block[0]
        print(first_bottleneck.downsample)
        first_bottleneck.downsample = DownsampleCat()
        print(first_bottleneck)

    @staticmethod
    def out_channels(block):
        """For blocks 2-5."""
        last_conv = list(block.modules())[-3]
        return last_conv.out_channels

    def block2(self):
        return self.modules[4]

    def block3(self):
        return self.modules[5]

    def block4(self):
        return self.modules[6]

    def block5(self):
        return self.modules[7]

