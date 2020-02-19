"""Head networks."""

import torch
import torch.nn as nn
from ..correlation_package.correlation import Correlation
def dequad(t):
    n_out_features = t.shape[1] // 4
    o = torch.zeros(
        (t.shape[0], n_out_features, t.shape[2] * 2 - 1, t.shape[3] * 2 - 1),
        device=t.device,
    )
    o[:, :, 0::2, 0::2] = t[:, 0:n_out_features]
    o[:, :, 1::2, 0::2] = t[:, n_out_features:n_out_features * 2, :-1, :]
    o[:, :, 0::2, 1::2] = t[:, n_out_features * 2:n_out_features * 3, :, :-1]
    o[:, :, 1::2, 1::2] = t[:, n_out_features * 3:n_out_features * 4, :-1, :-1]
    return o

class CompositeField(torch.nn.Module):
    def __init__(self, n, in_features, shortname=None,
                 dropout_p=0.0, quad=0,
                 n_confidences=1, n_vectors=0, n_scales=0,more = False):
        super(CompositeField, self).__init__()
        self.shortname = shortname or 'cf{}c{}v{}s{}'.format(
            n,
            n_confidences,
            n_vectors,
            n_scales,
        )
        self.dilation = 1.0
        self.apply_class_sigmoid = True
        self.more = more
        self.dropout = torch.nn.Dropout2d(p=dropout_p)
        self._quad = quad
        if self.more:
            self.downs = nn.Sequential(nn.Conv2d(in_features, in_features // 4, kernel_size=1, stride=1)
                                       , nn.BatchNorm2d(in_features // 4))
            self.corr = Correlation(pad_size=15, kernel_size=1, max_displacement=15, stride1=1, stride2=2,
                                    corr_multiply=1)
            self.corr2 = Correlation(pad_size=15, kernel_size=1, max_displacement=15, stride1=1, stride2=2,
                                    corr_multiply=1)
            self.corr_activation = nn.LeakyReLU(0.1, inplace=True)
            self.normalize = nn.BatchNorm2d(225)
            # self.conv_redir = nn.Sequential(nn.Conv2d(in_features // 4, 256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(68))
            # self.downs2 = nn.Sequential(nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(in_features))
            in_features = 1474
            in_features_c = 1249
        out_features = n * (4 ** self._quad)
        self.class_convs = torch.nn.ModuleList([
            torch.nn.Conv2d(in_features, out_features, 1)
            for _ in range(n_confidences)
        ])
        #
        if self.more:
            # regression
            self.reg_convs = torch.nn.ModuleList([
                torch.nn.Conv2d(in_features_c, 2*out_features, 1)
                for _ in range(n_vectors)
            ])
            self.reg_spreads = torch.nn.ModuleList([
                torch.nn.Conv2d(in_features_c, out_features, 1)
                for _ in self.reg_convs
            ])

        else:

            # regression
            self.reg_convs = torch.nn.ModuleList([
                torch.nn.Conv2d(in_features, 2*out_features, 1)
                for _ in range(n_vectors)
            ])
            self.reg_spreads = torch.nn.ModuleList([
                torch.nn.Conv2d(in_features, out_features, 1)
                for _ in self.reg_convs
            ])

        # scale
        self.scale_convs = torch.nn.ModuleList([
            torch.nn.Conv2d(in_features, out_features, 1)
            for _ in range(n_scales)
        ])

        # dequad
        self.dequad_op = torch.nn.PixelShuffle(2)

    def forward(self, x):  # pylint: disable=arguments-differ
        x = self.dropout(x)
        if self.more:
            x_int = self.downs(x)
            batchsize = x.shape[0]
            x1 = x_int[:int(batchsize // 2)]
            x2 = x_int[int(batchsize // 2):]
            out_corr = self.corr(x1, x2)  # False
            out_corr2 = self.corr2(x2, x1)  # False
            # print(x1.shape)
            # print(out_corr.shape)
            out_corr = self.normalize(self.corr_activation(out_corr))
            out_corr2 = self.normalize(self.corr_activation(out_corr2))

            x_int = torch.cat((x1, x2, out_corr,out_corr2), 1)
            x_sp_l = torch.cat((x1, x2, out_corr), 1)
            x_sp_r = torch.cat((x1, x2, out_corr2), 1)

            classes_x = [class_conv(x_int) for class_conv in self.class_convs]
            if self.apply_class_sigmoid:
                classes_x = [torch.sigmoid(class_x) for class_x in classes_x]

            # regressions
            regs_x = [reg_conv(x_sp) * self.dilation for reg_conv,x_sp in zip(self.reg_convs,[x_sp_l,x_sp_r])]
            regs_x_spread = [torch.clamp(reg_spread(x_sp), -3, 5)
                             for reg_spread,x_sp in zip(self.reg_spreads,[x_sp_l,x_sp_r])]

            # scale
            scales_x = [scale_conv(x) for scale_conv in self.scale_convs]
            scales_x = [torch.nn.functional.relu(scale_x) for scale_x in scales_x]

            for _ in range(self._quad):
                classes_x = [self.dequad_op(class_x)[:, :, :-1, :-1]
                             for class_x in classes_x]
                regs_x = [self.dequad_op(reg_x)[:, :, :-1, :-1]
                          for reg_x in regs_x]
                regs_x_spread = [self.dequad_op(reg_x_spread)[:, :, :-1, :-1]
                                 for reg_x_spread in regs_x_spread]
                scales_x = [self.dequad_op(scale_x)[:, :, :-1, :-1]
                            for scale_x in scales_x]
        else:

            # classification
            classes_x = [class_conv(x) for class_conv in self.class_convs]
            if self.apply_class_sigmoid:
                classes_x = [torch.sigmoid(class_x) for class_x in classes_x]

            # regressions
            regs_x = [reg_conv(x) * self.dilation for reg_conv in self.reg_convs]
            regs_x_spread = [torch.clamp(reg_spread(x), -3, 5)
                             for reg_spread in self.reg_spreads]

            # scale
            scales_x = [scale_conv(x) for scale_conv in self.scale_convs]
            scales_x = [torch.nn.functional.relu(scale_x) for scale_x in scales_x]

            for _ in range(self._quad):
                classes_x = [self.dequad_op(class_x)[:, :, :-1, :-1]
                             for class_x in classes_x]
                regs_x = [self.dequad_op(reg_x)[:, :, :-1, :-1]
                          for reg_x in regs_x]
                regs_x_spread = [self.dequad_op(reg_x_spread)[:, :, :-1, :-1]
                                 for reg_x_spread in regs_x_spread]
                scales_x = [self.dequad_op(scale_x)[:, :, :-1, :-1]
                            for scale_x in scales_x]
        regs_x = [
            reg_x.reshape(reg_x.shape[0],
                          reg_x.shape[1]//2,
                          2,
                          reg_x.shape[2],
                          reg_x.shape[3])
            for reg_x in regs_x
        ]

        return classes_x + regs_x + regs_x_spread + scales_x
