#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""U-Net model."""

# -- File info -- #
from torchvision.models.resnet import BasicBlock
from torch import nn
from torchvision import models
__author__ = 'Andreas R. Stokholm'
__contributor__ = 'Andrzej S. Kucik'
__copyright__ = ['Technical University of Denmark', 'European Space Agency']
__contact__ = ['stokholm@space.dtu.dk', 'andrzej.kucik@esa.int']
__version__ = '0.3.0'
__date__ = '2022-09-20'


# -- Third-party modules -- #
import torch


class UNet(torch.nn.Module):
    """PyTorch U-Net Class. Uses unet_parts."""

    def __init__(self, options):
        super().__init__()

        self.input_block = DoubleConv(options, input_n=len(options['train_variables']),
                                      output_n=options['unet_conv_filters'][0])

        self.contract_blocks = torch.nn.ModuleList()
        for contract_n in range(1, len(options['unet_conv_filters'])):
            self.contract_blocks.append(
                ContractingBlock(options=options,
                                 input_n=options['unet_conv_filters'][contract_n - 1],
                                 output_n=options['unet_conv_filters'][contract_n]))
            # only used to contract input patch.

        self.bridge = ContractingBlock(
            options, input_n=options['unet_conv_filters'][-1], output_n=options['unet_conv_filters'][-1])

        self.expand_blocks = torch.nn.ModuleList()
        self.expand_blocks.append(
            ExpandingBlock(options=options, input_n=options['unet_conv_filters'][-1],
                           output_n=options['unet_conv_filters'][-1]))

        for expand_n in range(len(options['unet_conv_filters']), 1, -1):
            self.expand_blocks.append(ExpandingBlock(options=options,
                                                     input_n=options['unet_conv_filters'][expand_n - 1],
                                                     output_n=options['unet_conv_filters'][expand_n - 2]))

        self.sic_feature_map = FeatureMap(input_n=options['unet_conv_filters'][0], output_n=options['n_classes']['SIC'])
        self.sod_feature_map = FeatureMap(input_n=options['unet_conv_filters'][0], output_n=options['n_classes']['SOD'])
        self.floe_feature_map = FeatureMap(
            input_n=options['unet_conv_filters'][0], output_n=options['n_classes']['FLOE'])

    def forward(self, x):
        """Forward model pass."""
        x_contract = [self.input_block(x)]
        for contract_block in self.contract_blocks:
            x_contract.append(contract_block(x_contract[-1]))
        x_expand = self.bridge(x_contract[-1])
        up_idx = len(x_contract)
        for expand_block in self.expand_blocks:
            x_expand = expand_block(x_expand, x_contract[up_idx - 1])
            up_idx -= 1

        return {'SIC': self.sic_feature_map(x_expand),
                'SOD': self.sod_feature_map(x_expand),
                'FLOE': self.floe_feature_map(x_expand)}


class FeatureMap(torch.nn.Module):
    """Class to perform final 1D convolution before calculating cross entropy or using softmax."""

    def __init__(self, input_n, output_n):
        super(FeatureMap, self).__init__()

        self.feature_out = torch.nn.Conv2d(input_n, output_n, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        """Pass x through final layer."""
        return self.feature_out(x)


class DoubleConv(torch.nn.Module):
    """Class to perform a double conv layer in the U-NET architecture. Used in unet_model.py."""

    def __init__(self, options, input_n, output_n):
        super(DoubleConv, self).__init__()

        self.double_conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=input_n,
                            out_channels=output_n,
                            kernel_size=options['conv_kernel_size'],
                            stride=options['conv_stride_rate'],
                            padding=options['conv_padding'],
                            padding_mode=options['conv_padding_style'],
                            bias=False),
            torch.nn.BatchNorm2d(output_n),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=output_n,
                            out_channels=output_n,
                            kernel_size=options['conv_kernel_size'],
                            stride=options['conv_stride_rate'],
                            padding=options['conv_padding'],
                            padding_mode=options['conv_padding_style'],
                            bias=False),
            torch.nn.BatchNorm2d(output_n),
            torch.nn.ReLU()
        )

    def forward(self, x):
        """Pass x through the double conv layer."""
        x = self.double_conv(x)

        return x


class ContractingBlock(torch.nn.Module):
    """Class to perform downward pass in the U-Net."""

    def __init__(self, options, input_n, output_n):
        super(ContractingBlock, self).__init__()

        self.contract_block = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.double_conv = DoubleConv(options, input_n, output_n)

    def forward(self, x):
        """Pass x through the downward layer."""
        x = self.contract_block(x)
        x = self.double_conv(x)
        return x


class ExpandingBlock(torch.nn.Module):
    """Class to perform upward layer in the U-Net."""

    def __init__(self, options, input_n, output_n):
        super(ExpandingBlock, self).__init__()

        self.padding_style = options['conv_padding_style']
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.double_conv = DoubleConv(options, input_n=input_n + output_n, output_n=output_n)

    def forward(self, x, x_skip):
        """Pass x through the upward layer and concatenate with opposite layer."""
        x = self.upsample(x)

        # Insure that x and skip H and W dimensions match.
        x = expand_padding(x, x_skip, padding_style=self.padding_style)
        x = torch.cat([x, x_skip], dim=1)

        return self.double_conv(x)


def expand_padding(x, x_contract, padding_style: str = 'constant'):
    """
    Insure that x and x_skip H and W dimensions match.
    Parameters
    ----------
    x :
        Image tensor of shape (batch size, channels, height, width). Expanding path.
    x_contract :
        Image tensor of shape (batch size, channels, height, width) Contracting path.
        or torch.Size. Contracting path.
    padding_style : str
        Type of padding.

    Returns
    -------
    x : ndtensor
        Padded expanding path.
    """
    # Check whether x_contract is tensor or shape.
    if type(x_contract) == type(x):
        x_contract = x_contract.size()

    # Calculate necessary padding to retain patch size.
    pad_y = x_contract[2] - x.size()[2]
    pad_x = x_contract[3] - x.size()[3]

    if padding_style == 'zeros':
        padding_style = 'constant'

    x = torch.nn.functional.pad(x, [pad_x // 2, pad_x - pad_x // 2, pad_y // 2, pad_y - pad_y // 2], mode=padding_style)

    return x


class UNet_sep_dec(torch.nn.Module):
    """PyTorch U-Net Class. Uses unet_parts."""

    def __init__(self, options):
        super().__init__()

        self.input_block = DoubleConv(options, input_n=len(
            options['train_variables']), output_n=options['unet_conv_filters'][0])

        self.contract_blocks = torch.nn.ModuleList()
        for contract_n in range(1, len(options['unet_conv_filters'])):
            self.contract_blocks.append(
                ContractingBlock(options=options,
                                 input_n=options['unet_conv_filters'][contract_n - 1],
                                 output_n=options['unet_conv_filters'][contract_n]))  # only used to contract input patch.

        self.bridge = ContractingBlock(
            options, input_n=options['unet_conv_filters'][-1], output_n=options['unet_conv_filters'][-1])

        self.expand_blocks = torch.nn.ModuleList()
        self.expand_blocks.append(
            ExpandingBlock(options=options, input_n=options['unet_conv_filters'][-1], output_n=options['unet_conv_filters'][-1]))

        for expand_n in range(len(options['unet_conv_filters']), 1, -1):
            self.expand_blocks.append(ExpandingBlock(options=options,
                                                     input_n=options['unet_conv_filters'][expand_n - 1],
                                                     output_n=options['unet_conv_filters'][expand_n - 2]))

        self.sic_feature_map = FeatureMap(input_n=options['unet_conv_filters'][0], output_n=options['n_classes']['SIC'])
        self.sod_feature_map = FeatureMap(input_n=options['unet_conv_filters'][0], output_n=options['n_classes']['SOD'])
        self.floe_feature_map = FeatureMap(
            input_n=options['unet_conv_filters'][0], output_n=options['n_classes']['FLOE'])

    def Decoder(self, x_contract):

        x_expand = self.bridge(x_contract[-1])
        up_idx = len(x_contract)
        for expand_block in self.expand_blocks:
            x_expand = expand_block(x_expand, x_contract[up_idx - 1])
            up_idx -= 1

        return x_expand

    def forward(self, x):
        """Forward model pass."""
        x_contract = [self.input_block(x)]
        for contract_block in self.contract_blocks:
            x_contract.append(contract_block(x_contract[-1]))

        return {'SIC': self.sic_feature_map(self.Decoder(x_contract)),
                'SOD': self.sod_feature_map(self.Decoder(x_contract)),
                'FLOE': self.floe_feature_map(self.Decoder(x_contract))}


class Sep_feat_dif_stages(torch.nn.Module):
    """PyTorch U-Net Class. Uses unet_parts."""

    def __init__(self, options):
        super().__init__()

        self.stage = options['common_features_last_layer']
        self.drop = nn.Identity()

        if 'resnet' in options['backbone']:
            a = Resnet_backbone(options, len(options['train_variables']), drop_rate=0.1, output_stride=16)
            self.comm_feat = a.comm_feat
            self.ind_feat1 = a.ind_feat1
            self.ind_feat2 = a.ind_feat2
            self.ind_feat3 = a.ind_feat3
            self.drop = a.drop
            count_stage = a.count_stage

        else:   # unet
            self.comm_feat, self.ind_feat1, self.ind_feat2, self.ind_feat3 = [torch.nn.ModuleList() for i in range(4)]

            # input_block
            count_stage = 1
            if count_stage <= self.stage:
                self.comm_feat.append(DoubleConv(options, input_n=len(
                    options['train_variables']), output_n=options['unet_conv_filters'][0]))
            else:
                a, b, c = [DoubleConv(options, input_n=len(options['train_variables']),
                                      output_n=options['unet_conv_filters'][0]) for i in range(3)]
                self.ind_feat1.append(a)
                self.ind_feat2.append(b)
                self.ind_feat3.append(c)

            # contract_blocks, only used to contract input patch
            for contract_n in range(1, len(options['unet_conv_filters'])):
                count_stage += 1
                if count_stage <= self.stage:
                    self.comm_feat.append(ContractingBlock(
                        options=options, input_n=options['unet_conv_filters'][contract_n - 1], output_n=options['unet_conv_filters'][contract_n]))
                else:
                    a, b, c = [ContractingBlock(options=options, input_n=options['unet_conv_filters']
                                                [contract_n - 1], output_n=options['unet_conv_filters'][contract_n]) for i in range(3)]
                    self.ind_feat1.append(a)
                    self.ind_feat2.append(b)
                    self.ind_feat3.append(c)

            # bridge
            count_stage += 1
            if count_stage <= self.stage:
                self.comm_feat.append(ContractingBlock(
                    options, input_n=options['unet_conv_filters'][-1], output_n=options['unet_conv_filters'][-1]))
            else:
                a, b, c = [ContractingBlock(options, input_n=options['unet_conv_filters'][-1],
                                            output_n=options['unet_conv_filters'][-1]) for i in range(3)]
                self.ind_feat1.append(a)
                self.ind_feat2.append(b)
                self.ind_feat3.append(c)

        # expand_blocks
        count_stage += 1
        if count_stage <= self.stage:
            self.comm_feat.append(ExpandingBlock(
                options=options, input_n=options['unet_conv_filters'][-1], output_n=options['unet_conv_filters'][-1]))
        else:
            a, b, c = [ExpandingBlock(options=options, input_n=options['unet_conv_filters'][-1],
                                      output_n=options['unet_conv_filters'][-1]) for i in range(3)]
            self.ind_feat1.append(a)
            self.ind_feat2.append(b)
            self.ind_feat3.append(c)

        for expand_n in range(len(options['unet_conv_filters']), 1, -1):
            count_stage += 1
            if count_stage <= self.stage:
                self.comm_feat.append(ExpandingBlock(
                    options=options, input_n=options['unet_conv_filters'][expand_n - 1], output_n=options['unet_conv_filters'][expand_n - 2]))
            else:
                a, b, c = [ExpandingBlock(options=options, input_n=options['unet_conv_filters'][expand_n - 1],
                                          output_n=options['unet_conv_filters'][expand_n - 2]) for i in range(3)]
                self.ind_feat1.append(a)
                self.ind_feat2.append(b)
                self.ind_feat3.append(c)

        self.sic_feature_map = FeatureMap(input_n=options['unet_conv_filters'][0], output_n=options['n_classes']['SIC'])
        self.sod_feature_map = FeatureMap(input_n=options['unet_conv_filters'][0], output_n=options['n_classes']['SOD'])
        self.floe_feature_map = FeatureMap(
            input_n=options['unet_conv_filters'][0], output_n=options['n_classes']['FLOE'])

    def Independent(self, ind_feat, features, up_idx):

        for block in ind_feat:
            if not isinstance(block, ExpandingBlock):
                features.append(self.drop(block(features[-1])))
                up_idx += 1
            else:
                features.append(block(features[-1], features[up_idx - 1]))
                up_idx -= 1

        return features[-1]

    def forward(self, x):
        """Forward model pass."""

        features = [x]
        up_idx = 0

        for block in self.comm_feat:
            if not isinstance(block, ExpandingBlock):
                features.append(self.drop(block(features[-1])))
                up_idx += 1
            else:
                features.append(block(features[-1], features[up_idx - 1]))
                up_idx -= 1

        return {'SIC': self.sic_feature_map(self.Independent(self.ind_feat1, features[:], up_idx)),
                'SOD': self.sod_feature_map(self.Independent(self.ind_feat2, features[:], up_idx)),
                'FLOE': self.floe_feature_map(self.Independent(self.ind_feat3, features[:], up_idx))}


class Resnet_backbone(torch.nn.Module):
    def __init__(self, options, in_chans, drop_rate, output_stride):
        super(Resnet_backbone, self).__init__()

        cnn_func = getattr(models, options['backbone'])
        self.enc = cnn_func()

        if options['backbone'] == 'resnet18':
            layers = [2, 2, 2, 2]           # number of layers in residual blocks
        elif options['backbone'] == 'resnet34':
            layers = [3, 4, 6, 3]
        elif options['backbone'] == 'resnet50':
            layers = [3, 4, 6, 3]
        elif options['backbone'] == 'resnet101':
            layers = [3, 4, 23, 3]

        # Custom strides
        self.strides = [1]
        for i in range(4):
            if output_stride > 1:
                self.strides.append(2)
                output_stride //= 2
            else:
                self.strides.append(1)

        self.stage = options['common_features_last_layer']
        self.comm_feat, self.ind_feat1, self.ind_feat2, self.ind_feat3 = [torch.nn.ModuleList() for i in range(4)]

        self.count_stage = 1
        self.enc.inplanes = options['unet_conv_filters'][0]
        # self.enc.conv1 = nn.Conv2d(in_chans, self.enc.inplanes, kernel_size=7, stride=self.strides[0], padding=3, bias=False)
        # self.enc.bn1 = nn.BatchNorm2d(options['unet_conv_filters'][0])
        # if self.strides[1] > 1:
        #     self.enc.maxpool = nn.MaxPool2d(kernel_size=3, stride=self.strides[1], padding=1)
        # else: self.enc.maxpool = nn.Identity()

        if self.count_stage <= self.stage:
            self.comm_feat.append(nn.Sequential(nn.Conv2d(in_chans, self.enc.inplanes, kernel_size=7, stride=self.strides[0], padding=3, bias=False),
                                  nn.BatchNorm2d(options['unet_conv_filters'][0]),
                                  nn.ReLU(inplace=True)))
        else:
            a, b, c = [nn.Sequential(nn.Conv2d(in_chans, self.enc.inplanes, kernel_size=7, stride=self.strides[0], padding=3, bias=False),
                                     nn.BatchNorm2d(options['unet_conv_filters'][0]),
                                     nn.ReLU(inplace=True)) for i in range(3)]
            self.ind_feat1.append(a)
            self.ind_feat2.append(b)
            self.ind_feat3.append(c)

        # self.enc.layer1 = self.enc._make_layer(BasicBlock, options['unet_conv_filters'][ 1], layers[0])
        self.count_stage += 1
        if self.count_stage <= self.stage:
            self.comm_feat.append(nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=self.strides[1], padding=1) if self.strides[1] > 1 else nn.Identity(),
                                                self.enc._make_layer(BasicBlock, options['unet_conv_filters'][1], layers[0])))

        else:
            aux = self.enc.inplanes
            a = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=self.strides[1], padding=1) if self.strides[1] > 1 else nn.Identity(),
                              self.enc._make_layer(BasicBlock, options['unet_conv_filters'][1], layers[0]))
            self.enc.inplanes = aux
            b = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=self.strides[1], padding=1) if self.strides[1] > 1 else nn.Identity(),
                              self.enc._make_layer(BasicBlock, options['unet_conv_filters'][1], layers[0]))
            self.enc.inplanes = aux
            c = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=self.strides[1], padding=1) if self.strides[1] > 1 else nn.Identity(),
                              self.enc._make_layer(BasicBlock, options['unet_conv_filters'][1], layers[0]))
            self.ind_feat1.append(a)
            self.ind_feat2.append(b)
            self.ind_feat3.append(c)

        # self.enc.layer2 = self.enc._make_layer(BasicBlock, options['unet_conv_filters'][ 2], layers[1], stride=self.strides[2])
        self.count_stage += 1
        if self.count_stage <= self.stage:
            self.comm_feat.append(self.enc._make_layer(
                BasicBlock, options['unet_conv_filters'][2], layers[1], stride=self.strides[2]))
        else:
            aux = self.enc.inplanes
            a = self.enc._make_layer(BasicBlock, options['unet_conv_filters'][2], layers[1], stride=self.strides[2])
            self.enc.inplanes = aux
            b = self.enc._make_layer(BasicBlock, options['unet_conv_filters'][2], layers[1], stride=self.strides[2])
            self.enc.inplanes = aux
            c = self.enc._make_layer(BasicBlock, options['unet_conv_filters'][2], layers[1], stride=self.strides[2])
            self.ind_feat1.append(a)
            self.ind_feat2.append(b)
            self.ind_feat3.append(c)

        # self.enc.layer3 = self.enc._make_layer(BasicBlock, options['unet_conv_filters'][ 3] , layers[2], stride=self.strides[3])
        self.count_stage += 1
        if self.count_stage <= self.stage:
            self.comm_feat.append(self.enc._make_layer(
                BasicBlock, options['unet_conv_filters'][3], layers[2], stride=self.strides[3]))
        else:
            aux = self.enc.inplanes
            a = self.enc._make_layer(BasicBlock, options['unet_conv_filters'][3], layers[2], stride=self.strides[3])
            self.enc.inplanes = aux
            b = self.enc._make_layer(BasicBlock, options['unet_conv_filters'][3], layers[2], stride=self.strides[3])
            self.enc.inplanes = aux
            c = self.enc._make_layer(BasicBlock, options['unet_conv_filters'][3], layers[2], stride=self.strides[3])
            self.ind_feat1.append(a)
            self.ind_feat2.append(b)
            self.ind_feat3.append(c)

        # self.enc.layer4 = self.enc._make_layer(BasicBlock, options['unet_conv_filters'][-1], layers[3], stride=self.strides[4])
        self.count_stage += 1
        if self.count_stage <= self.stage:
            self.comm_feat.append(self.enc._make_layer(
                BasicBlock, options['unet_conv_filters'][-1], layers[3], stride=self.strides[4]))
        else:
            aux = self.enc.inplanes
            a = self.enc._make_layer(BasicBlock, options['unet_conv_filters'][-1], layers[3], stride=self.strides[4])
            self.enc.inplanes = aux
            b = self.enc._make_layer(BasicBlock, options['unet_conv_filters'][-1], layers[3], stride=self.strides[4])
            self.enc.inplanes = aux
            c = self.enc._make_layer(BasicBlock, options['unet_conv_filters'][-1], layers[3], stride=self.strides[4])
            self.ind_feat1.append(a)
            self.ind_feat2.append(b)
            self.ind_feat3.append(c)

        self.drop = nn.Dropout2d(drop_rate)


class UNet_regression(UNet):
    def __init__(self, options):
        super().__init__(options)
        self.regression_layer = torch.nn.Linear(options['unet_conv_filters'][0], 1)

    def forward(self, x):
        """Forward model pass."""
        x_contract = [self.input_block(x)]
        for contract_block in self.contract_blocks:
            x_contract.append(contract_block(x_contract[-1]))

        x_expand = self.bridge(x_contract[-1])
        up_idx = len(x_contract)
        for expand_block in self.expand_blocks:
            x_expand = expand_block(x_expand, x_contract[up_idx - 1])
            up_idx -= 1

        return {'SIC': self.regression_layer(x_expand.permute(0, 2, 3, 1)),
                'SOD': self.sod_feature_map(x_expand),
                'FLOE': self.floe_feature_map(x_expand)}

class UNet_regression_all(UNet):
    def __init__(self, options):
        super().__init__(options)
        self.regression_layer = torch.nn.Linear(options['unet_conv_filters'][0], 1)

    def forward(self, x):
        """Forward model pass."""
        x_contract = [self.input_block(x)]
        for contract_block in self.contract_blocks:
            x_contract.append(contract_block(x_contract[-1]))

        x_expand = self.bridge(x_contract[-1])
        up_idx = len(x_contract)
        for expand_block in self.expand_blocks:
            x_expand = expand_block(x_expand, x_contract[up_idx - 1])
            up_idx -= 1

        return {'SIC': self.regression_layer(x_expand.permute(0, 2, 3, 1)),
                'SOD': self.regression_layer(x_expand.permute(0, 2, 3, 1)),
                'FLOE': self.regression_layer(x_expand.permute(0, 2, 3, 1))}




class UNet_sep_dec_regression(UNet):
    def __init__(self, options):
        super().__init__(options)

        # SIC decoder
        self.expand_sic_blocks = torch.nn.ModuleList()
        self.expand_sic_blocks.append(
            ExpandingBlock(options=options, input_n=options['unet_conv_filters'][-1],
                           output_n=options['unet_conv_filters'][-1]))
        for expand_n in range(len(options['unet_conv_filters']), 1, -1):
            self.expand_sic_blocks.append(ExpandingBlock(options=options,
                                                         input_n=options['unet_conv_filters'][expand_n - 1],
                                                         output_n=options['unet_conv_filters'][expand_n - 2]))
        # SOD decoder
        self.expand_sod_blocks = torch.nn.ModuleList()
        self.expand_sod_blocks.append(
            ExpandingBlock(options=options, input_n=options['unet_conv_filters'][-1],
                           output_n=options['unet_conv_filters'][-1]))
        for expand_n in range(len(options['unet_conv_filters']), 1, -1):
            self.expand_sod_blocks.append(ExpandingBlock(options=options,
                                                         input_n=options['unet_conv_filters'][expand_n - 1],
                                                         output_n=options['unet_conv_filters'][expand_n - 2]))
        # FLOE decoder
        self.expand_floe_blocks = torch.nn.ModuleList()
        self.expand_floe_blocks.append(
            ExpandingBlock(options=options, input_n=options['unet_conv_filters'][-1],
                           output_n=options['unet_conv_filters'][-1]))
        for expand_n in range(len(options['unet_conv_filters']), 1, -1):
            self.expand_floe_blocks.append(ExpandingBlock(options=options,
                                                          input_n=options['unet_conv_filters'][expand_n - 1],
                                                          output_n=options['unet_conv_filters'][expand_n - 2]))

        # regression layer
        self.regression_layer = torch.nn.Linear(options['unet_conv_filters'][0], 1)

    def forward(self, x):
        """Forward model pass."""
        x_contract = [self.input_block(x)]
        for contract_block in self.contract_blocks:
            x_contract.append(contract_block(x_contract[-1]))
        x_expand = self.bridge(x_contract[-1])

        up_idx = len(x_contract)
        x_expand_sic = x_expand
        x_expand_sod = x_expand
        x_expand_floe = x_expand

        # decoder SIC
        for expand_block in self.expand_sic_blocks:
            x_expand_sic = expand_block(x_expand_sic, x_contract[up_idx - 1])
            up_idx -= 1

        up_idx = len(x_contract)
        # decoder SOD
        for expand_block in self.expand_sod_blocks:
            x_expand_sod = expand_block(x_expand_sod, x_contract[up_idx - 1])
            up_idx -= 1

        up_idx = len(x_contract)
        # decoder FLOE
        for expand_block in self.expand_floe_blocks:
            x_expand_floe = expand_block(x_expand_floe, x_contract[up_idx - 1])
            up_idx -= 1

        return {'SIC': self.regression_layer(x_expand_sic.permute(0, 2, 3, 1)),
                'SOD': self.sod_feature_map(x_expand_sod),
                'FLOE': self.floe_feature_map(x_expand_floe)}


class UNet_sep_dec_mse(torch.nn.Module):
	"""PyTorch U-Net Class. Uses unet_parts."""

	def __init__(self, options):
		super().__init__()

		self.input_block = DoubleConv(options, input_n=len(options['train_variables']),
			output_n=options['unet_conv_filters'][0])

		self.contract_blocks = torch.nn.ModuleList()
		for contract_n in range(1, len(options['unet_conv_filters'])):
			self.contract_blocks.append(
					ContractingBlock(options=options,
					input_n=options['unet_conv_filters'][contract_n - 1],
					output_n=options['unet_conv_filters'][contract_n]))
		# only used to contract input patch.

		self.bridge = ContractingBlock(
			options, input_n=options['unet_conv_filters'][-1], output_n=options['unet_conv_filters'][-1])

		self.expand_sic_blocks = torch.nn.ModuleList()
		self.expand_sic_blocks.append(
			ExpandingBlock(options=options, input_n=options['unet_conv_filters'][-1],
				output_n=options['unet_conv_filters'][-1]))
		for expand_n in range(len(options['unet_conv_filters']), 1, -1):
			self.expand_sic_blocks.append(ExpandingBlock(options=options, 
					input_n=options['unet_conv_filters'][expand_n - 1],
					output_n=options['unet_conv_filters'][expand_n - 2]))

		self.expand_sod_blocks = torch.nn.ModuleList()
		self.expand_sod_blocks.append(
			ExpandingBlock(options=options, input_n=options['unet_conv_filters'][-1],
				output_n=options['unet_conv_filters'][-1]))
		for expand_n in range(len(options['unet_conv_filters']), 1, -1):
			self.expand_sod_blocks.append(ExpandingBlock(options=options, 
					input_n=options['unet_conv_filters'][expand_n - 1],
					output_n=options['unet_conv_filters'][expand_n - 2]))

		self.expand_floe_blocks = torch.nn.ModuleList()
		self.expand_floe_blocks.append(
			ExpandingBlock(options=options, input_n=options['unet_conv_filters'][-1],
				output_n=options['unet_conv_filters'][-1]))
		for expand_n in range(len(options['unet_conv_filters']), 1, -1):
			self.expand_floe_blocks.append(ExpandingBlock(options=options, 
					input_n=options['unet_conv_filters'][expand_n - 1],
					output_n=options['unet_conv_filters'][expand_n - 2]))
		self.sod_feature_map = FeatureMap(
      input_n=options['unet_conv_filters'][0], output_n=options['n_classes']['SOD'])
		self.floe_feature_map = FeatureMap(
			input_n=options['unet_conv_filters'][0], output_n=options['n_classes']['FLOE'])
		self.regression_layer = torch.nn.Linear(options['unet_conv_filters'][0], 1)
   
	def forward(self, x):
		"""Forward model pass."""
		x_contract = [self.input_block(x)]
		for contract_block in self.contract_blocks:
			x_contract.append(contract_block(x_contract[-1]))
		x_expand = self.bridge(x_contract[-1])
		up_idx = len(x_contract)
		
		x_expand_sic = x_expand
		x_expand_sod = x_expand
		x_expand_floe = x_expand

		for expand_block in self.expand_sic_blocks:
			x_expand_sic = expand_block(x_expand_sic, x_contract[up_idx - 1])
			up_idx -= 1

		up_idx = len(x_contract)
		for expand_block in self.expand_sod_blocks:
			x_expand_sod = expand_block(x_expand_sod, x_contract[up_idx - 1])
			up_idx -= 1
			
		up_idx = len(x_contract)
		for expand_block in self.expand_floe_blocks:
			x_expand_floe = expand_block(x_expand_floe, x_contract[up_idx - 1])
			up_idx -= 1

		return {'SIC': self.regression_layer(x_expand_sic.permute(0,2,3,1)),
				'SOD': self.sod_feature_map(x_expand_sod),
				'FLOE': self.floe_feature_map(x_expand_floe)}

