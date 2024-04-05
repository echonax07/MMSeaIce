__author__ = 'Muhammed Patel'
__contributor__ = 'Xinwwei chen, Fernando Pena Cantu,Javier Turnes, Eddie Park'
__copyright__ = ['university of waterloo']
__contact__ = ['m32patel@uwaterloo.ca', 'xinweic@uwaterloo.ca']
__version__ = '1.0.0'
__date__ = '2024-04-05'

import torch
from torch import nn
import torch.nn.functional as F


class OrderedCrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index=-100):
        super(OrderedCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, output: torch.Tensor, target: torch.Tensor):

        criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=self.ignore_index)
        loss = criterion(output, target)
        # calculate the hard predictions by using softmax followed by an argmax
        softmax = torch.nn.functional.softmax(output, dim=1)
        hard_prediction = torch.argmax(softmax, dim=1)
        # set the mask according to ignore index
        mask = target == self.ignore_index
        hard_prediction = hard_prediction[~mask]
        target = target[~mask]
        # calculate the absolute difference between target and prediction
        weights = torch.abs(hard_prediction-target) + 1
        # remove ignored index losses
        loss = loss[~mask]
        # if done normalization with weights the loss becomes of the order 1e-5
        # loss = (loss * weights)/weights.sum()
        loss = (loss * weights)
        loss = loss.mean()

        return loss


class MSELossFromLogits(nn.Module):
    def __init__(self, chart, ignore_index=-100):
        super(MSELossFromLogits, self).__init__()
        self.ignore_index = ignore_index
        self.chart = chart
        if self.chart == 'SIC':
            self.replace_value = 11
            self.num_classes = 12
        elif self.chart == 'SOD':
            self.replace_value = 6
            self.num_classes = 7
        elif self.chart == 'FLOE':
            self.replace_value = 7
            self.num_classes = 8
        else:
            raise NameError('The chart \'{self.chart} \'is not recognized')

    def forward(self, output: torch.Tensor, target: torch.Tensor):

        # replace ignore index value(for e.g 255) with a number 11. Becuz one hot encode requires
        # continous numbers (you cant one hot encode 255)
        target = torch.where(target == self.ignore_index,
                             torch.tensor(self.replace_value, dtype=target.dtype,
                                          device=target.device), target)
        # do one hot encoding
        target_one_hot = F.one_hot(target, num_classes=self.num_classes).permute(0, 3, 1, 2)

        # apply softmax on logits
        softmax = torch.softmax(output, dim=1, dtype=output.dtype)

        criterion = torch.nn.MSELoss(reduction='none')

        # calculate loss between softmax and one hot encoded target
        loss = criterion(softmax, target_one_hot.to(softmax.dtype))

        # drop the last channel since it belongs to ignore index value and should not
        # contribute to the loss

        loss = loss[:, :-1, :, :]
        loss = loss.mean()
        return loss

class WaterConsistencyLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.keys = ['SIC', 'SOD', 'FLOE']
        self.activation = nn.Softmax(dim=1)
    
    def forward(self, output):
        sic = self.activation(output[self.keys[0]])[:, 0, :, :]
        sod = self.activation(output[self.keys[1]])[:, 0, :, :]
        floe = self.activation(output[self.keys[2]])[:, 0, :, :]
        return torch.mean((sic-sod)**2 + (sod-floe)**2 + (floe-sic)**2)

# only applicable to regression outputs
class MSELossWithIgnoreIndex(nn.MSELoss):
    def __init__(self, ignore_index=255, reduction='mean'):
        super(MSELossWithIgnoreIndex, self).__init__(reduction=reduction)
        self.ignore_index = ignore_index

    def forward(self, input, target):
        mask = (target != self.ignore_index).type_as(input)
        diff = input.squeeze(-1) - target
        diff = diff * mask
        loss = torch.sum(diff ** 2) / mask.sum()
        return loss

# only applicable to regression outputs
class MSELossWithIgnoreIndex(nn.MSELoss):
    def __init__(self, ignore_index=255, reduction='mean'):
        super(MSELossWithIgnoreIndex, self).__init__(reduction=reduction)
        self.ignore_index = ignore_index

    def forward(self, input, target):
        mask = (target != self.ignore_index).type_as(input)
        diff = input.squeeze(-1) - target
        diff = diff * mask
        loss = torch.sum(diff ** 2) / mask.sum()
        return loss
