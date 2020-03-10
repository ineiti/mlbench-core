from mlbench_core.evaluation.pytorch.criterion import LabelSmoothing
from torch import nn


def build_criterion(padding_idx, smoothing):
    if smoothing == 0.0:
        criterion = nn.CrossEntropyLoss(ignore_index=padding_idx, size_average=False)
    else:
        criterion = LabelSmoothing(padding_idx, smoothing)

    return criterion
