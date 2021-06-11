
import torch.nn as nn


def get_criterion(pred, target):
    loss = nn.BCELoss(reduction="none")
#     loss = nn.BCEWithLogitsLoss(reduction="none") # for Bi
    return loss(pred, target)