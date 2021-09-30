import math
import os
import logging
import torch


def count_parameters(model):
    """
    count the number of parameters in a model
    :param model: pytorch Module
    :return: integer number of parameters in model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
