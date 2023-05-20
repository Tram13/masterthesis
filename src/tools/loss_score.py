import torch
from torch import nn


# Implements custom loss functions: MSE_variation and class_based_loss
class LossScore(nn.Module):
    def __init__(self):
        super(LossScore, self).__init__()

    def forward(self, output, target):
        return self.class_based_loss(output, target)

    def MSE_variation(self, output, target):
        # Variation on MSE
        loss = torch.square(target - output)
        # Higher loss for misprediction of low rated items
        is_3_or_less_stars_penalty = (target <= 0.5).float() * 0.5
        return (loss + is_3_or_less_stars_penalty.float()).mean()

    def class_based_loss(self, output, target):
        loss = torch.abs(target - output)
        accuracy = (loss > 0.125).float() * 0.5  # Penalty if wrong class
        return (loss + accuracy).mean()
