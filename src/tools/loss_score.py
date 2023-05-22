import torch
from torch import nn


# Implements custom loss functions: MSE_variation and class_based_loss
class LossScore(nn.Module):
    def __init__(self, impl: int):
        super(LossScore, self).__init__()
        self.impl = impl

    def forward(self, output, target):
        if self.impl == 1:
            return self.MSE_variation1(output, target)
        else:
            return self.MSE_variation2(output, target)

    def MSE_variation2(self, output, target):
        # Variation on MSE
        loss = torch.square(target - output)
        # Higher loss for misprediction of low rated items
        is_3_or_less_stars_penalty = (target <= 0.5).float() * 0.6
        is_5_penalty = (target >= 0.99).float() * 0.1
        return (loss + is_3_or_less_stars_penalty.float() + is_5_penalty.float()).mean()

    def MSE_variation1(self, output, target):
        loss = torch.square(target - output)
        loss = torch.mul(loss, 5)
        return loss.mean()

    def class_based_loss(self, output, target):  # Not used, since this is treated as a regression problem
        loss = torch.abs(target - output)
        accuracy = (loss > 0.125).float() * 0.5  # Penalty if wrong class
        return (loss + accuracy).mean()
