import torch
import torch.nn as nn
from vit_dynamic import vit_register_dynamic


class ClassificationLoss(nn.Module):
    def __init__(self):
        super(ClassificationLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, outputs, targets):
        loss = self.criterion(outputs, targets)
        return loss


# # Example usage:
# loss_fn = ClassificationLoss()
# outputs = vit_register_dynamic(inputs)
# loss = loss_fn(outputs, targets)
