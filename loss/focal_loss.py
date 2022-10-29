import torch
import torch.nn as nn
import torch.nn.functional as F
class FocalLoss(nn.Module):
    def __init__(self, alpha=1., gamma =0., reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
    def forward(self, input, target):
        bce_loss = F.binary_cross_entropy(input, target, reduction='none')
        pt = torch.exp(-bce_loss)
        loss = (1-pt) ** self.gamma * bce_loss
        if self.reduction == "mean":
            return loss.mean()
        else:
            return loss.sum()
        
class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, size_average=True):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.size_average = size_average
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        BCE_loss = self.criterion(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.size_average:
            return F_loss.mean()
        else:
            return F_loss.sum()
        
if __name__ == '__main__':
    inputs = torch.ones((1,1,448,448)) * 0.99
    targets = torch.ones((1,1,448,448))
    criterion = FocalLoss()
    loss = criterion(inputs, targets)
    print(loss)