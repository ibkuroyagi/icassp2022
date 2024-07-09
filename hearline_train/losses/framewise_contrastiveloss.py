import torch


class WideArangeCrossEntropy(torch.nn.Module):
    def __init__(self, t=0, embed_time_frame=12, reduction="mean", device="cpu"):
        super(self.__class__, self).__init__()
        """Wide arange cross entropy.
        Args:
            t: Aggregate mask size. t=0 as same as F.cross_entropy
            embed_time_frame: To reduce time, make mask at initialized time.
            reduction: mean, sum, none
        """
        self.t = t
        self.reduction = reduction
        mask = torch.eye(embed_time_frame).to(device)
        ones = torch.ones(embed_time_frame).to(device)
        for i in range(1, t + 1):
            mask += torch.diag(ones[i:], -i) + torch.diag(ones[i:], i)
        self.mask = mask
        self.repeat = mask.sum(1)

    def forward(self, framewise_similarity):
        """framewise_similarity: (T, T)
        tensor([[1.0000, 0.7137, 0.8940, 0.6736, 0.5624],
        [1.0325, 1.0000, 1.0394, 0.6507, 0.6672],
        [0.8883, 0.7138, 1.0000, 0.7158, 0.4635],
        [1.0834, 0.7234, 1.1587, 1.0000, 0.5474],
        [1.5063, 1.2352, 1.2493, 0.9115, 1.0000]])
        """
        numerator = (self.mask * framewise_similarity).sum(1)
        cross_entropy = (
            -numerator + torch.logsumexp(framewise_similarity, dim=1) * self.repeat
        )
        cross_entropy /= self.repeat
        if self.reduction == "mean":
            return cross_entropy.mean()
        elif self.reduction == "sum":
            return cross_entropy.sum()
        elif self.reduction == "none":
            return cross_entropy


class FramewiseContrastiveLoss(torch.nn.Module):
    def __init__(self, similar_fc, framewise_criterion):
        super(self.__class__, self).__init__()
        self.similar_fc = similar_fc
        self.framewise_criterion = framewise_criterion

    def forward(self, anchors, positives):
        """
        Args:
            anchors: (B, T, embed_time_frame)
            positives: (B, T, embed_time_frame)
        """
        for i, (anchor, positive) in enumerate(zip(anchors, positives)):
            framewise_similarity = self.similar_fc(anchor, positive)
            if i == 0:
                loss = self.framewise_criterion(framewise_similarity)
            else:
                loss += self.framewise_criterion(framewise_similarity)
        return loss / (i + 1)
