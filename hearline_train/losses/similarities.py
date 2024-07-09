import torch
import torch.nn as nn


class BilinearSimilarity(nn.Module):
    def __init__(self, n_embedding=512):
        super(self.__class__, self).__init__()
        """W:(n_embedding, n_embedding)"""
        self.bilinear = nn.Linear(n_embedding, n_embedding, bias=False)

    def forward(self, anchor, positive):
        """ "bilinear_similarity = anchor.T x W x positive"""
        # anchor_product = anchor x W
        # (B, n_embedding) = (B, n_embedding) x (n_embedding, n_embedding)
        anchor_product = self.bilinear(anchor)
        # bilinear_similarity = anchor_product x positive.T
        # (B, B) = (B, n_embedding) x (n_embedding, B)
        bilinear_similarity = torch.mm(anchor_product, positive.T)
        return bilinear_similarity


class CosineSimilarity(nn.Module):
    def __init__(self, temperature=0.2):
        super(self.__class__, self).__init__()
        self.temperature = temperature

    def forward(self, anchor, positive):
        """ "cosine_similarity = anchor.T x positive / |L2|"""
        # d: 同士の内積を要素とする行列
        d = torch.mm(anchor, positive.T)
        # コサイン類似度の分母に入れるための大きさの平方根
        anchor_l2 = torch.sqrt(torch.mul(anchor, anchor).sum(axis=1, keepdims=True))
        positive_l2 = torch.sqrt(
            torch.mul(positive, positive).sum(axis=1, keepdims=True)
        )
        return d / anchor_l2 / positive_l2 / self.temperature
