import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.ce_criterion = nn.BCELoss()

    def forward(self, logits, label):
        label = label / torch.sum(label, dim=1, keepdim=True) + 1e-10
        loss = -torch.mean(torch.sum(label * F.log_softmax(logits, dim=1), dim=1), dim=0)
        return loss


class GeneralizedCE(nn.Module):
    def __init__(self, q):
        self.q = q
        super(GeneralizedCE, self).__init__()

    def forward(self, logits, label):
        assert logits.shape[0] == label.shape[0]
        assert logits.shape[1] == label.shape[1]
        pos_factor = torch.sum(label, dim=1) + 1e-7
        neg_factor = torch.sum(1 - label, dim=1) + 1e-7
        first_term = torch.mean(torch.sum(((1 - (logits + 1e-7) ** self.q) / self.q) * label, dim=1) / pos_factor)
        second_term = torch.mean(
            torch.sum(((1 - (1 - logits + 1e-7) ** self.q) / self.q) * (1 - label), dim=1) / neg_factor)
        return first_term + second_term


class ContrastiveLoss(nn.Module):
    def __init__(self):
        super(ContrastiveLoss, self).__init__()
        self.ce_criterion = nn.CrossEntropyLoss()

    def NCE(self, q, k, neg, T=0.1):  # 0.1
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        neg = neg.permute(0, 2, 1)
        neg = nn.functional.normalize(neg, dim=1)
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,nck->nk', [q, neg])
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= T
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        loss = self.ce_criterion(logits, labels)

        return loss

    def forward(self, contrast_pairs):
        IA_refinement = self.NCE(
            torch.mean(contrast_pairs['IA'], 1),
            torch.mean(contrast_pairs['CA'], 1),
            contrast_pairs['CB']
        )

        IB_refinement = self.NCE(
            torch.mean(contrast_pairs['IB'], 1),
            torch.mean(contrast_pairs['CB'], 1),
            contrast_pairs['CA']
        )

        CA_refinement = self.NCE(
            torch.mean(contrast_pairs['CA'], 1),
            torch.mean(contrast_pairs['IA'], 1),
            contrast_pairs['CB']
        )

        CB_refinement = self.NCE(
            torch.mean(contrast_pairs['CB'], 1),
            torch.mean(contrast_pairs['IB'], 1),
            contrast_pairs['CA']
        )

        loss = IA_refinement + IB_refinement + CA_refinement + CB_refinement
        return loss
