import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.distributed_utils import mismatched_sizes_all_gather


def info_nce(x_emb, y_emb, scale):
    batch_size = x_emb.size(0)
    # Compute cross cosine similarities: batch_size x batch_size
    similarities = F.cosine_similarity(x_emb.unsqueeze(1), \
        y_emb.unsqueeze(0), dim=2) * scale
    labels = torch.arange(batch_size).to(x_emb.device)
    return similarities, labels


def gte_loss(x_emb, y_emb, scale):
    
    batch_size = x_emb.size(0)
    labels = torch.arange(batch_size, device=x_emb.device)
    xiy = F.cosine_similarity(x_emb.unsqueeze(1), \
        y_emb.unsqueeze(0), dim=2) * scale
    yix = F.cosine_similarity(y_emb.unsqueeze(1), \
        x_emb.unsqueeze(0), dim=2) * scale
    yix[labels, labels] = -torch.inf
    xix = F.cosine_similarity(x_emb.unsqueeze(1), \
        x_emb.unsqueeze(0), dim=2) * scale
    xix[labels, labels] = -torch.inf
    yiy = F.cosine_similarity(y_emb.unsqueeze(1), \
        y_emb.unsqueeze(0), dim=2) * scale
    yiy[labels, labels] = -torch.inf
    
    similarities = torch.cat([
        xiy, yix, xix, yiy
    ], dim=1)
    return similarities, labels
    
    
class SimeCSELoss:
    def __init__(
        self,
        sim_func=gte_loss,
        temperature:float=0.05,
    ):
        self.scale = 1 / temperature
        self.sim_func = sim_func
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def __call__(
        self,
        x_embs,
        y_embs
    ):
        if torch.distributed.is_initialized():
            full_x_embs = mismatched_sizes_all_gather(x_embs)
            full_x_embs = torch.cat(full_x_embs)

            full_y_embs = mismatched_sizes_all_gather(y_embs)
            full_y_embs = torch.cat(full_y_embs)
        else:
            full_x_embs = x_embs
            full_y_embs = y_embs

        similarities, labels = self.sim_func(full_x_embs, full_y_embs, self.scale)
        loss = self.cross_entropy_loss(similarities, labels)
        return loss


def info_nce(x_emb, y_emb, labels=None, temperature=0.05):
    batch_size = x_emb.size(0)
    # Compute cross cosine similarities: batch_size x batch_size
    similarities = F.cosine_similarity(x_emb.unsqueeze(1), \
        y_emb.unsqueeze(0), dim=2) / temperature
    # Log sum exp for stability
    logits = similarities - torch.logsumexp(similarities, dim=1, keepdim=True)
    if labels is not None: # mask gradients to same labels
        mask = labels.unsqueeze(0).repeat(batch_size, 1) == labels.unsqueeze(1)
        torch.diagonal(mask).fill_(False)
        logits.masked_fill_(mask, -torch.inf)
    diag_labels = torch.arange(batch_size).to(x_emb.device)
    return F.cross_entropy(logits, diag_labels)

def align_loss(x, y, alpha=2):
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()

def uniform_loss(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

def cov_loss(x):
    cov = torch.cov(x.transpose(0,1))
    l_cov = torch.norm(cov - torch.eye(cov.size(0)).to(x.device), p='fro')
    return l_cov.to(x.device)

def gte_info_nce(x_emb, y_emb, labels=None, temperature=0.05, anchor_emb=None):
    batch_size = x_emb.size(0)
    # concat two embs and cross sim
    embs = torch.cat([x_emb, y_emb], dim=0)
    cross_sims = F.cosine_similarity(embs.unsqueeze(1), \
        embs.unsqueeze(0), dim=2) / temperature
    # mask diagonal with -torch.inf i.e. mask q_i,q_i and d_i,d_i
    diag_indices = torch.arange(2*batch_size, device=x_emb.device)
    cross_sims[diag_indices, diag_indices] = -torch.inf
    # concat qiqj, qidj, diqj, didj, for some 1 and all j
    cross_sims = torch.cat([cross_sims[:batch_size, :], cross_sims[batch_size:, batch_size:]], dim=1)
    labels = torch.arange(batch_size, batch_size*2, device=x_emb.device)

    if anchor_emb is not None:
        anchor_label = torch.ones(batch_size, device=x_emb.device)
        # logits = anchor_sims - torch.logsumexp(anchor_sims, dim=1, keepdim=True)
        # anchor_labels = torch.arange(batch_size, device=x_emb.device)
        return F.cross_entropy(cross_sims, labels) + cosine_similarity(x_emb, anchor_emb, anchor_label)  + cov_loss(x_emb) + cov_loss(y_emb)
    return F.cross_entropy(cross_sims, labels) #+ 0.1*cov_loss(x_emb) + 0.1*cov_loss(y_emb)#+ 0.2*align_loss(x_emb, y_emb) #+ uniform_loss(embs)


def cosine_similarity(x_emb, y_emb, labels):
    return F.cosine_embedding_loss(x_emb, y_emb, labels)


# def _create_temperatures(labels, logits):
#     b = labels.size(0)
#     temperatures = torch.zeros((b, b), device=logits.device, dtype=logits.dtype)

#     main_channel, auxiliary_channel = 0.05, 1.5
#     temperatures.fill_(main_channel)
#     index = 0
#     mask = labels.unsqueeze(0).repeat(batch_size, 1) == labels.unsqueeze(1)
#     torch.diagonal(mask).fill_(False)
#     temperatures.masked_fill_(mask, auxiliary_channel)
#     return temperatures
