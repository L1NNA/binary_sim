import torch
import torch.nn as nn
import torch.nn.functional as F

def info_nce(x_emb, temperature=0.05):
    batch_size = x_emb.size(0)
    labels = torch.arange(batch_size, requires_grad=False) \
                .to(x_emb.device)
    # Compute cosine similarities: batch_size x batch_size
    similarities = F.cosine_similarity(x_emb.unsqueeze(1), \
        x_emb.unsqueeze(0), dim=2) / temperature
    logits = similarities - torch.logsumexp(similarities, dim=1, keepdim=True)
    return F.cross_entropy(logits, labels)

def info_nce_xy(x_emb, y_emb, labels=None, temperature=0.05):
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
    diag_labels = torch.arange(batch_size).to(labels.device)
    return F.cross_entropy(logits, diag_labels)


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
