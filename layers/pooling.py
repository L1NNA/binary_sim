import torch


def mean_pooling(output):
    # b x s x d
    return torch.mean(output, dim=1)

def mask_mean_pooling(output, mask):
    # output: b x seq_len x d
    # mask: b x seq_len
    return (mask.unsqueeze(-1) * output).sum(1)  / mask.sum(-1, keepdim=True)

def cls_pooling(output):
    # b x s x d
    return output[:, 0, :]

def causal_pooling(output):
    # b x s x d -> b x d
    return output[:, -1, :]

def any_max_pooling(output):
    """
    A special pooling technique for binary classification
    where if any token is 1, then the whole
    """
    assert output.size(-1) == 2
    output2 = torch.softmax(output, dim=-1)
    # b
    i = torch.argmax(output2[:, :, 1], dim=1)
    # b x 1 x 2
    i = i.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 2)
    # b x 2
    return output.gather(1, i).squeeze(1)

