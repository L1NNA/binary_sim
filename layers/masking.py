import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

def create_causal_mask(seq_len, attention_mask, device):
    """
    Create a causal mask for the attention mechanism.
    
    :param seq_len: the sequence length.
    :param attention_mask: A tensor of shape (batch_size, seq_length) 
        containing 1s for non-padding tokens and 0s for padding tokens.
    :param device: The device on which the tensors will be allocated (e.g., 'cpu' or 'cuda').
    :return: A causal mask of shape (batch_size, 1, seq_length, seq_length)
        Trues for masked tokens and Falses for non-masked tokens.
    """
    # Create a causal mask (lower triangular matrix)
    # seq_len x seq_len
    causal_mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device), diagonal=1)

    if attention_mask is not None:
        batch_size, attn_seq_len = attention_mask.size()
        assert attn_seq_len == seq_len 
        # Expand the causal mask to match the batch size
        ### b x 1 x seq_len x seq_len
        causal_mask = causal_mask.expand(batch_size, 1, seq_len, seq_len).clone()
        
        # Apply the padding mask to the causal mask
        ## Expand the attention mask to match the causal mask shape
        ### b x 1 x 1 x seq_len
        expanded_attention_mask = attention_mask[:, None, None, :]
        
        ## Mask locations with 0
        causal_mask.masked_fill_(expanded_attention_mask == 0, True)

    return causal_mask