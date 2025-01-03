import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
# from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
from torch import Tensor
from typing import Optional, Tuple, Union
from einops import einsum, rearrange

def scaled_dot_product_attention(q, k, v, masking=None):
    """
    The Attention mechanism
    """
    d_k = k.size(-1)
    weights = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)
    if masking is not None:
        weights = weights.masked_fill_(masking==0, -1e9)
    attn = F.softmax(weights, dim=-1)
    output = torch.matmul(attn, v)
    return output, attn

class FFN(nn.Module):
    """
    Feed forward net
    """

    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super(FFN, self).__init__()

        self.linear_1 = nn.Linear(d_model, d_ff)
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x):
        x = self.linear_1(x)
        x = F.relu(x)
        x = self.linear_2(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x
    
def generate_alibi_bias(nq, nk, q_heads, kv_heads, device = 'cuda', dtype = torch.bfloat16): # b g h n s, where n = s
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_heads = q_heads // kv_heads
    # Create relative positions matrix: [seq_len, seq_len]
    relative_positions = torch.arange(nk, device=device).unsqueeze(0).to(dtype=dtype) - torch.arange(nk, device=device).unsqueeze(1).to(dtype=dtype)
    relative_positions = -torch.abs(relative_positions) 
    relative_positions = relative_positions[-nq:]
    relative_positions = rearrange(relative_positions, 'n m -> () () () n m')  # [1, 1, 1, seq_len, seq_len]
    relative_positions = relative_positions.expand(-1, num_heads, -1, -1, -1)  # [1, num_heads, 1, seq_len, seq_len]

    # Define head-specific slopes (different for each head)
    slopes = torch.tensor([2 ** (-i / num_heads) for i in range(num_heads)], device=device, dtype = dtype).view(1, -1, 1, 1, 1) # compute slopes for each head

    # Apply linear bias with slopes
    alibi_bias = slopes * relative_positions  # [1, num_heads, 1, seq_len, seq_len]

    return alibi_bias

def generate_blk_bias(q_heads, kv_heads, blk_mask, device = 'cuda', dtype = torch.bfloat16, use_log = False, flexattn = False, alibi=False):
    
    num_heads = q_heads // kv_heads if alibi else q_heads

    for ex in blk_mask:
        temp2 = torch.arange(ex.size(0), device = device)[ex.bool()]
        temp3 = temp2[1:] - temp2[:-1] 
        temp3 = torch.cat((temp2[:1], temp3))
        if use_log:
            temp3 = torch.log(temp3).long()
        ex[temp2] = temp3
    
    if alibi:
        blk_mask = rearrange(blk_mask, 'b s -> b () () () s')
        # blk_mask = blk_mask.expand(-1, num_heads, -1, -1, -1)  # [bsz, num_heads, 1, 1, seq_len]
    elif flexattn:
        blk_mask = rearrange(blk_mask, 'b s -> b () s').expand(-1, num_heads, -1)  # [bsz, num_heads, seq_len, 1]
        slopes = torch.tensor([2 ** (-i / num_heads) for i in range(num_heads)], device=device, dtype = dtype).view(1, -1, 1) # compute slopes for each head
        blk_mask = slopes * blk_mask  # [1, num_heads, 1, seq_len]
    else:
        blk_mask = rearrange(blk_mask, 'b s -> b () s ()').expand(-1, num_heads, -1, -1)  # [bsz, num_heads, seq_len, 1]
        slopes = torch.tensor([2 ** (-i / num_heads) for i in range(num_heads)], device=device, dtype = dtype).view(1, -1, 1, 1) # compute slopes for each head
        blk_mask = slopes * blk_mask  # [bsz, num_heads, seq_len, 1]
        # blk_mask = blk_mask.expand(-1, num_heads, -1, -1)  # [1, num_heads, 1, 1, seq_len]

    
    

    
    

    # Define head-specific slopes (different for each head)
    # slopes = torch.tensor([2 ** (-i / num_heads) for i in range(num_heads)], device=device, dtype = dtype).view(1, -1, 1, 1, 1) # compute slopes for each head

    # Apply linear bias with slopes
    # blk_bias = slopes * blk_mask  # [bsz, num_heads, 1, 1, seq_len]

    return blk_mask

def scaled_dot_product_gqa(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    dropout: float = 0.0,
    scale: Optional[float] = None,
    attention_mask: Optional[Tensor] = None, # b x n x s
    mask: Optional[Tensor] = None,
    is_causal: Optional[bool] = None,
    need_weights: bool = False,
    average_attn_weights: bool = False,
    use_alibi = True,
    blk_mask = None,
):
    """Scaled dot product attention with support for grouped queries.

    Einstein notation:
    - b: batch size
    - n / s: sequence length
    - h: number of heads
    - g: number of groups
    - d: dimension of query/key/value

    Args:
        query: Query tensor of shape (b, n, h, d)
        key: Key tensor of shape (b, s, h, d)
        value: Value tensor of shape (b, s, h, d)
        dropout: Dropout probability (default: 0.0)
        scale: Scale factor for query (default: d_query ** 0.5)
        mask: Mask tensor of shape (b, n, s) or (b, s). If 'ndim == 2', the mask is
            applied to all 'n' rows of the attention matrix. (default: None)
        force_grouped: If True, apply grouped-query attention even if the number of
            heads is equal for query, key, and value. (default: False)

    Returns:
        2-tuple of:
        - Attention output with shape (b, n, h, d)
        - (Optional) Attention weights with shape (b, h, n, s). Only returned if
          'need_weights' is True.
    """
    if (mask is not None) and (is_causal is not None):
        raise ValueError(
            "Only one of 'mask' and 'is_causal' should be provided, but got both."
        )
    elif not query.ndim == key.ndim == value.ndim == 4:
        raise ValueError(
            f"Expected query, key, and value to be 4-dimensional, but got shapes "
            f"{query.shape}, {key.shape}, and {value.shape}."
        )

    # Move sequence length dimension to axis 2.
    # This makes the attention operations below *much* faster.
    # query = rearrange(query, "b n h d -> b h n d")
    # key = rearrange(key, "b s h d -> b h s d")
    # value = rearrange(value, "b s h d -> b h s d")

    bq, hq, nq, dq = query.shape
    bk, hk, nk, dk = key.shape
    bv, hv, nv, dv = value.shape
    if not (bq == bk == bv and dq == dk == dv):
        raise ValueError(
            "Expected query, key, and value to have the same batch size (dim=0) and "
            f"embedding dimension (dim=3), but got query: {query.shape}, "
            f"key: {key.shape}, and value: {value.shape}."
        )
    elif (hk != hv) or (nk != nv):
        raise ValueError(
            "Expected key and value to have the same size in dimensions 1 and 2, but "
            f"got key: {key.shape} and value: {value.shape}."
        )
    elif hq % hk != 0:
        raise ValueError(
            "Expected query heads to be a multiple of key/value heads, but got "
            f"query: {query.shape} and key/value: {key.shape}."
        )

    if scale is None:
        scale = query.size(-1) ** 0.5
    query = query / scale

    num_head_groups = hq // hk
    query = rearrange(query, "b (h g) n d -> b g h n d", g=num_head_groups)
    similarity = einsum(query, key, "b g h n d, b h s d -> b g h n s")

    ### apply ALiBi
    if use_alibi and blk_mask is None:
        alibi = generate_alibi_bias(nq, nk, hq, hk, device = similarity.device, dtype = similarity.dtype)
        similarity = similarity + alibi

    if use_alibi and blk_mask is not None:
        
        alibi = generate_alibi_bias(nq, nk, hq, hk, device = similarity.device, dtype = similarity.dtype)
        blk_mask = generate_blk_bias(hq, hk, blk_mask, device = similarity.device, dtype = similarity.dtype)
        similarity = similarity + blk_mask + alibi

    # if is_causal:
    #     # Mask out the upper triangular portion of the attention matrix. This prevents
    #     # the model from attending to tokens in the future.
    #     mask = torch.ones((bq, nq, nk), device=query.device, dtype=torch.bool).tril_()
    #     mask = mask * attention_mask  ### apply attention mask

    if mask is not None:
        # Expand mask to match the shape of the attention matrix.
        # If mask is 2D, assume that it is applied to the key/value sequence dimension.
        # Else if mask is 3D, assume that it is applied to the query/key/value sequence
        # dimension for all attention heads.
        #
        # Users could also provide a 4D mask, which is applied to the query/key/value
        # sequence dimension for each attention head (though I don't have a particular
        # use case in mind for that).
        if mask.ndim == 2:
            mask = rearrange(mask, "b s -> b () () () s")
        elif mask.ndim == 3:
            mask = rearrange(mask, "b n s -> b () () n s")
        # Mask similarity values by setting them to negative infinity.  This guarantees
        # that they will not contribute to the softmax computation below.
        # similarity.masked_fill_(mask == 0, torch.finfo(similarity.dtype).min)

        # if blk_mask is not None:
        #     blk_mask = rearrange(blk_mask, "b s -> b () () () s")
        #     mask *= blk_mask
        similarity += mask

    attention = F.softmax(similarity, dim=-1, dtype=torch.float32).to(query.dtype)
    if dropout > 0.0:
        attention = F.dropout(attention, p=dropout)

    # Apply attention matrix to the value Tensor.
    out = einsum(attention, value, "b g h n s, b h s d -> b g h n d")
    # Move head dimension back to axis 2
    out = rearrange(out, "b g h n d -> b n (h g) d").contiguous()

    attn_weights: Optional[Tensor] = None
    if need_weights:
        # Move the sequence dimensions back to positions 1, 2.  Move the head dimension
        # to position 3.  This more closely matches the return shape of the attention
        # output: (b, n, h, d).
        attn_weights = rearrange(attention, "b g h n s -> b n s (h g)")
        if average_attn_weights:
            attn_weights = attn_weights.mean(dim=1)

    return out, attn_weights

    
class GroupedQueryAttentionWithALiBi(nn.Module):
    def __init__(self, config, layer_idx):
        super(GroupedQueryAttentionWithALiBi, self).__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.d_model = config.hidden_size
        self.q_heads = config.num_attention_heads
        self.kv_heads = config.kv_heads
        self.dropout = config.dropout
        self.d_ff = config.intermediate_size
        self.gamma_init = config.gamma_init
        self.is_causal = None
        self.head_dim = self.d_model // self.q_heads
        dtype = torch.bfloat16

        if self.q_heads % self.kv_heads != 0:
            raise ValueError(
                f"query_heads ({self.q_heads}) must be divisible by "
                f"kv_heads ({self.kv_heads})"
            )
        elif (self.d_model % self.q_heads != 0) or (self.d_model % self.kv_heads != 0):
            raise ValueError(
                f"embed_dim ({self.d_model}) must be divisible by "
                f"query_heads ({self.d_model}) and kv_heads ({self.kv_heads})"
            )
        
        if not self.head_dim % 8 == 0:
            raise ValueError(
                f"head_dim (embed_dim / num_heads = {self.head_dim}) must be divisible by 8"
            )
        if not self.head_dim <= 128:
            raise ValueError(
                f"head_dim (embed_dim / num_heads = {self.head_dim}) must be <= 128"
            )


        # Query, Key, and Value projections
        # self.q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        # self.k_proj = nn.Linear(self.d_model, self.head_dim*self.kv_heads, bias=False)
        # self.v_proj = nn.Linear(self.d_model, self.head_dim*self.kv_heads, bias=False)


        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=True)
        self.k_proj = nn.Linear(self.d_model, self.head_dim*self.kv_heads, bias=True)
        self.v_proj = nn.Linear(self.d_model, self.head_dim*self.kv_heads, bias=True)

        self.out_proj = nn.Linear(
            self.d_model, self.d_model, bias = False
        )

        # self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_normal_(self.q_proj.weight)
        if self.q_proj.bias is not None:
            nn.init.constant_(self.q_proj.bias, 0)
        nn.init.xavier_normal_(self.k_proj.weight)
        if self.k_proj.bias is not None:
            nn.init.constant_(self.k_proj.bias, 0)

        # NOTE: We follow the initialization strategy from MAGNETO.  See:
        # https://arxiv.org/pdf/2210.06423.pdf, Fig. 2
        # Gain (self.gamma_init) should be provided as a keyword argument when
        # initializing the larger Transformer model, since it requires knowledge
        # of the number of encoder/decoder layers in the model.

        nn.init.xavier_normal_(self.v_proj.weight, gain=self.gamma_init)
        if self.v_proj.bias is not None:
            nn.init.constant_(self.v_proj.bias, 0)
        nn.init.xavier_normal_(self.out_proj.weight, gain=self.gamma_init)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0)


    def forward(self, hidden_states, 
                attention_mask=None, 
                position_ids=None, 
                past_key_values=None, 
                output_attentions=False, 
                use_cache=False,
                use_flash_attn = False,
                use_alibi = True,
                blk_mask = None):
        b_size, seq_len, _ = hidden_states.size()
        # attention_mask = attention_mask.unsqueeze(1)  #---> [b, 1, seq_len]
        q = self.q_proj(hidden_states).reshape(b_size, seq_len, self.q_heads, self.head_dim).transpose(1,2) #[b, seq_len, q_heads, head_dim] 
        k = self.k_proj(hidden_states).reshape(b_size, seq_len, self.kv_heads, self.head_dim).transpose(1,2) #[b, seq_len, kv_heads, head_dim] 
        v = self.v_proj(hidden_states).reshape(b_size, seq_len, self.kv_heads, self.head_dim).transpose(1,2) #[b, seq_len, kv_heads, head_dim] 

        if past_key_values is not None:
            k, v = past_key_values.update(k, v, self.layer_idx)

        if use_flash_attn:
            assert output_attentions == False, "output_attentions is not supported with flash_attn"
            output = flash_attn_func(q, k, v, dropout_p=0.1, softmax_scale=None, causal=True,
                window_size=(-1, -1), alibi_slopes=None, deterministic=False)
        else:
            # q = rearrange(q, 'b n q d -> b q n d')  #---> [b, q_heads, seq_len, head_dim]
            # k = rearrange(k, 'b n k d -> b k n d')  #---> [b, kv_heads, seq_len, head_dim]
            # v = rearrange(v, 'b n v d -> b v n d')  #---> [b, kv_heads, seq_len, head_dim]
            output, attn = scaled_dot_product_gqa(q, k, v, 
                                                  mask=attention_mask, 
                                                  dropout = self.dropout, 
                                                  need_weights=output_attentions,
                                                  use_alibi = use_alibi,
                                                  blk_mask = blk_mask)


        output = rearrange(output, 'b n h d -> b n (h d)')  #---> [b, seq_len, d_model]
        output = self.out_proj(output)

        if output_attentions:
            attn = None
        
        return output, attn, past_key_values
    
class BlockAttentionWithALiBi(nn.Module):
    def __init__(self, config, layer_idx):
        super(BlockAttentionWithALiBi, self).__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.d_model = config.hidden_size
        self.q_heads = config.num_attention_heads
        self.kv_heads = config.kv_heads
        self.dropout = config.dropout
        self.d_ff = config.intermediate_size
        self.gamma_init = config.gamma_init
        self.is_causal = None
        self.head_dim = self.d_model // self.q_heads
        dtype = torch.bfloat16

        if self.q_heads % self.kv_heads != 0:
            raise ValueError(
                f"query_heads ({self.q_heads}) must be divisible by "
                f"kv_heads ({self.kv_heads})"
            )
        elif (self.d_model % self.q_heads != 0) or (self.d_model % self.kv_heads != 0):
            raise ValueError(
                f"embed_dim ({self.d_model}) must be divisible by "
                f"query_heads ({self.d_model}) and kv_heads ({self.kv_heads})"
            )
        
        if not self.head_dim % 8 == 0:
            raise ValueError(
                f"head_dim (embed_dim / num_heads = {self.head_dim}) must be divisible by 8"
            )
        if not self.head_dim <= 128:
            raise ValueError(
                f"head_dim (embed_dim / num_heads = {self.head_dim}) must be <= 128"
            )


        # Query, Key, and Value projections
        # self.q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        # self.k_proj = nn.Linear(self.d_model, self.head_dim*self.kv_heads, bias=False)
        # self.v_proj = nn.Linear(self.d_model, self.head_dim*self.kv_heads, bias=False)


        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=True)
        self.k_proj = nn.Linear(self.d_model, self.head_dim*self.kv_heads, bias=True)
        self.v_proj = nn.Linear(self.d_model, self.head_dim*self.kv_heads, bias=True)

        self.out_proj = nn.Linear(
            self.d_model, self.d_model, bias = False
        )

        # self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_normal_(self.q_proj.weight)
        if self.q_proj.bias is not None:
            nn.init.constant_(self.q_proj.bias, 0)
        nn.init.xavier_normal_(self.k_proj.weight)
        if self.k_proj.bias is not None:
            nn.init.constant_(self.k_proj.bias, 0)

        # NOTE: We follow the initialization strategy from MAGNETO.  See:
        # https://arxiv.org/pdf/2210.06423.pdf, Fig. 2
        # Gain (self.gamma_init) should be provided as a keyword argument when
        # initializing the larger Transformer model, since it requires knowledge
        # of the number of encoder/decoder layers in the model.

        nn.init.xavier_normal_(self.v_proj.weight, gain=self.gamma_init)
        if self.v_proj.bias is not None:
            nn.init.constant_(self.v_proj.bias, 0)
        nn.init.xavier_normal_(self.out_proj.weight, gain=self.gamma_init)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0)


    def forward(self, hidden_states, 
                attention_mask=None, 
                position_ids=None, 
                past_key_values=None, 
                output_attentions=False, 
                use_cache=False,
                use_flash_attn = False,
                use_alibi = True,):
        b_size, seq_len, _ = hidden_states.size()
        # attention_mask = attention_mask.unsqueeze(1)  #---> [b, 1, seq_len]
        q = self.q_proj(hidden_states).reshape(b_size, seq_len, self.q_heads, self.head_dim).transpose(1,2) #[b, seq_len, q_heads, head_dim] 
        k = self.k_proj(hidden_states).reshape(b_size, seq_len, self.kv_heads, self.head_dim).transpose(1,2) #[b, seq_len, kv_heads, head_dim] 
        v = self.v_proj(hidden_states).reshape(b_size, seq_len, self.kv_heads, self.head_dim).transpose(1,2) #[b, seq_len, kv_heads, head_dim] 

        if past_key_values is not None:
            k, v = past_key_values.update(k, v, self.layer_idx)

        if use_flash_attn:
            assert output_attentions == False, "output_attentions is not supported with flash_attn"
            output = flash_attn_func(q, k, v, dropout_p=0.1, softmax_scale=None, causal=True,
                window_size=(-1, -1), alibi_slopes=None, deterministic=False)
        else:
            # q = rearrange(q, 'b n q d -> b q n d')  #---> [b, q_heads, seq_len, head_dim]
            # k = rearrange(k, 'b n k d -> b k n d')  #---> [b, kv_heads, seq_len, head_dim]
            # v = rearrange(v, 'b n v d -> b v n d')  #---> [b, kv_heads, seq_len, head_dim]
            output, attn = scaled_dot_product_gqa(q, k, v, 
                                                  mask=attention_mask, 
                                                  dropout = self.dropout, 
                                                  need_weights=output_attentions,
                                                  use_alibi = use_alibi)


        output = rearrange(output, 'b n h d -> b n (h d)')  #---> [b, seq_len, d_model]
        output = self.out_proj(output)

        if output_attentions:
            attn = None
        
        return output, attn, past_key_values


class CausalTransformerLayer(nn.Module):

    def __init__(self, 
                 config,
                 layer_idx,
                 ):
        super().__init__()
        self.config = config
        self.input_layernorm = nn.RMSNorm(self.config.hidden_size)
        self.attn = GroupedQueryAttentionWithALiBi(config, layer_idx)
        self.post_attention_layernorm = nn.RMSNorm(self.config.hidden_size)
        self.ff1 = LlamaFeedForward(self.config.hidden_size, self.config.intermediate_size)
        self.blk_attn = BlockAttentionWithALiBi(config, layer_idx)

    def forward(self, 
                hidden_states, 
                attention_mask=None,
                position_ids=None,
                past_key_values=None,
                output_attentions=None,
                use_cache=None,
                use_flash_attn = False,
                use_alibi = True,
                blk_mask = None
                ):
        # torch.cuda.synchronize()
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # self attention
        hidden_states, self_attn_weights, present_key_value = self.attn(hidden_states, 
                                                                        attention_mask=attention_mask, 
                                                                        position_ids=position_ids,
                                                                        past_key_values=past_key_values,
                                                                        output_attentions=output_attentions,
                                                                        use_cache=use_cache,
                                                                        use_flash_attn=use_flash_attn,
                                                                        use_alibi=use_alibi,
                                                                        blk_mask = blk_mask,
                                                                        )
        hidden_states = residual + hidden_states  ## residual connection

        # if blk_mask is not None:
        #     blk_hidden_states, blk_attn_weights, blk_kv_value = self.attn(hidden_states, 
        #                                                                 attention_mask=attention_mask, 
        #                                                                 position_ids=position_ids,
        #                                                                 past_key_values=past_key_values,
        #                                                                 output_attentions=output_attentions,
        #                                                                 use_cache=use_cache,
        #                                                                 use_flash_attn=use_flash_attn,
        #                                                                 use_alibi=use_alibi,
        #                                                                 blk_mask=blk_mask,
        #                                                                 )

        # hidden_states += blk_hidden_states

        # fully connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.ff1(hidden_states)
        hidden_states = residual + hidden_states ## residual connection

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs
    
class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.d_k = d_model // n_heads
        self.num_heads = n_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, q, k, v, masking=None):
        batch_size = q.size(0)

        q = self.q_linear(q).reshape(batch_size, -1, self.num_heads, self.d_k)
        k = self.k_linear(k).reshape(batch_size, -1, self.num_heads, self.d_k)
        v = self.v_linear(v).reshape(batch_size, -1, self.num_heads, self.d_k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        output, attn = scaled_dot_product_attention(q, k, v, masking=masking)
        output = output.transpose(1, 2) \
                    .reshape(batch_size, -1, self.num_heads * self.d_k)

        output = self.out(output)
        if self.dropout is not None:
            output = self.dropout(output)
        return output, attn
    

# class TransformerLayer(nn.Module):

#     def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
#         super(TransformerLayer, self).__init__()

#         self.attn = MultiHeadAttention(d_model, n_heads, dropout=dropout)
#         self.norm_1 = nn.LayerNorm(d_model)

#         self.ffn = FFN(d_model, d_ff, dropout=dropout)
#         self.norm_2 = nn.LayerNorm(d_model)

#     def forward(self, x, masking=None):
#         output, attn = self.attn(x, x, x, masking=masking)
#         output = self.norm_1(x + output)

#         output2 = self.ffn(output)
#         output2 = self.norm_2(output2 + output)
#         return output2, attn
    
    
class MultiQueryAttention(nn.Module):
    def __init__(self, emb_dim, n_heads, dtype):
        super(MultiQueryAttention, self).__init__()
        self.emb_dim = emb_dim
        self.n_heads = n_heads
        self.head_dim = emb_dim // n_heads

        # Query, Key, and Value projections
        self.q_proj = nn.Linear(emb_dim, emb_dim, dtype=dtype, bias=False)
        self.k_proj = nn.Linear(emb_dim, self.head_dim, dtype=dtype, bias=False)
        self.v_proj = nn.Linear(emb_dim, self.head_dim, dtype=dtype, bias=False)

        self.out = nn.Linear(emb_dim, emb_dim, dtype=dtype, bias=False)

    def forward(self, x, masking):
        batch_size, seq_len, _ = x.size()

        # Project queries, keys, and values
        q = self.q_proj(x).reshape(batch_size, seq_len,
                        self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        v = self.v_proj(x).unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        
        # Compute attention scores
        attn_output, _ = scaled_dot_product_attention(q, k, v, masking=masking)

        # Concatenate heads and project to output
        attn_output = attn_output.transpose(1, 2) \
            .contiguous().view(batch_size, seq_len, self.emb_dim)
        out = self.out(attn_output)

        return out
    
    
class LlamaFeedForward(nn.Module):
    def __init__(self, emb_dim, ff_dim):
        super().__init__()
        self.gate = nn.Linear(emb_dim, ff_dim, bias=False)
        self.ff1 = nn.Linear(emb_dim, ff_dim,  bias=False)
        self.ff2 = nn.Linear(ff_dim, emb_dim,  bias=False)

    def forward(self, x):
        x_gate = self.gate(x)
        x1 = self.ff1(x)
        gated_x1 = nn.functional.silu(x_gate) * x1
        return self.ff2(gated_x1)
    
    
class TransformerLayer(nn.Module):

    def __init__(self, emb_dim, ff_dim, n_heads, dtype):
        super(TransformerLayer, self).__init__()
        self.norm1 = nn.RMSNorm(emb_dim, dtype=dtype)
        self.attn = MultiQueryAttention(emb_dim, n_heads, dtype)
        self.norm2 = nn.RMSNorm(emb_dim, dtype=dtype)
        self.ffn = LlamaFeedForward(emb_dim, ff_dim, dtype)

    def forward(self, x, mask):
        attn_output = self.attn(self.norm1(x), mask)
        x = x + attn_output
        ffn_output = self.ffn(self.norm2(x))
        x = x + ffn_output
        return x