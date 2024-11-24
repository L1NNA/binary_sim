import torch
import torch.nn as nn
import torch.nn.functional as F


class PrefixSFT(nn.Module):

    def __init__(self, backbone, prefix_length = 5, hidden_size = 768):
        super(PrefixSFT, self).__init__()
        self.embedding = backbone.embed_tokens
        self.dtype = backbone.dtype
        self.model = backbone
        self.prefix_length = prefix_length
        self.hidden_size = hidden_size
        self.prefix_embedding_x = nn.Parameter(torch.randn(self.prefix_length, self.hidden_size)).to(backbone.device).to(self.dtype)
        self.prefix_embedding_y = nn.Parameter(torch.randn(self.prefix_length, self.hidden_size)).to(backbone.device).to(self.dtype)


    def single_forward(self, b, input_ids, attention_mask, x_or_y):
        # expand prefix_embedding to [b, prefix_length, hidden_size]
        if x_or_y == 'x':
            prefix_embedding = self.prefix_embedding_x
        else:
            prefix_embedding = self.prefix_embedding_y
        
        prefix_embedding.unsqueeze(0).expand([b, -1, -1]).to(input_ids.device)
        
        ## compute input embedding
        embedded = self.embedding(input_ids)

        ## concatenate prefix_embedding with input embeddings
        input_embeds = torch.cat((prefix_embedding, embedded), dim=1)

        ## expand attention mask
        extended_attention_mask = torch.cat(
            [torch.ones(b, self.prefix_length, device=input_ids.device), attention_mask], dim=1
        )

        ## call backbone forward with added prefix and get outputs
        outputs = self.model(inputs_embeds = input_embeds, attention_mask = extended_attention_mask).last_hidden_state[:, -1, :]
        return outputs


    def forward(self, input_ids, attention_mask, y_input_ids=None, y_attention_mask=None, labels=None):
        ## flattened, input is [batch, max_length]
        b, w= input_ids.size()
        loss = None
        x_output = self.single_forward(b, input_ids, attention_mask, 'x')
        if y_input_ids is not None:
            y_output = self.single_forward(b, y_input_ids, y_attention_mask, 'y')
            pred = F.cosine_similarity(x_output, y_output)
            if labels is not None:
                loss = F.cosine_embedding_loss(x_output, y_output, labels)
        else:
            pred = x_output
        
        
        return {
            "preds": pred,
            "loss": loss
        }
    
    
class Sanity(nn.Module):

    def __init__(self, backbone, prefix_length = 5, hidden_size = 768):
        super(Sanity, self).__init__()
        self.model = backbone


    def forward(self, input_ids, attention_mask, y_input_ids=None, y_attention_mask=None, labels=None):
        ## flattened, input is [batch, max_length]
        b, w= input_ids.size()
        loss = None
        x_output = self.model(input_ids, attention_mask).last_hidden_state[:, -1, :]
        if y_input_ids is not None:
            y_output = self.model(y_input_ids, y_attention_mask).last_hidden_state[:, -1, :]
            pred = F.cosine_similarity(x_output, y_output)
            if labels is not None:
                loss = F.cosine_embedding_loss(x_output, y_output, labels)
        else:
            pred = x_output
        
        
        return {
            "preds": pred,
            "loss": loss
        }