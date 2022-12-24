import math
import torch
import torch.nn as nn

def get_attention_pad_mask(embedding):
    batch_size, seq_len, feature_size = embedding.size()

    atten_pad_mask = embedding.data.eq(0).int()
    atten_pad_mask= torch.sum(atten_pad_mask, dim=2).eq(feature_size).unsqueeze(1)
    return atten_pad_mask.expand(batch_size, seq_len, seq_len)

def get_scaledDotProductAttention(Q, K, V, attention_pad_mask):
        #Q, K, V shape: Batch x n_head x seq_len x d_model
        d_k = K.size(-1)
        
        attention_score = torch.matmul(Q, K.transpose(-1, -2)) #Q·K.T
        attention_score /= math.sqrt(d_k) #(Q·K.T) / sqrt(d_k)
        
        attention_score.masked_fill_(attention_pad_mask, -1e9) #pad 위치 column -1e9로 변경
        attention_prob = nn.functional.softmax(attention_score, dim=-1)
        query_attention = torch.matmul(attention_prob, V) #Batch x n_head x seq_len x d_model
        return query_attention

def GELU(x):
    return x * 0.5 * (1. + torch.erf(x / math.sqrt(2.)))