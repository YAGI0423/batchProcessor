import torch
import torch.nn as nn

import coBert.layerUtil as layerUtil

class EmbeddingLayer(nn.Module):
    def __init__(self, max_len, d_model, device):
        super(EmbeddingLayer, self).__init__()
        self.device= device

        self.position_emb = nn.Embedding(max_len, d_model)
        self.seg_emb = nn.Embedding(2, d_model)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x, segment_mask):
        pos = self.__get_position_mask(x)
        embedding = x + self.position_emb(pos) + self.seg_emb(segment_mask)
        return self.norm(embedding)
        
    def __get_position_mask(self, x):
        batch_size, seq_len, _ = x.shape
        pos = torch.torch.arange(seq_len, dtype=torch.long, device=self.device)
        pos = pos.unsqueeze(0).repeat(batch_size, 1)
        return pos

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super(MultiHeadAttention, self).__init__()
        self.config = config
        
        self.W_Q = nn.Linear(config['d_model'], config['d_model'] * config['n_heads'])
        self.W_K = nn.Linear(config['d_model'], config['d_model'] * config['n_heads'])
        self.W_V = nn.Linear(config['d_model'], config['d_model'] * config['n_heads'])
        
        self.W_out = nn.Linear(config['d_model'] * config['n_heads'], config['d_model'])
        
    def forward(self, Q, K, V, attention_pad_mask):
        batch_size = Q.size(0)
        
        #Batch x n_head x seq_len x d_model
        #why `transpose`: n_head별로 (seq_len x d_model) · (d_model x seq_len)으로 matmul 하기 위해
        q_s = self.W_Q(Q).view(batch_size, -1, self.config['n_heads'], self.config['d_model']).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, self.config['n_heads'], self.config['d_model']).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, self.config['n_heads'], self.config['d_model']).transpose(1, 2)
        
        #Batch x n_head x seq_len x seq_len
        atten_pad_mask = attention_pad_mask.unsqueeze(1).repeat(1, self.config['n_heads'], 1, 1)

        context = layerUtil.get_scaledDotProductAttention(q_s, k_s, v_s, atten_pad_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.config['n_heads'] * self.config['d_model'])
        #Batch x seq_len x n_head * d_model
        
        output = self.W_out(context)
        return output

class PositionWiseFeedForward(nn.Module):
    def __init__(self, config):
        super(PositionWiseFeedForward, self).__init__()
        self.W_1 = nn.Linear(config['d_model'], config['d_ff'])
        self.W_2 = nn.Linear(config['d_ff'], config['d_model'])
        
        self.dropout = nn.Dropout(config['dropout_rate'])

    def forward(self, x):
        out = layerUtil.GELU(self.W_1(x))
        out = self.dropout(out)
        out = self.W_2(out)
        return out

class AddNorm(nn.Module):
    def __init__(self, config):
        super(AddNorm, self).__init__()
        self.normLayer = nn.LayerNorm(config['d_model'])
        
    def forward(self, x, sub_layer):
        residual = x
        add = sub_layer(x) + residual
        norm = self.normLayer(add)
        return norm

class EncoderLayer(nn.Module):
    def __init__(self, config):
        super(EncoderLayer, self).__init__()
        self.multi_head_atten = MultiHeadAttention(config)
        self.multi_head_addNorm = AddNorm(config)
        
        self.ffnn = PositionWiseFeedForward(config)
        self.ffnn_addNorm = AddNorm(config)
        
    def forward(self, enc_inputs, enc_attention_pad_mask):
        multi_head = self.multi_head_addNorm(
            enc_inputs, sub_layer=lambda x_: self.multi_head_atten(x_, x_, x_, enc_attention_pad_mask)
        )
        feed_forward = self.ffnn_addNorm(multi_head, sub_layer=self.ffnn)
        return feed_forward