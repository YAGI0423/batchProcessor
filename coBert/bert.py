import torch
import torch.nn as nn

import coBert.layers as layers
import coBert.layerUtil as layerUtil

class BERT(nn.Module):
    def __init__(self, config):
        super(BERT, self).__init__()
        self.config = config

        self.embedding = layers.EmbeddingLayer(config)
        self.enc_layers = nn.ModuleList([layers.EncoderLayer(config) for _ in range(config['n_layers'])])
        
    def forward(self, input_ids, segment_ids):
        #input_ids shape: Batch x seq_len x d_model
        #segment_ids shape: Batch x seq_len

        out = self.embedding(input_ids, segment_ids)

        if self.config['is_cls_embedding']:
            batch_size, _, d_model = input_ids.shape
            cls_emb = torch.ones(batch_size, 1, d_model, device=self.config['device'])
            cls_input_ids = torch.cat([cls_emb, input_ids], dim=1)
    
            enc_atten_pad_mask = layerUtil.get_attention_pad_mask(cls_input_ids)
        else:
            enc_atten_pad_mask = layerUtil.get_attention_pad_mask(input_ids)

        #Input & output shape: Batch x seq_len x d_model
        for layer in self.enc_layers:
            out = layer(out, enc_atten_pad_mask)
        return out