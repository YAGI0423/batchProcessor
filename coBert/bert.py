import torch
import torch.nn as nn

import coBert.layers as layers
import coBert.layerUtil as layerUtil

class BERT(nn.Module):
    def __init__(self, config):
        super(BERT, self).__init__()
        self.embedding = layers.EmbeddingLayer(config['max_len'], config['d_model'], config['device'])
        self.enc_layers = nn.ModuleList([layers.EncoderLayer(config) for _ in range(config['n_layers'])])

    def forward(self, input_ids, segment_ids):
        #input_ids shape: Batch x seq_len x d_model
        #segment_ids shape: Batch x seq_len
        enc_atten_pad_mask = layerUtil.get_attention_pad_mask(input_ids)
        out = self.embedding(input_ids, segment_ids)

        #Input & output shape: Batch x seq_len x d_model
        for layer in self.enc_layers:
            out = layer(out, enc_atten_pad_mask)
        return out

class CoBERT(nn.Module):
    def __init__(self, config):
        super(CoBERT, self).__init__()
        config = config.copy()
        if config['is_cls_embedding']:
            config['max_len'] += 1
            self.cls_embedding = nn.Embedding(1, config['d_model'])
        self.config = config

        self.featByEmb = nn.Linear(config['feature_size'], config['d_model'])
        self.bert = BERT(config)
        
    def forward(self, input_ids, segment_ids):
        #input_ids shape: Batch x seq_len x d_model
        #segment_ids shape: Batch x seq_len
        out = layerUtil.GELU(self.featByEmb(input_ids))

        if self.config['is_cls_embedding']:
            batch_size = input_ids.size(0)
            cls_ids = torch.zeros(batch_size, 1, dtype=torch.int32, device=self.config['device'])
            cls_emb = self.cls_embedding(cls_ids)

            out = torch.cat([cls_emb, out], dim=1)
            segment_ids = torch.cat([cls_ids, segment_ids], dim=1)
        out = self.bert(out, segment_ids)
        return out