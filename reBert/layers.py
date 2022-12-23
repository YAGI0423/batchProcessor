import torch
import torch.nn as nn

class EmbeddingLayer(nn.Module):
    def __init__(self, config):
        super(EmbeddingLayer, self).__init__()
        self.config = config
        self.position_emb = nn.Embedding(config['max_len'], config['d_model'])
        self.seg_emb = nn.Embedding(2, config['d_model'])
        
        self.norm = nn.LayerNorm(config['d_model'])
        
    def forward(self, x, segment_mask):
        pos = self.__to_position_mask(x)
        
        embedding = x + self.position_emb(pos) + self.seg_emb(segment_mask)
        return self.norm(embedding)
        
    def __to_position_mask(self, x):
        batch_size, seq_len, _ = x.shape
        pos = torch.torch.arange(seq_len, dtype=torch.long)
        pos = pos.unsqueeze(0).repeat(batch_size, 1)
        pos = pos.to(self.config['device'])
        return pos