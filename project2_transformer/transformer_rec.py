"""
基于Transformer的生成式序列推荐模型
核心：因果掩码自注意力机制，实现"用户历史→下一物品"的端到端生成
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class GenerativeRecommender(nn.Module):
    """
    轻量级生成式推荐模型
    用Transformer Decoder实现序列生成
    """
    def __init__(self, num_items, embed_dim=64, num_heads=4, num_layers=2, max_seq_len=10):
        super().__init__()
        self.token_embedding = nn.Embedding(num_items, embed_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(embed_dim, num_items)
        self.max_seq_len = max_seq_len
    
    def forward(self, src):
        B, seq_len = src.shape
        
        # Embedding + 位置编码
        src_emb = self.token_embedding(src)
        pos = torch.arange(seq_len, device=src.device).unsqueeze(0).expand(B, -1)
        pos_emb = self.pos_embedding(pos)
        src_emb = src_emb + pos_emb
        
        # 因果掩码（保证自回归）
        mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
        mask = mask.to(src.device)
        
        output = self.decoder(tgt=src_emb, memory=src_emb, tgt_mask=mask)
        last_output = output[:, -1, :]
        logits = self.output_layer(last_output)
        
        return logits
