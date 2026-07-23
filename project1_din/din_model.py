"""
基于DIN的序列CTR预估模型
核心：目标注意力池化（Target Attention Pooling）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DIN(nn.Module):
    """
    Deep Interest Network
    通过目标注意力机制对用户历史行为序列进行自适应聚合
    """
    def __init__(self, num_users, num_items, emb_dim=32, max_seq_len=20):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, emb_dim, padding_idx=0)
        self.item_emb = nn.Embedding(num_items, emb_dim, padding_idx=0)
        self.max_seq_len = max_seq_len
        
        # 目标注意力网络
        self.attn_fc = nn.Sequential(
            nn.Linear(emb_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # 预测网络
        self.fc = nn.Sequential(
            nn.Linear(emb_dim * 3, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def attention(self, hist_emb, target_emb):
        """
        计算历史序列中每个物品与目标物品的注意力权重
        """
        B, seq_len, emb_dim = hist_emb.shape
        target_emb = target_emb.unsqueeze(1).expand(-1, seq_len, -1)
        concat = torch.cat([hist_emb, target_emb], dim=-1)
        attn_weights = self.attn_fc(concat).squeeze(-1)
        
        # mask padding位置
        mask = (hist_emb.sum(dim=-1) != 0).float()
        attn_weights = attn_weights * mask
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        hist_pooled = torch.bmm(attn_weights.unsqueeze(1), hist_emb).squeeze(1)
        return hist_pooled
    
    def forward(self, user_id, hist_items, target_item):
        user_emb = self.user_emb(user_id)
        hist_emb = self.item_emb(hist_items)
        target_emb = self.item_emb(target_item)
        
        hist_pooled = self.attention(hist_emb, target_emb)
        combined = torch.cat([user_emb, hist_pooled, target_emb], dim=-1)
        
        return self.fc(combined).squeeze(-1)
