import torch
import torch.nn as nn
from models.cnn_backbone import CNNBackbone
from models.mix_transformer import MixTransformer
from models.hybrid_attention import HybridAttention

class PCHM(nn.Module):
    """Parallel Cross-Hybrid Modeling Module"""
    def __init__(self, config):
        super(PCHM, self).__init__()
        
        # CNN Backbone for local feature extraction
        self.cnn_backbone = CNNBackbone(
            in_channels=config['in_channels'],
            stem_channels=config['stem_channels'],
            layer_channels=config['layer_channels'],
            kernel_sizes=config['kernel_sizes']
        )
        
        # Mix-Transformer for global dependency modeling
        self.transformer_backbone = MixTransformer(
            in_channels=config['in_channels'],
            embed_dim=config['embed_dim'],
            depths=config['depths'],
            num_heads=config['num_heads'],
            window_size=config['window_size'],
            mlp_ratio=config['mlp_ratio']
        )
        
        # Hybrid Attention for feature fusion
        self.hybrid_attention = HybridAttention(
            in_channels=config['layer_channels'][-1] + config['embed_dim'] * (2 ** (len(config['depths']) - 1)),
            num_groups=8
        )
        
        # Classification head for BP category
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(config['layer_channels'][-1] + config['embed_dim'] * (2 ** (len(config['depths']) - 1)), 4)
        )
        
    def forward(self, x):
        # x: [B, C, L] where C=4 (ECG, PPG_RED, PPG_IR, PPG_VELOCITY)
        
        # Extract features from both backbones
        cnn_features = self.cnn_backbone(x)  # [B, C1, L1]
        transformer_features = self.transformer_backbone(x)  # [B, C2, L2]
        
        # Concatenate features
        combined_features = torch.cat([cnn_features, transformer_features], dim=1)  # [B, C1+C2, L]
        
        # Apply hybrid attention
        attended_features = self.hybrid_attention(combined_features)  # [B, C1+C2, L]
        
        # Classify BP category
        bp_category = self.classifier(attended_features)  # [B, 4]
        
        return attended_features, bp_category

