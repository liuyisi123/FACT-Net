
import torch
import torch.nn as nn
from models.feature_extractor import DilatedConvExtractor
from models.transformer_encoder_decoder import TransformerEncoderDecoder

class SHCT(nn.Module):
    """Serial Hybrid CNN-Transformer Module"""
    def __init__(self, config):
        super(SHCT, self).__init__()
        
        # Feature extractor with dilated convolutions
        self.feature_extractor = DilatedConvExtractor(
            in_channels=config['in_channels'],
            out_channels=config['out_channels'],
            dilation_rates=config['dilation_rates'],
            kernel_size=config['kernel_size']
        )
        
        # Transformer encoder-decoder
        self.transformer = TransformerEncoderDecoder(
            d_model=config['d_model'],
            nhead=config['nhead'],
            num_encoder_layers=config['num_encoder_layers'],
            num_decoder_layers=config['num_decoder_layers'],
            dim_feedforward=config['dim_feedforward'],
            dropout=config['dropout']
        )
        
        # Feature reconstructor
        self.reconstructor = nn.Sequential(
            nn.Linear(config['d_model'], 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1)
        )
        
    def forward(self, x, bp_category):
        # x: [B, C, L] where C=5 (ECG, PPG_RED, PPG_IR, PPG_VELOCITY, BP_CATEGORY)
        
        # Extract features
        features = self.feature_extractor(x)  # [B, out_channels, L']
        
        # Reshape for transformer
        features = features.permute(2, 0, 1)  # [L', B, out_channels]
        
        # Pass through transformer
        transformed = self.transformer(features)  # [L', B, d_model]
        
        # Reshape back
        transformed = transformed.permute(1, 2, 0)  # [B, d_model, L']
        
        # Reconstruct ABP waveform
        abp = self.reconstructor(transformed.permute(0, 2, 1))  # [B, L', 1]
        abp = abp.squeeze(-1)  # [B, L']
        
        return abp