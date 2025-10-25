import torch
import torch.nn as nn
from models.pchm import PCHM
from models.shct import SHCT

class FACTNet(nn.Module):
    """Full FACT-Net Architecture"""
    def __init__(self, config):
        super(FACTNet, self).__init__()
        
        # Stage I: Parallel Cross-Hybrid Modeling
        self.stage1 = PCHM(config['stage1'])
        
        # Stage II: Serial Hybrid CNN-Transformer
        self.stage2 = SHCT(config['stage2'])
        
    def forward(self, ecg, ppg_red, ppg_ir, ppg_velocity):
        # Stage I input: [B, 4, L]
        stage1_input = torch.stack([ecg, ppg_red, ppg_ir, ppg_velocity], dim=1)
        
        # Stage I forward pass
        stage1_features, bp_category = self.stage1(stage1_input)
        
        # Prepare Stage II input with BP category
        bp_category_expanded = bp_category.argmax(dim=1, keepdim=True).float()
        bp_category_expanded = bp_category_expanded.unsqueeze(-1).expand(-1, -1, stage1_input.size(-1))
        
        stage2_input = torch.cat([stage1_input, bp_category_expanded], dim=1)  # [B, 5, L]
        
        # Stage II forward pass
        abp_reconstructed = self.stage2(stage2_input, bp_category)
        
        return abp_reconstructed, bp_category

