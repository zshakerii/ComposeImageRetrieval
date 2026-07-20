import torch
import torch.nn as nn


class CrossModalAttentionFusion(nn.Module):
    """ماژول Attention برای ترکیب هوشمند تصویر و متن"""

    def __init__(self, img_dim=768, txt_dim=768, hidden_dim=512, num_heads=8):
        super().__init__()

        # Project به فضای مشترک
        self.img_proj = nn.Linear(img_dim, hidden_dim)
        self.txt_proj = nn.Linear(txt_dim, hidden_dim)

        # Multi-head Cross-Attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # Feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        # Layer Normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, img_feat, txt_feat):
        """
        Args:
            img_feat: (batch, img_dim)
            txt_feat: (batch, txt_dim)
        Returns:
            fused_feat: (batch, hidden_dim)
        """
        # Project
        img = self.img_proj(img_feat).unsqueeze(1)  # (B, 1, H)
        txt = self.txt_proj(txt_feat).unsqueeze(1)  # (B, 1, H)

        # Cross-Attention: تصویر به متن توجه می‌کند
        attn_out, attn_weights = self.cross_attn(
            query=img,
            key=txt,
            value=txt
        )

        # Residual + Norm
        img = self.norm1(img + attn_out)

        # Feed-forward
        ffn_out = self.ffn(img)
        fused = self.norm2(img + ffn_out)

        return fused.squeeze(1), attn_weights


# استفاده در کد اصلی
fusion_module = CrossModalAttentionFusion(
    img_dim=768,  # بسته به مدل
    txt_dim=768,
    hidden_dim=512,
    num_heads=8
).to(device)

# در حلقه retrieval
combined_feat, attn_weights = fusion_module(img_feat, txt_feat)
