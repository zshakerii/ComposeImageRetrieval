import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging


# ==========================================
# تنظیمات Logging
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==========================================
# 1. ماژول Fusion پیشرفته
# ==========================================
class MLPFusion(nn.Module):

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 1024,
        output_dim: int = 512,
        dropout_rate: float = 0.1,
        use_residual: bool = True
    ):
        super().__init__()

        self.use_residual = use_residual
        self.output_dim = output_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_dim // 2, output_dim)
        )

        # Residual projection
        self.residual_proj = nn.Linear(
            input_dim * 2,
            output_dim
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        img_feat: torch.Tensor,
        txt_feat: torch.Tensor
    ) -> torch.Tensor:

        # Normalize inputs
        img_feat = F.normalize(img_feat, dim=-1)
        txt_feat = F.normalize(txt_feat, dim=-1)

        # Concatenate
        combined_feat = torch.cat([img_feat, txt_feat], dim=-1)

        # Main branch
        fused_feat = self.net(combined_feat)

        # Residual branch
        if self.use_residual:
            residual = self.residual_proj(combined_feat)
            fused_feat = fused_feat + residual

        # Final normalization
        return F.normalize(fused_feat, dim=-1)


# ==========================================
# 2. دیتاست بهینه
# ==========================================
class CIRDataset(Dataset):

    def __init__(
        self,
        data_list: List[Dict],
        feature_cache: Optional[Path] = None
    ):
        self.data_list = data_list
        self.feature_cache = feature_cache

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(
        self,
        idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        item = self.data_list[idx]

        # در پروژه واقعی:
        # ref_feat = torch.load(...)
        # text_feat = torch.load(...)
        # target_feat = torch.load(...)

        ref_feat = torch.randn(512, dtype=torch.float32)
        text_feat = torch.randn(512, dtype=torch.float32)
        target_feat = torch.randn(512, dtype=torch.float32)

        return ref_feat, text_feat, target_feat


# ==========================================
# 3. Contrastive Loss پیشرفته
# ==========================================
class InfoNCELoss(nn.Module):

    def __init__(
        self,
        temperature: float = 0.07,
        symmetric: bool = True
    ):
        super().__init__()

        self.temperature = temperature
        self.symmetric = symmetric

    def forward(
        self,
        fused_features: torch.Tensor,
        target_features: torch.Tensor
    ) -> torch.Tensor:

        fused_features = F.normalize(fused_features, dim=-1)
        target_features = F.normalize(target_features, dim=-1)

        logits = (
            fused_features @ target_features.T
        ) / self.temperature

        labels = torch.arange(
            logits.size(0),
            device=logits.device
        )

        loss_i2t = F.cross_entropy(logits, labels)

        if self.symmetric:
            loss_t2i = F.cross_entropy(logits.T, labels)
            loss = (loss_i2t + loss_t2i) / 2
        else:
            loss = loss_i2t

        return loss


# ==========================================
# 4. Training Loop حرفه‌ای
# ==========================================
def train_fusion_model(
    train_loader: DataLoader,
    fusion_module: nn.Module,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    num_epochs: int,
    device: str = "cuda",
    grad_clip: float = 1.0,
    save_path: str = ""
):

    fusion_module.to(device)

    use_amp = (device == "cuda")
    scaler = GradScaler('cuda') if use_amp else None
    #scaler = GradScaler()

    best_loss = float("inf")

    for epoch in range(num_epochs):

        fusion_module.train()

        total_loss = 0.0

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{num_epochs}"
        )

        for ref_feat, text_feat, target_feat in progress_bar:

            ref_feat = ref_feat.to(device, non_blocking=True)
            text_feat = text_feat.to(device, non_blocking=True)
            target_feat = target_feat.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with autocast('cuda'):
                    fused_feat = fusion_module(ref_feat, text_feat)
                    loss = criterion(fused_feat, target_feat)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(fusion_module.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                fused_feat = fusion_module(ref_feat, text_feat)
                loss = criterion(fused_feat, target_feat)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(fusion_module.parameters(), grad_clip)
                optimizer.step()

            total_loss += loss.item()

            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}"
            })

        avg_loss = total_loss / len(train_loader)

        logger.info(
            f"Epoch {epoch + 1} | Avg Loss: {avg_loss:.4f}"
        )

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss

            torch.save({
                "epoch": epoch,
                "model_state_dict": fusion_module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss
            }, save_path)

            logger.info(f"Best model saved -> {save_path}")

    return fusion_module


# ==========================================
# 5. Utility Functions
# ==========================================
def count_parameters(model: nn.Module) -> int:
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


# ==========================================
# 6. Main
# ==========================================
if __name__ == "__main__":

    torch.backends.cudnn.benchmark = True

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    logger.info(f"Using device: {device}")

    # Model
    fusion_net = MLPFusion(
        input_dim=512,
        hidden_dim=1024,
        output_dim=512,
        dropout_rate=0.1
    )

    logger.info(
        f"Trainable params: {count_parameters(fusion_net):,}"
    )

    # Optimizer
    optimizer = optim.AdamW(
        fusion_net.parameters(),
        lr=1e-4,
        weight_decay=1e-2,
        betas=(0.9, 0.98)
    )

    # Loss
    criterion = InfoNCELoss(
        temperature=0.07,
        symmetric=True
    )

    # Dummy data
    dummy_data = [
        {"id": i}
        for i in range(5000)
    ]

    # Dataset
    train_dataset = CIRDataset(dummy_data)

    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=True
    )

    logger.info("Starting training...")

    trained_model = train_fusion_model(
        train_loader=train_loader,
        fusion_module=fusion_net,
        optimizer=optimizer,
        criterion=criterion,
        num_epochs=5,
        device=device
    )

    logger.info("Training completed successfully!")
