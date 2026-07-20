import logging
from pathlib import Path
from typing import Optional, Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# =========================================================
# تنظیمات اولیه
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

torch.backends.cudnn.benchmark = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================================================
# مدل Fusion
# =========================================================

class MLPFusion(nn.Module):
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 1024,
        output_dim: int = 512,
        num_layers: int = 3,
        dropout_rate: float = 0.1,
        use_residual: bool = True
    ):
        super().__init__()

        self.use_residual = use_residual

        self.input_norm = nn.LayerNorm(input_dim * 2)

        layers = []
        in_dim = input_dim * 2

        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_rate)
            ])
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

        if use_residual:
            if input_dim != output_dim:
                self.residual_proj = nn.Linear(input_dim, output_dim)
            else:
                self.residual_proj = nn.Identity()

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor
    ) -> torch.Tensor:

        x = torch.cat([image_features, text_features], dim=-1)

        x = self.input_norm(x)

        fused = self.mlp(x)

        if self.use_residual:
            residual = self.residual_proj(image_features)
            fused = fused + residual

        return fused


# =========================================================
# Dataset
# =========================================================

class CIRDataset(Dataset):
    def __init__(
        self,
        num_samples: int = 5000,
        feature_dim: int = 512,
        feature_dir: Optional[Path] = None
    ):
        self.num_samples = num_samples
        self.feature_dim = feature_dim
        self.feature_dir = feature_dir

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):

        # داده تصادفی
        # در حالت واقعی:
        # image_feat = torch.load(...)
        # text_feat = ...
        # target_feat = ...

        image_feat = torch.randn(self.feature_dim)
        text_feat = torch.randn(self.feature_dim)

        # target مشابه image + text
        target_feat = image_feat + 0.1 * text_feat

        return (
            image_feat.float(),
            text_feat.float(),
            target_feat.float()
        )


# =========================================================
# InfoNCE Loss
# =========================================================

class InfoNCELoss(nn.Module):
    def __init__(
        self,
        temperature: float = 0.07,
        symmetric: bool = True
    ):
        super().__init__()

        self.temperature = temperature
        self.symmetric = symmetric
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(
        self,
        fused_features: torch.Tensor,
        target_features: torch.Tensor
    ):

        fused_features = nn.functional.normalize(
            fused_features,
            dim=-1
        )

        target_features = nn.functional.normalize(
            target_features,
            dim=-1
        )

        logits = (
            fused_features @ target_features.T
        ) / self.temperature

        labels = torch.arange(
            logits.size(0),
            device=logits.device
        )

        loss_i2t = self.cross_entropy(logits, labels)

        if self.symmetric:
            loss_t2i = self.cross_entropy(logits.T, labels)
            loss = (loss_i2t + loss_t2i) / 2
        else:
            loss = loss_i2t

        return loss


# =========================================================
# شمارش پارامترها
# =========================================================

def count_parameters(model):
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


# =========================================================
# Train
# =========================================================

def train_fusion_model(
    model,
    train_loader,
    optimizer,
    criterion,
    device,
    num_epochs=10,
    grad_clip=1.0,
    save_path="best_model.pth"
):

    scaler = GradScaler()

    best_loss = float("inf")

    model.to(device)

    for epoch in range(num_epochs):

        model.train()

        total_loss = 0

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{num_epochs}"
        )

        for image_feat, text_feat, target_feat in progress_bar:

            image_feat = image_feat.to(
                device,
                non_blocking=True
            )

            text_feat = text_feat.to(
                device,
                non_blocking=True
            )

            target_feat = target_feat.to(
                device,
                non_blocking=True
            )

            optimizer.zero_grad(set_to_none=True)

            with autocast():

                fused = model(
                    image_feat,
                    text_feat
                )

                loss = criterion(
                    fused,
                    target_feat
                )

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                grad_clip
            )

            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        avg_loss = total_loss / len(train_loader)

        logger.info(
            f"Epoch {epoch+1} | Avg Loss: {avg_loss:.4f}"
        )

        if avg_loss < best_loss:

            best_loss = avg_loss

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss
            }, save_path)

            logger.info(
                f"Best model saved -> {save_path}"
            )

    logger.info("Training finished")


# =========================================================
# Evaluation
# =========================================================

def evaluate_model(
    model,
    dataloader,
    device,
    top_k=[1, 5, 10]
):

    model.eval()

    all_fused = []
    all_targets = []

    with torch.no_grad():

        for image_feat, text_feat, target_feat in dataloader:

            image_feat = image_feat.to(device)
            text_feat = text_feat.to(device)

            fused = model(
                image_feat,
                text_feat
            )

            all_fused.append(
                fused.cpu().numpy()
            )

            all_targets.append(
                target_feat.numpy()
            )

    all_fused = np.vstack(all_fused)
    all_targets = np.vstack(all_targets)

    similarities = cosine_similarity(
        all_fused,
        all_targets
    )

    results = {}

    for k in top_k:

        top_indices = np.argsort(
            -similarities,
            axis=1
        )[:, :k]

        gt = np.arange(len(all_fused))[:, None]

        correct = np.any(
            top_indices == gt,
            axis=1
        )

        recall = correct.mean()

        results[f"Recall@{k}"] = recall

        logger.info(
            f"Recall@{k}: {recall:.4f}"
        )

    return results


# =========================================================
# Inference
# =========================================================

def inference(
    model,
    ref_feature,
    text_feature,
    device
):

    model.eval()

    with torch.no_grad():

        ref_feature = torch.tensor(
            ref_feature,
            dtype=torch.float32
        ).unsqueeze(0).to(device)

        text_feature = torch.tensor(
            text_feature,
            dtype=torch.float32
        ).unsqueeze(0).to(device)

        fused = model(
            ref_feature,
            text_feature
        )

    return fused.cpu().numpy()


# =========================================================
# Search
# =========================================================

def search_target_image(
    fused_feature,
    gallery_features,
    top_k=5
):

    similarities = cosine_similarity(
        fused_feature,
        gallery_features
    )[0]

    top_indices = np.argsort(
        -similarities
    )[:top_k]

    top_scores = similarities[top_indices]

    return top_indices, top_scores


# =========================================================
# Visualization
# =========================================================

def visualize_embeddings(
    model,
    dataloader,
    device,
    num_samples=300
):

    model.eval()

    fused_features = []
    target_features = []

    with torch.no_grad():

        for idx, (
            image_feat,
            text_feat,
            target_feat
        ) in enumerate(dataloader):

            if idx * dataloader.batch_size >= num_samples:
                break

            image_feat = image_feat.to(device)
            text_feat = text_feat.to(device)

            fused = model(
                image_feat,
                text_feat
            )

            fused_features.append(
                fused.cpu().numpy()
            )

            target_features.append(
                target_feat.numpy()
            )

    fused_features = np.vstack(fused_features)
    target_features = np.vstack(target_features)

    combined = np.concatenate([
        fused_features,
        target_features
    ])

    tsne = TSNE(
        n_components=2,
        random_state=42
    )

    reduced = tsne.fit_transform(combined)

    fused_2d = reduced[:len(fused_features)]
    target_2d = reduced[len(fused_features):]

    plt.figure(figsize=(10, 8))

    plt.scatter(
        fused_2d[:, 0],
        fused_2d[:, 1],
        label="Fused",
        alpha=0.7
    )

    plt.scatter(
        target_2d[:, 0],
        target_2d[:, 1],
        label="Target",
        alpha=0.7
    )

    plt.legend()
    plt.title("t-SNE Embeddings")
    plt.show()


# =========================================================
# TorchScript Export
# =========================================================

def export_torchscript(
    model,
    save_path="fusion_model.pt"
):

    model.eval()

    example_image = torch.randn(1, 512).to(DEVICE)
    example_text = torch.randn(1, 512).to(DEVICE)

    traced_model = torch.jit.trace(
        model,
        (example_image, example_text)
    )

    traced_model.save(save_path)

    logger.info(
        f"TorchScript saved -> {save_path}"
    )


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":

    logger.info(f"Using device: {DEVICE}")

    # Dataset
    train_dataset = CIRDataset(
        num_samples=5000
    )

    test_dataset = CIRDataset(
        num_samples=1000
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Model
    model = MLPFusion(
        input_dim=512,
        hidden_dim=1024,
        output_dim=512,
        dropout_rate=0.1
    )

    logger.info(
        f"Trainable params: {count_parameters(model):,}"
    )

    # Loss
    criterion = InfoNCELoss(
        temperature=0.07,
        symmetric=True
    )

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-2
    )

    # Training
    train_fusion_model(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=DEVICE,
        num_epochs=5,
        save_path="best_model.pth"
    )

    # Load best model
    checkpoint = torch.load(
        "best_model.pth",
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    logger.info("Best model loaded")

    # Evaluation
    evaluate_model(
        model,
        test_loader,
        DEVICE
    )

    # Inference
    sample_image = np.random.randn(512)
    sample_text = np.random.randn(512)

    fused_feature = inference(
        model,
        sample_image,
        sample_text,
        DEVICE
    )

    logger.info(
        f"Inference output shape: {fused_feature.shape}"
    )

    # Search
    gallery_features = np.random.randn(
        10000,
        512
    )

    top_indices, top_scores = search_target_image(
        fused_feature,
        gallery_features,
        top_k=5
    )

    logger.info("Top Retrieval Results:")

    for idx, score in zip(
        top_indices,
        top_scores
    ):
        logger.info(
            f"Image {idx} | Similarity: {score:.4f}"
        )

    # Visualization
    visualize_embeddings(
        model,
        test_loader,
        DEVICE
    )

    # Export
    export_torchscript(
        model,
        "fusion_model.pt"
    )

    logger.info("Pipeline completed successfully")
