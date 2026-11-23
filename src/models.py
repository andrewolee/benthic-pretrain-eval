"""ResNet-50 backbone construction for the two pretraining regimes under test."""
import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# From DalhousieAI/benthicnet_probes `benthicnet_norms/labelled_mean_std.txt`
# -- stats over BenthicNet's own labelled image partition, used here instead
# of ImageNet norms when the backbone was pretrained on BenthicNet.
BENTHICNET_MEAN = (0.36269833, 0.41986692, 0.34385223)
BENTHICNET_STD = (0.20742336, 0.20954326, 0.18345217)


def _load_benthicnet_backbone(ckpt_path):
    model = resnet50(weights=None)
    model.fc = nn.Identity()

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state = checkpoint["state_dict"]

    # Mirrors DalhousieAI/benthicnet_probes `load_model_state`: SSL checkpoint
    # keys are prefixed "backbone." (sometimes "encoder."); torchvision's
    # resnet50 state dict has no such prefix.
    model_keys = set(model.state_dict().keys())
    loading_state = {}
    for key, value in state.items():
        for prefix in ("backbone.", "encoder."):
            if key.startswith(prefix):
                key = key[len(prefix):]
                break
        if key in model_keys:
            loading_state[key] = value

    missing = model_keys - set(loading_state.keys())
    print(f"Loaded {len(loading_state)}/{len(model_keys)} backbone tensors from {ckpt_path}")
    if missing:
        print(f"  unmatched (left at random init): {sorted(missing)}")

    model.load_state_dict(loading_state, strict=False)
    return model


def build_backbone(pretrain, benthicnet_ckpt=None):
    """Returns (backbone, feature_dim, norm_mean, norm_std).

    pretrain: "imagenet" | "benthicnet" | "random"
    """
    if pretrain == "imagenet":
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        model.fc = nn.Identity()
        return model, 2048, IMAGENET_MEAN, IMAGENET_STD

    if pretrain == "random":
        model = resnet50(weights=None)
        model.fc = nn.Identity()
        return model, 2048, IMAGENET_MEAN, IMAGENET_STD

    if pretrain == "benthicnet":
        if not benthicnet_ckpt:
            raise ValueError("benthicnet_ckpt path is required for pretrain='benthicnet'")
        model = _load_benthicnet_backbone(benthicnet_ckpt)
        return model, 2048, BENTHICNET_MEAN, BENTHICNET_STD

    raise ValueError(f"Unknown pretrain option: {pretrain!r}")


class LinearProbe(nn.Module):
    """Backbone (frozen, or unfrozen for fine-tuning) + trainable linear head."""

    def __init__(self, backbone, feature_dim, num_classes, freeze_backbone=True):
        super().__init__()
        self.backbone = backbone
        self.freeze_backbone = freeze_backbone
        for param in self.backbone.parameters():
            param.requires_grad_(not freeze_backbone)
        self.head = nn.Linear(feature_dim, num_classes)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()  # keep frozen backbone's batchnorm/dropout in eval regardless
        return self

    def forward(self, x):
        if self.freeze_backbone:
            with torch.no_grad():
                features = self.backbone(x)
        else:
            features = self.backbone(x)
        return self.head(features)
