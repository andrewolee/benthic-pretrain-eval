#!/usr/bin/env bash
# Pulls the BenthicNet Barlow Twins ResNet-50 (400-epoch) SSL checkpoint from
# Hugging Face. This is the "treatment" encoder we compare against
# ImageNet-pretrained ResNet-50.
set -euo pipefail

DEST="data/checkpoints"
mkdir -p "$DEST"

python3 - "$DEST" <<'EOF'
import sys
from huggingface_hub import hf_hub_download

dest = sys.argv[1]
path = hf_hub_download(
    repo_id="MLIsaac/benthicnet_bt_rn50_e400",
    filename="bt-400e_rn50_epoch=399.ckpt",
    local_dir=dest,
)
print(f"Downloaded to {path}")
EOF
