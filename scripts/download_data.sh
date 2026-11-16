#!/usr/bin/env bash
# Sparse-clone just the labeled patch dataset from Seattle Aquarium's
# CCR_benthic_analyses repo, instead of pulling the whole (~1.5GB) repo.
set -euo pipefail

DEST="data/CCR_benthic_analyses"
SUBDIR="image_patch_aggregation/data/standard_patches"

if [ -d "$DEST/$SUBDIR" ]; then
  echo "Already present at $DEST/$SUBDIR"
  exit 0
fi

mkdir -p data
git clone --filter=blob:none --no-checkout --depth 1 \
  https://github.com/Seattle-Aquarium/CCR_benthic_analyses.git "$DEST"

cd "$DEST"
git sparse-checkout init --cone
git sparse-checkout set "$SUBDIR"
git checkout main

echo "Patches ready at $DEST/$SUBDIR"
