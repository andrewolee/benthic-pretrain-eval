# benthic-pretrain-eval

Does BenthicNet's self-supervised ResNet-50 (pretrained on 1M unlabeled
seafloor images) transfer better than a generic ImageNet-pretrained ResNet-50
to a real, low-label benthic classification task?

Downstream task: classifying ROV image patches from Seattle Aquarium's
Coastal Climate Resilience (CCR) program into 27 percent-cover classes
(algae, substrate, sessile/mobile species), using the labeled patches
already in `Seattle-Aquarium/CCR_benthic_analyses`.

Method: freeze the backbone, train only a linear head (linear probe) on top
of each pretrained encoder, and compare macro-F1 / balanced accuracy on a
held-out test split. A `random`-init backbone is included as a sanity-check
lower bound.

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

./scripts/download_data.sh      # ~4,100 labeled patches, sparse-cloned
./scripts/download_weights.sh   # BenthicNet Barlow Twins ResNet-50 (400e), ~700MB
```

## Run

```bash
python -m src.train --pretrain imagenet
python -m src.train --pretrain benthicnet
python -m src.train --pretrain random   # control

python -m src.compare runs/imagenet_seed42 runs/benthicnet_seed42 runs/random_seed42
```

## Notes

- Normalization stats differ per backbone: ImageNet mean/std for the
  `imagenet`/`random` runs, BenthicNet's own labeled-partition mean/std
  (from `DalhousieAI/benthicnet_probes`) for the `benthicnet` run.
- Classes with fewer than `--min-per-class` (default 20) patches are
  dropped so every class can be stratified into train/val/test; the
  `unknown` catch-all class is always excluded.
- Data source: https://github.com/Seattle-Aquarium/CCR_benthic_analyses
  (`image_patch_aggregation/data/standard_patches`)
- Weights source: https://huggingface.co/MLIsaac/benthicnet_bt_rn50_e400
