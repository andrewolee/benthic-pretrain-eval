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

## Results (linear probe, label-budget sweep)

Context: Seattle Aquarium's existing production classifier for this same
task (via CoralNet-Toolbox -> Ultralytics YOLO) already starts from
COCO/ImageNet-pretrained weights, not BenthicNet. So this isn't just an
abstract pretraining comparison -- it's a direct test of whether their
current choice (ImageNet) is leaving performance on the table relative to
domain-specific SSL pretraining.

Setup: frozen backbone + trainable linear head, 25 classes (CCR
percent-cover categories, `unknown` and one <20-sample class excluded),
stratified 55/15/30 train/val/test split, same split across all runs and
budgets. "Budget" = max training examples per class (5/10/20/50/"full" =
all available, up to ~400/class for the most common categories). Metric:
macro-F1 on the held-out test set, picked at the best-val-F1 epoch.

| Budget/class | ImageNet | BenthicNet (SSL) | Random init |
|---|---|---|---|
| 5    | 0.542 | 0.409 | 0.020 |
| 10   | 0.630 | 0.495 | 0.030 |
| 20   | 0.717 | 0.585 | 0.100 |
| 50   | 0.805 | 0.633 | 0.154 |
| full | 0.827 | 0.692 | 0.204 |

See `runs/sweep/sweep_results.csv` and `runs/sweep/sweep_plot.png` for the
full numbers/plot.

**Headline finding: ImageNet-pretrained beats BenthicNet-pretrained at
every label budget tested, including the sparsest (5/class) -- no
crossover anywhere in the sweep.** This is the opposite of what a naive
reading of BenthicNet's paper would predict for a low-label regime, and
the gap is consistent (~0.13-0.17 macro-F1 in ImageNet's favor), not
noise. Both pretrained backbones comfortably beat the random-init
control, confirming the eval pipeline itself is sound.

### Why might this diverge from BenthicNet's paper?

Ranked by how likely each explanation seems:

1. **Frozen linear probe vs. fine-tuning.** This experiment only trains a
   linear head on frozen features. SSL-pretrained encoders (Barlow Twins
   included) are well known in the broader SSL literature to probe worse
   than supervised ImageNet features even when the SSL representation is
   otherwise strong -- the features just aren't as linearly organized by
   class. Fine-tuning (unfreezing the backbone) is the standard way SSL
   methods close or reverse that gap, and is a strong candidate for where
   BenthicNet's paper-reported low-label advantage actually comes from.
   Not yet tested here.
2. **Domain mismatch within "benthic."** BenthicNet-1M is a *global*
   compilation, likely skewed toward whatever regions/survey programs
   contributed most images (plausibly warmer-water/coral-heavy). Our data
   is cold-water Pacific NW kelp forest -- different turbidity, color
   cast, and texture. "Pretrained on seafloor images" doesn't guarantee a
   close visual match to *this* seafloor's images.
3. **Checkpoint choice.** We used the raw 400-epoch Barlow Twins SSL
   checkpoint (`MLIsaac/benthicnet_bt_rn50_e400`). The `benthicnet_probes`
   repo's config filenames (`hp`, `hl`, `hft`, `ohft`, ...) suggest several
   training recipe variants; the paper's best-reported numbers may start
   from a checkpoint already adapted further (e.g. on BenthicNet's own
   labeled subset) rather than pure SSL-only weights.
4. **Scale of "low-label" may differ.** The paper's low-label claim may be
   benchmarked against a larger labeled pool where "scarce" means a
   different absolute scale than our 5-50/class here.

### Practical takeaway (so far)

For Seattle Aquarium's actual use case, the status quo choice
(ImageNet-pretrained backbones via CoralNet-Toolbox) looks like a
reasonable one -- this project hasn't found evidence that swapping in
BenthicNet's SSL-pretrained encoder would help, at least under a frozen
linear probe. That's a useful, legitimate finding on its own, independent
of whether a future fine-tuning experiment changes the picture.

### Natural next step: fine-tuning

The strongest remaining test of BenthicNet's claimed advantage is
fine-tuning (unfreezing the backbone) specifically at **low** label
budgets (5-10/class) -- that's the regime where (a) ImageNet can't lean on
having lots of data to adapt with, and (b) fine-tuning removes ImageNet's
"already linearly organized" edge from the frozen-probe comparison. If
BenthicNet still loses there, that's a fairly strong signal the
domain-specific pretraining isn't paying off for this dataset even in its
best-case scenario. Not yet implemented/run as of this writing.
