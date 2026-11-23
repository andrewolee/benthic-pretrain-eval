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

# Label-budget sweep (frozen probe)
python -m src.sweep --pretrains imagenet,benthicnet,random --budgets 5,10,20,50,full

# Replicate the BenthicNet paper's exact probe recipe (100 epochs, warmup-cosine LR)
python -m src.train --pretrain imagenet --lr-schedule warmup-cosine --epochs 100
python -m src.train --pretrain benthicnet --lr-schedule warmup-cosine --epochs 100

# Fine-tuning (unfreeze the backbone) at low budgets
python -m src.sweep --pretrains imagenet,benthicnet --budgets 5,10,full \
  --fine-tune --lr-schedule warmup-cosine --epochs 100
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

### Methodology check: replicating BenthicNet's exact probe recipe

Read the BenthicNet paper (Lowe, Misiuk, Xu et al. 2024/2025, Sci Data)
directly rather than guessing at their eval protocol. Their actual
linear-probe recipe: 100 epochs, LR linearly warmed up from `3e-6` to
`3e-5` over 10 epochs, then cosine-annealed back down to `3e-6` -- quite
different from the flat `lr=1e-3`/30-epoch probe we'd been running.
Implemented as `--lr-schedule warmup-cosine` in `src/train.py`.

Their own results (Table 5/6 of the paper) were also informative: on a
large dataset (43K train images) ImageNet and BenthicNet tie in linear
probe; on a smaller, more specific one (German Bank 2010, 2.7K train
images) BenthicNet wins the linear probe decisively (43.2 vs 30.0 F1).
Their explanation: BenthicNet's SSL pretraining advantage shows up on
*smaller, more specific* classification tasks, not as a function of
"fine-tune vs. frozen" per se. Our CCR dataset (3,937 images, 25 classes)
is in that "smaller/specific" regime by their own framing, which raised
the question of why we weren't seeing anything close to their result.

Re-ran the full-budget comparison with their exact recipe (on a rented
RunPod RTX 3090, 100 epochs each):

| Recipe | ImageNet | BenthicNet | Gap |
|---|---|---|---|
| Our original (flat `lr=1e-3`, 30 epochs) | 0.827 | 0.692 | 0.135 |
| Paper's exact recipe (warmup-cosine, 100 epochs) | 0.557 | 0.513 | 0.044 |

**The gap shrank ~3x** once the LR schedule matched the paper's. This
supports the hypothesis that a meaningful chunk of ImageNet's earlier
lead was an artifact of our aggressive flat learning rate suiting
ImageNet's features better, not a true reflection of pretraining quality
-- ImageNet still edges ahead, but it's close now, not a blowout.

Caveat: both absolute scores dropped substantially under the paper's
recipe (ImageNet: 0.827 -> 0.557). Their LR schedule (peak `3e-5`) was
likely tuned for their much larger datasets (43K / 2.7K images); on our
smaller dataset (2,164 training images, ~34 batches/epoch) that peak LR
may be too conservative to fully converge in 100 epochs. So this recipe
is more apples-to-apples for *comparing* the two backbones, but may not
be *optimal* for either at our data scale.

### Follow-up: is the narrower gap about LR schedule shape, or just LR magnitude?

Swept the warmup-cosine peak LR at full budget to check (rented RunPod RTX
3090, 100 epochs each):

| Peak LR | ImageNet | BenthicNet | Gap |
|---|---|---|---|
| `3e-5` (paper's exact value) | 0.557 | 0.513 | 0.044 |
| `1e-4` | 0.743 | 0.621 | 0.122 |
| `3e-4` | 0.796 | 0.679 | 0.117 |
| flat `1e-3` (original) | 0.827 | 0.692 | 0.135 |

The gap re-widens almost monotonically as peak LR increases, landing back
near the original 0.135 by just `1e-4` -- nowhere close to staying at the
paper's exact value. So the earlier "3x narrower gap" wasn't really about
the warmup-cosine *shape* being a fairer comparison -- it was mostly that
the paper's specific peak LR (`3e-5`) is conservative enough to leave both
heads undertrained on our much smaller dataset, which shrinks the measured
gap as a side effect rather than revealing a more accurate one.

### Fine-tuning results

Ran `--fine-tune` (unfreeze the backbone) at low budgets + full, using the
paper's warmup-cosine recipe (fine-tune stage: flat LR = peak_lr * 0.1,
per the paper's "one tenth the learning rate used for the linear probe"),
on the same rented RTX 3090:

| Budget/class | ImageNet (fine-tuned) | BenthicNet (fine-tuned) | BenthicNet advantage |
|---|---|---|---|
| 5    | 0.082 | 0.444 | ~5.4x |
| 10   | 0.205 | 0.604 | ~2.9x |
| full | 0.738 | 0.819 | ~1.11x |

**BenthicNet wins at every budget under fine-tuning, with the advantage
shrinking cleanly as the budget grows** -- the opposite of every
frozen-probe result above, and a close match to the paper's own framing
(advantage concentrated in smaller/low-label tasks).

Caveat on the low-budget ImageNet numbers: this is a *single-stage*
fine-tune (backbone unfrozen from the start, head randomly initialized),
not the paper's two-stage recipe (probe first, then unfreeze). Jointly
training a from-scratch head and the full backbone on just 125-250 images
can be unstable -- plausibly why ImageNet's fine-tuned score at budget=5
(0.082) is even lower than its own frozen-probe score at the same budget
(0.542). BenthicNet tolerates this better, which may itself reflect its
SSL pretraining having seen a more diverse range of benthic imagery. Not
independently verified; a two-stage fine-tune would be a cleaner test of
this specific caveat.

**But does this mean BenthicNet is the better choice overall? No --
compare fine-tuned BenthicNet to *probed* ImageNet, not fine-tuned
ImageNet:**

| Budget/class | ImageNet (frozen probe, best) | BenthicNet (fine-tuned, best) | Winner |
|---|---|---|---|
| 5    | 0.542 | 0.444 | ImageNet, by 0.098 |
| 10   | 0.630 | 0.604 | ImageNet, by 0.026 |
| full | 0.827 | 0.819 | ImageNet, by 0.008 (~tied) |

**ImageNet's cheap frozen probe still wins outright at every budget**,
even against BenthicNet's far more expensive fine-tuned result. The
honest summary of the fine-tuning experiment isn't "BenthicNet wins" --
it's "*if* you're going to fine-tune, BenthicNet is the better backbone
to fine-tune, but fine-tuning anything doesn't beat just doing a cheap
ImageNet probe in the first place."

### Would more data (e.g. the full Kelp Quest/Zooniverse dataset) help BenthicNet?

Likely not -- if anything, probably the opposite. BenthicNet's relative
advantage shrinks monotonically as the budget grows in every comparison
above (5.4x -> 2.9x -> 1.11x under fine-tuning). The paper's own two
benchmark datasets show the same pattern: roughly tied on their large
(43K-image) dataset, BenthicNet winning decisively on their small
(2.7K-image) one. Seattle Aquarium's Kelp Quest/Zooniverse pipeline has
accumulated ~50K+ verified classifications -- more than 10x our current
training set. Scaling up to that would likely push this comparison
further toward the "large dataset" regime, where the paper's own results
(and ours) suggest ImageNet's position holds or improves, not BenthicNet's.
(Caveat: this assumes the larger set is "more of the same" rather than a
meaningfully different/harder task.)

### Overall conclusion

Across every configuration tested -- frozen probe, fine-tuned, five label
budgets, four LR schedules -- **ImageNet-pretrained features are either
the best choice outright, or close enough that BenthicNet's
domain-specific SSL pretraining doesn't clearly earn its extra cost on
this dataset.** BenthicNet's pretraining does show a real, measurable
benefit in one specific, narrower sense: *given that you're fine-tuning*,
it's a better starting point than ImageNet, especially at low label
counts. But that's a different and much narrower claim than "BenthicNet
pretraining helps this task" -- and for Seattle Aquarium's actual
pipeline, which uses a cheap ImageNet-pretrained backbone via
CoralNet-Toolbox, this project found no evidence that switching to
BenthicNet would improve results enough to justify the switch.
