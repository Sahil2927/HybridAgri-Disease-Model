# HybridAgri-Disease-Model
**SimCLR + MixMatch for Tomato Leaf Disease Classification**

An empirical study of self-supervised pretraining (SimCLR) combined with semi-supervised fine-tuning (MixMatch) on the PlantVillage tomato subset, plus three proposed remediation methods (GEPS — Geometry-Enhanced Pseudo-labeling for Semi-supervised learning).

This repository contains the complete reproducible pipeline used for the project, from data setup through every method tested.

---

## TL;DR — the main finding

The composability of SimCLR + MixMatch is **regime-dependent**:

| Method | 1% labels | 10% labels |
|---|---|---|
| Supervised (ImageNet) | 76.96% | 95.30% |
| SimCLR + linear probe | 81.15% | 88.17% |
| SimCLR + full fine-tune | 84.47% | 97.68% |
| **MixMatch (ImageNet)** | **86.47%** | 97.75% |
| **SimCLR + MixMatch (full FT)** | 82.16% | **98.00%** |
| SimCLR + MixMatch (frozen except L4) | 81.40% | 95.68% |
| GEPS v1 (agreement-weighted) | 81.28% | 97.93% |
| GEPS v4 (annealed prior, Path A) | 82.22% | 97–98% |
| GEPS v5 (confidence-filtered, Path B) | 81.97% | 97–98% |

- At **10% labels**, SimCLR + MixMatch achieves the best result of any method (98.00%).
- At **1% labels**, the same combination underperforms MixMatch alone by **4.31 points** (82.16% vs 86.47%).
- Three principled GEPS variants designed to recover this 1% gap all fail.

A diagnostic analysis (per-class geometric pseudo-label accuracy, agreement-trajectory tracking) explains why: the classifier rapidly internalizes SimCLR's geometric structure within ~10 epochs, making explicit geometric supervision redundant; and class-imbalanced anchor scarcity produces systematically wrong geometric pseudo-labels for visually similar minority classes.

---

## Repository contents

```
.
├── Hybrid_research.ipynb        # complete end-to-end notebook (109 cells)
└── README.md                    # this file
```

The notebook covers:

| Stage | What it does |
|---|---|
| 0 | Environment check, GPU verification |
| 1 | Dataset download (PlantVillage via Kaggle), tomato subset extraction, stratified split (1% / 80% / 10%) |
| 2 | t-SNE sanity check on ImageNet features (go/no-go gate for the project) |
| 3 | SimCLR pretraining — two versions: from-scratch and ImageNet-init |
| 4 | Feature space evaluation (k-NN accuracy, silhouette, t-SNE on SimCLR embeddings) |
| 5 | Baselines B1–B5 at 1% labels |
| 6 | GEPS v1 — agreement-weighted blend |
| 7 | GEPS v2–v3 — frozen geometry + distillation (both broken by an environment bug — kept for diagnostic value) |
| 8 | Path A vs Path B design decision; GEPS v4 (annealed) and v5 (confidence filter) |
| 9 | Bug discovery — `requires_grad` propagation through `copy.deepcopy` |
| 10 | Fixed reruns of all GEPS variants |
| 11 | Per-class diagnostic of geometric pseudo-label accuracy |
| 12 | Full 10% labels comparison (all methods rerun) |

---

## Setup

This was developed and tested on Google Colab with a T4 GPU. The notebook is self-contained and downloads the dataset itself.

### Required:
- Google Colab (or any environment with a CUDA GPU)
- A [Kaggle account](https://www.kaggle.com/) with an API key (`kaggle.json`)
- Approximately **2 GB of Google Drive space** for checkpoints and figures

### Run order (in Colab):
1. Open `Hybrid_research.ipynb` in Colab
2. Enable GPU runtime (`Runtime → Change runtime type → T4 GPU`)
3. Run cells sequentially; the notebook will prompt you to upload `kaggle.json` for dataset download
4. Total wall-clock time: approximately **3 hours** for the complete pipeline at both label fractions

### Dependencies
Install line in the notebook handles them, but for reference:
```
torch, torchvision, scikit-learn, matplotlib, tqdm, Pillow
```

---

## What's in the notebook

### Methods implemented

**Baselines:**
- **B1** — Supervised ResNet-18 from ImageNet initialization, trained on labeled set only
- **B2** — SimCLR encoder frozen + single linear classifier on labeled set
- **B3** — SimCLR encoder + classifier, all layers trainable (full fine-tune)
- **B4** — Standard MixMatch (with MixUp) starting from ImageNet weights, no SimCLR
- **B5-full** — SimCLR encoder + classifier with MixMatch loss, all layers trainable
- **B5-frozen** — SimCLR encoder + classifier with MixMatch, only `layer4` + classifier trainable (the configuration commonly seen in agricultural ML pipelines)

**Proposed methods (GEPS family):**
- **GEPS v1** — Classifier and geometric pseudo-labels are blended; unsupervised loss is weighted by their agreement (Bhattacharyya coefficient raised to power γ)
- **GEPS v4 (Path A)** — Linear annealing schedule: pseudo-label starts as 70% geometric / 30% classifier and ends as 10% / 90% by the final epoch
- **GEPS v5 (Path B)** — FixMatch-style confidence filtering: use geometric pseudo-label as a hard target only when its max probability exceeds 0.7 AND it agrees with the classifier's prediction; otherwise fall back to the classifier; mask the loss when there's confident disagreement

### Diagnostics

**Per-class geometric pseudo-label accuracy** (sample run, 1% labels, seed 42):

| Class | Anchors | Geo accuracy |
|---|---|---|
| YellowLeaf Curl Virus | 32 | 97.5% |
| Late_blight | 19 | 88.4% |
| Bacterial_spot | 21 | 87.7% |
| mosaic_virus | 10 | 81.1% |
| Septoria_leaf_spot | 17 | 78.5% |
| healthy | 15 | 76.1% |
| Spider_mites | 16 | 62.9% |
| Target_Spot | 14 | 61.4% |
| Leaf_Mold | 9 | 45.3% |
| Early_blight | 10 | 37.0% |

This per-class breakdown is the smoking gun for why geometric pseudo-labels don't help: q_geo is highly confident on classes it gets wrong, injecting wrong labels into ~25% of unlabeled training samples.

**Agreement trajectory:** During GEPS v1 training, the Bhattacharyya agreement α between classifier and geometric pseudo-labels rises from 0.37 (epoch 1) to 0.95 (epoch 15) and saturates. The classifier rapidly internalizes SimCLR's k-NN structure on its own — making explicit geometric supervision largely redundant.

---

## Dataset

- **Source:** [emmarex/plantdisease](https://www.kaggle.com/datasets/emmarex/plantdisease) on Kaggle (PlantVillage)
- **Subset used:** 10 tomato disease classes only
- **Total images:** 16,011
- **Split (stratified by class, seed=42, with a per-class floor of 10):**
  - 1% setting: 164 labeled / 14,250 unlabeled / 1,597 validation
  - 10% setting: 1,597 labeled / 12,817 unlabeled / 1,597 validation

### Class distribution

| Class index | Class name | Total images |
|---|---|---|
| 0 | Tomato_Bacterial_spot | 2,127 |
| 1 | Tomato_Early_blight | 1,000 |
| 2 | Tomato_Late_blight | 1,909 |
| 3 | Tomato_Leaf_Mold | 952 |
| 4 | Tomato_Septoria_leaf_spot | 1,771 |
| 5 | Tomato_Spider_mites_Two_spotted_spider_mite | 1,676 |
| 6 | Tomato__Target_Spot | 1,404 |
| 7 | Tomato__Tomato_YellowLeaf__Curl_Virus | 3,208 |
| 8 | Tomato__Tomato_mosaic_virus | 373 |
| 9 | Tomato_healthy | 1,591 |

Note: mosaic_virus has only 373 total images and would receive just 3 labeled samples under strict 1%. A per-class floor of 10 is enforced to ensure every class has enough anchors for both training and geometric pseudo-labeling.

---

## Hyperparameters

Held constant across methods for fair comparison:

| Setting | Value |
|---|---|
| Backbone | ResNet-18 |
| Image size | 224×224 |
| Batch size | 128 |
| Seed | 42 (multi-seed verification done for key results) |
| SimCLR pretraining epochs | 10 |
| SimCLR temperature | 0.5 |
| MixMatch augmentations K | 2 |
| MixMatch sharpen T | 0.5 |
| MixUp α | 0.75 |
| MixMatch λ_u | 1.0 |
| Fine-tune epochs | 30 |
| Fine-tune LR (full FT) | 1e-4 |
| Fine-tune LR (frozen-L4) | 1e-3 |

GEPS-specific:

| Setting | Value |
|---|---|
| k (nearest anchors) | 10 |
| τ (anchor softmax temperature) | 0.1 |
| γ (agreement exponent, v1) | 2.0 |
| λ_g schedule (v4, Path A) | linear, 0.7 → 0.1 |
| τ_conf (v5, Path B) | 0.7 |

---

## Project journey (briefly)

The project went through several non-trivial transitions worth noting for anyone replicating:

1. **SimCLR from-scratch was weaker than ImageNet alone.** First attempt used random initialization for SimCLR and produced 74% k-NN accuracy on the validation set — below ImageNet's 80%. Initializing SimCLR from ImageNet weights and continuing pretraining gave 84%. The notebook keeps both versions; only the ImageNet-init one is used downstream.

2. **10% labels was too easy.** Initial baseline showed supervised ImageNet hitting 95.30%, leaving no headroom. Pivoted to 1% labels where methods can be cleanly separated.

3. **The naive SimCLR + MixMatch hybrid underperformed.** At 1% labels, the combination scored 82.16% — below MixMatch alone (86.47%). This was reproducible across seeds. It is the central puzzle of the project.

4. **Four GEPS variants did not recover the gap.** Each variant addresses a different conceptual issue (agreement weighting, frozen geometry, annealed priors, confidence filtering). None beats 82.5% at 1% labels.

5. **An encoder-freeze bug caused four false-negative experiments.** A Colab disconnect required reloading the SimCLR model; the recovery code set `requires_grad = False` on all parameters for use in computing geometric pseudo-labels, but `copy.deepcopy` propagated that flag into every subsequent trainable GEPS model. Result: only the classifier head (5K of 11M parameters) was actually training. Fixed by maintaining separate frozen and trainable SimCLR references.

6. **Re-examining 10% labels revealed the two-regime story.** Running all methods at 10% labels showed that SimCLR + MixMatch full-FT achieves the best result of any method tested (98.00%). The hybrid does work — just not at the extreme 1% label fraction where the labeled signal is insufficient to stabilize the encoder against MixMatch's noisy pseudo-label gradients.

---

## Findings

The empirical conclusions:

- **ImageNet-initialized SimCLR continued pretraining beats both ImageNet alone and from-scratch SimCLR.** The 4-point k-NN gap (80% → 84%) confirms domain-specific SSL adds value over generic pretraining.

- **The naive SimCLR + MixMatch hybrid achieves the best 10%-label result of any method.** SSL + semi-SSL composes effectively when labeled signal is sufficient.

- **The same hybrid fails at 1% labels by 4.3 points.** Composability inverts at extreme label scarcity.

- **GEPS in any of its three principled formulations does not recover the 1% gap.** Geometric pseudo-labels are not the right intervention for this failure mode.

- **The classifier internalizes SimCLR geometry within ~10 epochs at 1% labels (within 2 epochs at 10% labels).** Explicit geometric supervision becomes redundant after this convergence.

- **Per-class anchor scarcity makes geometric pseudo-labels confidently wrong on minority/visually-similar classes.** Especially Early_blight (37%) and Leaf_Mold (45%) accuracy.

---

## Citation and use

This work was conducted as a research project at BIT Mesra. The notebook is provided as-is for reproducibility and for use as a baseline by future work.

If you find the empirical observations useful (particularly the regime-dependent composability result and the agreement-trajectory diagnostic), please cite the upcoming paper (link will be added once published).

---

## Acknowledgments

- PlantVillage dataset: Hughes & Salathé (2015); Mohanty, Hughes & Salathé (2016)
- SimCLR: Chen et al., *A Simple Framework for Contrastive Learning of Visual Representations*, ICML 2020
- MixMatch: Berthelot et al., *MixMatch: A Holistic Approach to Semi-Supervised Learning*, NeurIPS 2019
- SimCLRv2 (paradigm reference): Chen et al., *Big Self-Supervised Models are Strong Semi-Supervised Learners*, NeurIPS 2020
- OpenCoS (named SimCLR + MixMatch baseline): Saito et al., ICLR 2021 workshops

---

## Contact

Issues and discussion via GitHub Issues. For the full project report and paper draft, see the project repository's `report/` directory.
