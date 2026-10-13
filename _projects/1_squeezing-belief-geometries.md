---
layout: page
title: Squeezing belief geometries
description: Superposition and interference when a transformer must represent several hidden processes in too narrow a residual stream
img: assets/img/projects/squeezing-belief-geometries-card.png
importance: 1
---

**Work in progress.** Small-scale, preliminary results from three seeds.

## Question

What does a transformer do when it must represent several processes in a residual stream too narrow to hold them all, and
does that make training on one process damage the others?

Small transformers trained on hidden-Markov data store the Bayesian belief state of the process as a linear geometry in
their residual stream ([Shai et al. 2024](https://arxiv.org/abs/2405.15943)). Each Mess3 process needs exactly 2
dimensions for its belief, so $$K$$ processes need $$2K$$ dimensions for a factored representation. That makes this an
exactly solvable test bed for _superposition_ ([Elhage et al. 2022](https://transformer-circuits.pub/2022/toy_model/index.html)):
the required dimension, the optimal loss and the true beliefs are all known, so every cost of squeezing is a distance
from a known answer.

{% include figure.liquid loading="eager" path="assets/img/projects/squeezing-belief-geometries-summary.png" class="img-fluid rounded z-depth-1" zoomable=true %}

<div class="caption">
  K = 3 Mess3 processes, residual width 3 to 32, mean over 3 seeds with the range across seeds shaded.
</div>

## Setup

Two data regimes map onto the sparsity condition at the centre of superposition theory:

- **Sparse:** each sequence comes from one of the $$K$$ processes, like features that are rarely active together.
- **Dense:** $$K$$ independent processes run side by side and each token is the tuple of their sub-tokens, like features
  that are always active together.

For each regime the residual width $$d$$ is swept from 3 to 32 with everything else fixed, so the residual stream is the
bottleneck. Each trained model is measured for:

- **Decodability:** held-out $$R^2$$ of a linear map from activations to each process's belief.
- **Overlap:** each process gets a 2-dimensional plane (how the activation moves as its belief moves). The overlap of two
  planes is $$\lVert Q_A^\top Q_B \rVert_F^2 / 2$$, the mean $$\cos^2$$ of their principal angles.
- **Frame bound:** $$K$$ planes of dimension 2 in $$\mathbb{R}^d$$ must overlap by at least
  $$\max\!\left(0, \frac{2K/d - 1}{K - 1}\right)$$ on average. Overlap near the bound is _forced_; overlap well above it is
  _chosen_.
- **Interference:** shift one process's parameters, fine-tune on that process alone, and measure how far predictions for
  the unchanged processes move (KL divergence), against a control that fine-tunes on the same process unchanged.

## Findings so far

1. **Sparse processes share space by choice.** All three beliefs stay decodable down to width 4, and their planes overlap
   far more than geometry requires (mean overlap 0.11–0.14 at widths 8–32, where zero was possible) at almost no cost in
   loss.
2. **Dense processes are kept orthogonal, and dropped when they don't fit.** The planes the model keeps are almost
   exactly orthogonal at every width. Short of width, it abandons whole processes rather than overlap them, even at
   $$d = 2K$$.
3. **Interference doesn't need overlapping representations.** In the dense regime, drift on the unchanged processes peaks
   at width 6 at 30 times the control, although their belief planes are orthogonal there, so interference must travel
   through shared weights rather than shared directions.
4. **In the sparse regime, interference falls about 100-fold as width grows**, reaching the control's level from width 16.
   Overlap and crowding fall together here, so a pairwise test at fixed width (in progress) is separating the two.

## Status and limitations

Tiny models (4 layers, widths 3–32) on synthetic data with three processes; whether these regimes appear in language
models is an open question, not a claim made here. Next step: the pairwise overlap-versus-interference test at fixed
width.

This project was developed with substantial help from Claude (Anthropic), which wrote most of the code. The research
question is mine, the experimental design was developed in discussion with Claude, I ran the experiments, and the
interpretation is my own.

**Code:** Python and PyTorch, CPU only. The repository will be linked here once it's public.
