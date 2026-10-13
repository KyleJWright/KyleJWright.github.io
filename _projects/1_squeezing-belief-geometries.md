---
layout: page
title: Squeezing belief geometries
description: Reproducing the belief-state simplex of Shai et al. in a small transformer, then squeezing it into a residual stream too narrow to hold it
img: assets/img/projects/squeezing-belief-geometries-card.png
importance: 1
github: https://github.com/KyleJWright/squeezing-belief-geometries
---

Reproducing the belief-state simplex of Shai et al. (2024) in a small transformer, then squeezing it: what happens to the
simplex when the residual stream is too narrow to hold it, and when two processes must share a stream with room for one
but not both? The learned geometry is compared with the Grassmannian packing problem.

**Code and full write-up:** [github.com/KyleJWright/squeezing-belief-geometries](https://github.com/KyleJWright/squeezing-belief-geometries).
The repository README has every table and the full theory section.

## Background

While at the Iliad Intensive I was introduced to the paper
[_Transformers Represent Belief State Geometry in their Residual Stream_](https://arxiv.org/abs/2405.15943) (Shai et
al. 2024). The image of a hidden Markov model encoded in the network is striking. Seeing the belief-state simplex in the
residual stream gives strong support to the idea that training learns models of beliefs about the hidden state, updated
optimally, as predicted in theory (at least in some simple cases).

I wanted to get my hands dirty: to see whether I could reproduce these results, with the assistance of Claude, and then
extend the code to ask some simple follow-up questions.

During the Intensive I was also introduced to a version of the _superposition hypothesis_ of
[Elhage et al. (2022)](https://transformer-circuits.pub/2022/toy_model/index.html), which suggests that in deep networks
it is typical for some linearly represented features not to be orthogonal. The cartoon picture is trying to squeeze more
than $$n$$ vectors into $$\mathbb{R}^n$$: they can't all be mutually orthogonal.

I'm interested in better understanding how this squeezing works in the setting of Shai et al.

## Summary

- **Reproduction.** A 4-layer transformer trained on the Mess3 process carries the fractal belief simplex in its final
  residual stream ($$R^2 = 1.00$$ on held-out prefixes). Shuffling beliefs before fitting gives $$R^2 = 0.00$$.
- **One process, too narrow.** The simplex degrades gracefully down to width 4 and fails at width 3, where the decoded
  beliefs collapse onto a curve. This threshold follows from LayerNorm: a model of width $$d$$ can only produce a family of
  predictions with $$d-2$$ parameters, and Mess3 needs 2.
- **Two processes.** When only one process is active at a time (_sparse_), both survive down to width 4, sharing nearly
  the same directions and told apart by position. When both are active at once (_dense_), the model gives up one process
  entirely as width falls, and it gives up the one whose memory is worth ~30 times less.
- **Packing.** In the dense regime, whenever both beliefs are kept their planes are close to orthogonal, the optimal
  Grassmannian packing. In the sparse regime the overlaps are far above optimal, because the sparse problem is an
  arrangement of affine patches, not a packing of subspaces.

## Setup

**Process.** Mess3, the process used by Shai et al., is a hidden Markov model with 3 hidden states and 3 tokens,
parametrised by $$(x, \alpha)$$. The _belief_ $$\eta$$ is the posterior over hidden states given the tokens seen so far;
it lives in a triangle, and over all prefixes it traces out a fractal.

**Model.** A pre-LayerNorm transformer with 4 layers, one attention head and context length 10, trained on freshly
sampled sequences. Only the residual width $$d$$ is varied; width 64 is the size used by Shai et al.

**Simplex images.** For every distinct prefix, take the final residual stream, fit an affine map to the exact belief on
half the prefixes, and plot the other half in the triangle, coloured by the true belief. Two numbers go with each image:

- **$$R^2$$:** the fraction of belief variance recovered on held-out prefixes.
- **Beyond last token:** the last token alone predicts about 68% of a Mess3 belief, which gives three coarse blobs. This
  measures how much of the rest, the fine structure that needs memory of earlier tokens, the model captures: 1 is
  perfect, and 0 or below means no memory beyond the last token.

**Two processes.** Process A has $$(x, \alpha) = (0.05, 0.85)$$ and B has $$(0.15, 0.60)$$. In the **sparse** regime each
sequence comes from one process, chosen at random, with separate tokens for each. In the **dense** regime both run side
by side, and each token is the pair of their tokens.

All numbers are means over 3 seeds; images are seed 0.

## 1. Reproducing the simplex, and 2. squeezing one process

{% include figure.liquid loading="eager" path="assets/img/projects/sbg-simplex-single.png" class="img-fluid rounded z-depth-1" zoomable=true %}

<div class="caption">
  One Mess3 process: the belief simplex in the final residual stream as the residual width shrinks. Top left: the true
  beliefs. Top row: the width-64 model recovers the fractal, and a control fit on shuffled beliefs collapses to the
  centre, so the image is a property of the trained network, not of the fitting procedure.
</div>

| Width | $$R^2$$ | Beyond last token | Loss above optimum |
| ----- | ------- | ----------------- | ------------------ |
| 64    | 1.00    | 0.99              | 0.01%              |
| 8     | 0.98    | 0.92              | 0.01%              |
| 4     | 0.93    | 0.78              | 0.08%              |
| 3     | 0.75    | 0.22              | 1.9%               |
| 2     | 0.61    | −0.24             | 37%                |

Width 8 keeps much of the fractal detail. Width 4 keeps the triangle and most of the memory, at almost no cost in loss.
Width 3 breaks: the decoded beliefs collapse onto a one-dimensional curve through the triangle.

**Why width 4.** Every read of the residual stream goes through a LayerNorm, which removes the mean coordinate and then
normalises to a sphere. So the output head sees a point on a sphere of dimension $$d - 2$$, and the next-token
distributions a model of width $$d$$ can produce form a family with at most $$d - 2$$ parameters. For Mess3 the optimal
prediction is an invertible affine function of the belief, a 2-parameter family, so exact prediction needs
$$d \ge 4$$. At $$d = 3$$ the head sees a circle, a 1-parameter family, which matches the curve in the width-3 image.

## 3. Two processes in a stream with room for one

{% include figure.liquid loading="lazy" path="assets/img/projects/sbg-simplex-pair.png" class="img-fluid rounded z-depth-1" zoomable=true %}

<div class="caption">
  Two Mess3 processes squeezed together as the residual width shrinks. Left: sparse, one process per sequence. Right:
  dense, both processes every token.
</div>

Beyond last token, per process:

| Width | Sparse A | Sparse B | Dense A | Dense B |
| ----- | -------- | -------- | ------- | ------- |
| 8     | 0.96     | 0.89     | 0.94    | 0.74    |
| 6     | 0.94     | 0.85     | 0.85    | 0.22    |
| 5     | 0.94     | 0.69     | 0.80    | −1.70   |
| 4     | 0.91     | 0.54     | 0.75    | −3.35   |
| 3     | −0.34    | 0.08     | 0.44    | −3.76   |

**Sparse: both survive to width 4,** with B losing detail first, and both fail at width 3, the same threshold as for one
process. Only one process is active at a time, so the model never needs more than one 2-parameter family of predictions
at once.

**Dense: one process is given up.** B is already weak at width 6 and gone below it; the large negative values mean the
decoder cannot even recover B's last token, so B is dropped, not kept at low resolution. Two beliefs active at once need
a 4-parameter family of predictions, so exact prediction needs $$d \ge 6$$; width 6 is necessary but in practice not
quite enough.

**Why B.** B's history is worth about 30 times less in next-token loss than A's (0.002 against 0.061 nats for memory
beyond the last token), so it is the cheap one to drop. Dropping B entirely costs 0.50% of the dense optimal loss, and
the width-4 dense models sit at 0.50% above optimum: they predict A almost perfectly and B not at all.

## 4. Comparison with Grassmannian packing

Each belief occupies a 2-dimensional plane in the residual stream: the directions the activation moves as that belief
moves. The _overlap_ of two planes is the mean $$\cos^2$$ of their two principal angles (0 for orthogonal planes, 1 for
identical ones). Planes are measured in the part of the stream that LayerNorm can see, which has dimension $$d - 1$$.
The question is how close the learned arrangement comes to the best possible one.

{% include figure.liquid loading="lazy" path="assets/img/projects/sbg-packing.png" class="img-fluid rounded z-depth-1" zoomable=true %}

<div class="caption">
  Left: largest overlap between the two learned belief planes against the optimal Grassmannian packing and the Rankin
  bound. Right: in the sparse regime, how far apart the two belief patches sit, in units of their spread.
</div>

| Width | Optimum | Random planes | Sparse | Dense         |
| ----- | ------- | ------------- | ------ | ------------- |
| 4     | 0.50    | 0.67          | 0.75   | –             |
| 5     | 0       | 0.50          | 0.29   | 0.26 (1 seed) |
| 6     | 0       | 0.40          | 0.14   | 0.013         |
| 8     | 0       | 0.29          | 0.11   | 0.020         |

**Dense: near the optimal packing.** Whenever both beliefs are kept, their planes are nearly orthogonal: overlap about
0.01–0.02, against 0.3–0.4 for random planes. Two planes fit orthogonally once $$d \ge 5$$, so packing is not what limits
the dense models: the readout bound $$d \ge 6$$ is reached first.

**Sparse: not a packing problem.** Overlaps are far above the optimum, and at width 4 above even the random baseline, yet
both processes are decoded well and loss is within 0.14% of optimal. The two belief patches share directions but sit
about 6 spreads apart, so position, not direction, tells them apart. Only one patch is active at a time, so nothing
forces their directions apart.

The conclusions are unchanged when angles are recomputed in the metric the output head's LayerNorm imposes, or with
planes fitted to optimal logits instead of beliefs. The repository's theory section derives the readout bounds, the
packing optimum, and why the softmax turns the dense problem into one of independent planes.

## Limitations

- **Small scale.** Three seeds, one pair of Mess3 processes, one architecture, fixed training lengths. The dense
  comparison with packing rests on 7 models that keep both beliefs.
- **One process pair.** A and B differ a lot in how much their memory is worth, which decides which one is dropped. A
  symmetric pair would test whether the model then shares the cost or still picks one.
- **Final layer only.** Planes and overlaps are measured where the output head reads.
- **Linear measurement of a nonlinear readout.** Planes are affine fits before the final normalisation.

This project grew out of the Iliad Intensive. The code was written with the assistance of Claude (Anthropic).
