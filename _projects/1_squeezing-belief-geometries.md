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
but not both?

The short answer: **the model keeps what its output needs and, as room runs out, gives up the rest, cheapest first.** The
limit is set by what the output layer can read, not by how many directions fit in the space.

**Code and full write-up:** [github.com/KyleJWright/squeezing-belief-geometries](https://github.com/KyleJWright/squeezing-belief-geometries).
The repository README has every table, and
[THEORY.md](https://github.com/KyleJWright/squeezing-belief-geometries/blob/main/THEORY.md) has the mathematics.

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

**The main finding.** The model keeps what the next-token prediction needs and, as room runs out, gives up everything
else, cheapest first. Each experiment shows a different piece of this:

- **Reproduction.** A 4-layer transformer trained on the Mess3 process carries the fractal belief simplex in its final
  residual stream ($$R^2 = 1.00$$ on held-out prefixes); shuffling beliefs before fitting gives $$R^2 = 0.00$$.
- **One process, too narrow.** Fine fractal detail goes first, at almost no cost in loss. The simplex survives down to
  width 4 and fails at width 3, where the decoded beliefs collapse onto a curve. The threshold matches what the output
  layer can read: after LayerNorm, a model of width $$d$$ can produce a family of predictions with at most $$d-2$$
  parameters, and Mess3 needs 2.
- **Two processes at once (dense).** As width falls, the model drops one process entirely, and it is the one whose memory
  is worth ~30 times less in loss. Dropping it costs almost exactly what that memory is worth (0.50% of the optimal loss).
- **Two processes, one at a time (sparse).** Both survive to width 4. With one shared set of tokens the two beliefs
  collapse onto a single plane at every width, even with room for two, because the output needs only one prediction, and
  it costs nothing measurable.
- **Which process is running?** With room to spare (width 8) the model also tracks which process is running, as well as an
  optimal observer can, in one extra direction. Squeezed, that is the first thing to go: at width 4 the model learns it
  early in training and then gives it up.
- **Packing was the wrong lens.** I expected the Grassmannian packing problem to describe how the beliefs share the space.
  It mostly does not: when both beliefs are kept their planes are near-orthogonal, but the readout limit always binds
  first, and in the sparse case the problem is not a packing problem at all.
- **The planes are not flat.** Measured with a Gauss map, every belief patch is a wrinkled sheet whose local tangent
  planes tilt 12–50° from its best-fit plane. Linear decoders never notice, because the wrinkles lie outside the plane.
  The planes are the linearly readable summary of curved patches.

## Why this matters

Two lessons from this toy setting may be worth testing more widely.

- **Whether a latent variable is represented depends on room, not only on whether it is useful.** The same training gives
  a model that tracks which process is running (width 8) and one that learns it briefly and then gives it up (width 4)
  while staying within 0.15% of the optimal loss, because the output does not need it. So a probe that finds nothing in a
  narrow model says little about whether the variable matters to the task or can be learned, and a signal found in a wide
  model need not survive in a narrower one. Methods that look for latent knowledge with probes need to allow for this.
- **What gets superposed or dropped is decided by the readout.** The two behaviours of Elhage et al.'s toy models
  reappear: dense features force the least valuable one out, sparse ones share directions. But here what decides is what
  the output needs and what each piece is worth, not how many vectors fit in the space.

These are small models trained on a toy process; whether the lessons carry over is an open question.

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
sequence comes from one process, chosen at random. It comes in two versions: with **separate tokens** for each process,
so every token reveals the process, and with one **shared** set of 3 tokens, so the model must infer the process from the
statistics of what it has seen. In the **dense** regime both run side by side, and each token is the pair of their
tokens.

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
prediction is an invertible affine function of the belief, a 2-parameter family. If the model's prediction varies
smoothly with the belief, exact prediction then needs $$d \ge 4$$. At $$d = 3$$ the head sees a circle, a 1-parameter
family, which matches the curve in the width-3 image. This is a heuristic, not a theorem: with a context of 10 there are
only finitely many beliefs, and nothing forces the network's map from prefixes to predictions to be smooth.

## 3. Two processes in a stream with room for one

{% include figure.liquid loading="lazy" path="assets/img/projects/sbg-simplex-pair.png" class="img-fluid rounded z-depth-1" zoomable=true %}

<div class="caption">
  Two Mess3 processes squeezed together as the residual width shrinks. Left: sparse, one process per sequence. Right:
  dense, both processes every token.
</div>

Beyond last token, per process (sparse here means separate tokens; the shared-token version is below):

| Width | Sparse A | Sparse B | Dense A | Dense B |
| ----- | -------- | -------- | ------- | ------- |
| 8     | 0.96     | 0.89     | 0.94    | 0.74    |
| 6     | 0.94     | 0.85     | 0.85    | 0.22    |
| 5     | 0.94     | 0.69     | 0.80    | −1.70   |
| 4     | 0.91     | 0.54     | 0.75    | −3.35   |
| 3     | −0.34    | 0.08     | 0.44    | −3.76   |

Loss above optimum:

| Width                   | 8     | 6     | 5     | 4     | 3    | 2   |
| ----------------------- | ----- | ----- | ----- | ----- | ---- | --- |
| Sparse, separate tokens | 0.02% | 0.03% | 0.04% | 0.14% | 1.6% | 90% |
| Sparse, shared tokens   | 0.01% | 0.03% | 0.05% | 0.15% | 2.0% | 13% |
| Dense                   | 0.03% | 0.07% | 0.28% | 0.50% | 1.6% | 16% |

**Sparse: both survive to width 4,** with B losing detail first, and both fail at width 3, the same threshold as for one
process. Only one process is active at a time, so the model never needs more than one 2-parameter family of predictions
at once.

**Dense: one process is given up.** B is already weak at width 6 and gone below it; the large negative values mean the
decoder cannot even recover B's last token, so B is dropped, not kept at low resolution. Two beliefs active at once need
a 4-parameter family of predictions, so by the same heuristic exact prediction needs $$d \ge 6$$; in practice width 6 is
not quite enough.

**Why B.** B's history is worth about 30 times less in next-token loss than A's (0.002 against 0.061 nats for memory
beyond the last token), so it is the cheap one to drop. Dropping B entirely costs 0.50% of the dense optimal loss, and
the width-4 dense models sit at 0.50% above optimum: they predict A almost perfectly and B not at all.

### Sparse with one shared set of tokens

{% include figure.liquid loading="lazy" path="assets/img/projects/sbg-simplex-shared.png" class="img-fluid rounded z-depth-1" zoomable=true %}

<div class="caption">
  Sparse regime with separate tokens per process (left) and one shared set of tokens (right), as the residual width
  shrinks.
</div>

| Width | Separate tokens: A | Separate tokens: B | Shared tokens: A | Shared tokens: B |
| ----- | ------------------ | ------------------ | ---------------- | ---------------- |
| 8     | 0.96               | 0.89               | 0.78             | 0.69             |
| 6     | 0.94               | 0.85               | 0.75             | 0.66             |
| 5     | 0.94               | 0.69               | 0.73             | 0.68             |
| 4     | 0.91               | 0.54               | 0.63             | 0.67             |
| 3     | −0.34              | 0.08               | 0.11             | 0.11             |

(Beyond last token, per process.) With shared tokens, both beliefs still survive down to width 4 and fail at width 3,
the same threshold as before: whichever process is running, the prediction is a distribution over the same 3 tokens, a
2-parameter family. B holds up better than with separate tokens (0.67 against 0.54 at width 4). Loss is as close to
optimal as with separate tokens (0.15% above at width 4, against 0.14%), even though the optimal predictor now has to
weigh up which process is running.

The scores are lower at the wider widths, but this does not mean the model is worse. Each process's belief is computed as
if the process were known. With shared tokens the model cannot know it, especially early in a sequence, so even the
Bayes-optimal predictor would not carry these beliefs exactly, and the ceiling is below 1.

## 4. Geometry of the belief patches

Each belief occupies a patch of the residual stream. Two questions about its shape: how do two processes' patches sit
relative to each other, and is each patch flat? Everything here is measured in the part of the stream LayerNorm can see,
of dimension $$d - 1$$.

The starting hypothesis was the cartoon of superposition: two beliefs squeezed into too little room should arrange their
planes as an optimal Grassmannian packing, as far apart as the space allows. **This turned out to be mostly the wrong
lens**, for two reasons: the readout limit binds before packing ever does, and the patches are not flat planes but
wrinkled sheets.

### Overlap between processes

The _overlap_ of two belief planes is the mean $$\cos^2$$ of their principal angles: 0 for orthogonal planes, 1 for
identical ones. Each plane is the best linear fit to its patch, so overlap measures how much two beliefs interfere for a
linear reader, which is how every later layer first reads the stream.

{% include figure.liquid loading="lazy" path="assets/img/projects/sbg-packing.png" class="img-fluid rounded z-depth-1" zoomable=true %}

<div class="caption">
  Left: largest overlap between the two learned belief planes against the optimal Grassmannian packing and the Rankin
  bound, for sparse (separate and shared tokens) and dense models. Right: in the sparse regime, how far apart the two
  belief patches sit, in units of their spread.
</div>

| Width | Optimum | Random planes | Sparse, separate tokens | Offset separation | Sparse, shared tokens | Offset separation | Dense         |
| ----- | ------- | ------------- | ----------------------- | ----------------- | --------------------- | ----------------- | ------------- |
| 3     | 1       | 1             | 1.00                    | 2.4               | 1.00                  | 0.05              | –             |
| 4     | 0.50    | 0.67          | 0.75                    | 5.9               | 0.94                  | 0.17              | –             |
| 5     | 0       | 0.50          | 0.29                    | 6.0               | 0.905                 | 0.21              | 0.26 (1 seed) |
| 6     | 0       | 0.40          | 0.14                    | 6.0               | 0.94                  | 0.33              | 0.013         |
| 8     | 0       | 0.29          | 0.11                    | 5.9               | 0.97                  | 0.38              | 0.020         |

_Offset separation_ is the distance between the centres of the two processes' patches, divided by their typical
spread; _random planes_ is the expected overlap of independent random planes, $$2/(d-1)$$.

- **Dense: near-orthogonal, but never constrained.** Whenever both beliefs are kept (all seeds at widths 6 and 8) the
  planes are nearly orthogonal, overlap 0.01–0.02 against 0.3–0.4 for random planes. But orthogonality is easy there: it
  is possible from $$d \ge 5$$, and the readout bound $$d \ge 6$$ is reached first. At width 5 only one seed keeps B at all
  (overlap 0.26). Near-orthogonality may also simply be inherited from the optimal logits, whose A and B parts lie in
  orthogonal subspaces; the data cannot tell these apart.
- **Sparse, separate tokens: an arrangement, not a packing.** Overlaps are far above the optimum, yet both processes
  decode well and loss is within 0.14% of optimal: the patches share directions but sit about 6 spreads apart. That
  separation comes from the token embeddings, not from anything the model had to learn, as the shared-token models show.
- **Sparse, shared tokens: one plane for two beliefs.** Without separate tokens the offset drops to 0.05–0.4 spreads, and
  the planes coincide (overlap 0.9–0.97) at every width, including width 8, where there is room for two. Whichever process
  is running, the prediction is a distribution over the same 3 tokens, and one plane carries it for both, at no
  measurable cost in loss. Section 5 asks whether the model also tracks which process is running.

These conclusions hold in the metric the output head's LayerNorm imposes and with planes fitted on optimal logits instead
of beliefs.

### Curvature within a process: the Gauss map

A linear decoder cannot tell whether a patch is flat: a surface that bends into an extra direction decodes perfectly once
that direction is projected away. So the bending is measured directly.

- **Gauss map.** At 64 anchor beliefs spread over the triangle, a local affine fit over nearby beliefs gives the local
  tangent plane, a point of the same Grassmannian. The _tilt_ is its angle from the global plane. A synthetic flat patch
  gives the tilt that finite neighbourhoods produce on their own: the _flat floor_.
- **Bending.** The average activation at each belief minus the global fit, split into the part outside the plane
  (bending) and inside it (a reparametrisation of the plane, which is not curvature), as a fraction of the variance the
  global fit explains.

{% include figure.liquid loading="lazy" path="assets/img/projects/sbg-gauss.png" class="img-fluid rounded z-depth-1" zoomable=true %}

<div class="caption">
  Gauss map: local tilt across the triangle for selected models (top), and the excess tilt and out-of-plane bending by
  width (bottom).
</div>

| Model                      | Excess tilt, width 8 | width 4 | Bending, width 8    | width 4 |
| -------------------------- | -------------------- | ------- | ------------------- | ------- |
| One process                | 34.8°                | 27.5°   | 17.5%               | 20.1%   |
| Sparse, separate tokens, A | 19.1°                | 12.0°   | 4.2%                | 1.7%    |
| Sparse, separate tokens, B | 35.7°                | 24.7°   | 22.5%               | 9.3%    |
| Sparse, shared tokens, A   | 34.5°                | 27.4°   | 26.1%               | 29.5%   |
| Sparse, shared tokens, B   | 50.1°                | 29.1°   | 33.6%               | 47.6%   |
| Dense, A                   | 21.6°                | 17.2°   | 5.4% (both beliefs) | –       |

(Excess tilt is the mean tilt above the flat floor, which is about 2–7°. Bending is undefined for dense models of width 5
or less, where the two planes fill the readable space.)

- **No patch is flat.** Local tangent planes tilt 12–50° beyond the flat floor, in every model.
- **Wrinkled, not bowed.** Quadratic terms in the belief capture only a few percent of the activation variance, far less
  than the total bending: the patches are corrugated at fine scales rather than curved like a bowl.
- **Most bent at the corners.** In most maps, tilt is largest near the vertices and edges, where beliefs are most
  confident, and smallest in the middle.
- **Shared tokens bend most.** One possible reason, not tested: with shared tokens the model also tracks which process is
  running (section 5), and that coordinate may appear here as bending out of the shared plane.
- **This is much of what the linear fits miss.** The global affine fit explains between about 40% and 97% of the
  activation variance, depending on the model, and this bending accounts for roughly a quarter to four-fifths of what it
  misses.

So the planes are the linearly readable summary of curved patches: each belief uses more than two dimensions, and the
extra ones carry wrinkles that a linear reader never sees. The packing of planes describes linear interference between
beliefs, not their full geometry. A caveat: with a context of 10 there are finitely many beliefs, so the "surface" is an
average over nearby beliefs, and the finest wrinkles cannot be told apart from scatter.

## 5. Disentangling: does the model know which process it is in?

With shared tokens nothing tells the model which process produced a sequence. The exact posterior
$$\pi = P(A \mid \text{prefix})$$ says how far the evidence has gone, and it goes less far than one might expect: guessing
the process from $$\pi$$ is right only 62% of the time after two tokens and 81% after ten. The full belief therefore lives
in the 5-simplex over (process, state), the _join_ of the two triangles, and $$\pi$$ says where along the join it sits.

**What to expect.** Shai et al. find that the residual stream carries the full belief state, not just the next-token
prediction. Here that is the 5-dimensional joint belief, with coordinates $$(\pi, \eta_A, \eta_B)$$. The output, by
contrast, needs only the 2-dimensional prediction $$\pi\, \eta_A N_A + (1 - \pi)\, \eta_B N_B$$, in which $$\pi$$ is
already folded. So the final layer could hold anything from the 2-dimensional prediction up to the full 5-dimensional
belief. And $$\pi$$ competes with the prediction for the $$d - 2$$ dimensions the output head can use, where the
prediction is worth far more.

### Does it track the process?

A ridge probe is fitted from the residual stream to $$\pi$$ at every layer, on the shared-token models of section 3.

{% include figure.liquid loading="lazy" path="assets/img/projects/sbg-probe.png" class="img-fluid rounded z-depth-1" zoomable=true %}

<div class="caption">
  Probe for which process is running: accuracy by position against the Bayes-optimal guess (left), probe R² by layer
  and width (middle), and how confident the exact posterior is (right).
</div>

| Width                                         | 8     | 6     | 5     | 4     | 3     |
| --------------------------------------------- | ----- | ----- | ----- | ----- | ----- |
| Probe $$R^2$$ for $$\pi$$, final layer        | 0.87  | 0.62  | 0.32  | 0.12  | 0.08  |
| Probe accuracy after 10 tokens (Bayes: 0.806) | 0.798 | 0.762 | 0.665 | 0.585 | 0.563 |
| Loss above optimum                            | 0.01% | 0.03% | 0.05% | 0.15% | 2.0%  |

At width 8 the model tracks the process essentially as well as possible: its probe follows the Bayes-optimal accuracy at
every position, and the posterior is built up through the layers. Combined with section 4, the final layer holds **one
shared belief plane plus a separate direction for $$\pi$$**, as far as linear probes can tell: nearer the
low-dimensional end, and not two separate belief models.

As the stream narrows, process tracking is the first thing to go. Loss stays within 0.15% of optimal down to width 4,
but the probe falls to $$R^2 = 0.12$$ and its accuracy stalls after a few tokens. This fits the readout argument of
section 2: the output head has $$d - 2$$ dimensions, the prediction needs 2, and only what is left can carry $$\pi$$. At
width 4 nothing is left. Dropping $$\pi$$ costs little: the optimal prediction already mixes the two processes'
predictions in proportion to $$\pi$$, so the posterior is folded into the prediction.

### Does training learn one model and then split it?

Several settings learn a merged description first and split it later (causal-state splitting, deterministic annealing,
stagewise learning in deep networks), which suggests a hypothesis: the shared-token model first learns one averaged
process, then splits into two. Shared-token models at widths 8 and 4 were retrained with snapshots through training, and
every snapshot was analysed in the same way.

{% include figure.liquid loading="lazy" path="assets/img/projects/sbg-develop.png" class="img-fluid rounded z-depth-1" zoomable=true %}

<div class="caption">
  Shared-token training at width 8 (left) and 4 (right): loss above optimum, overlap of the A and B belief planes,
  offset between their patches, probe R² for which process is running, and each belief's R², by layer and training
  step.
</div>

| Step                                 | 25   | 200   | 400   | 1,000 | 4,000 | 15,000 |
| ------------------------------------ | ---- | ----- | ----- | ----- | ----- | ------ |
| Width 8: loss above optimum          | 2.6% | 0.87% | 0.35% | 0.28% | 0.13% | 0.03%  |
| Width 8: plane overlap (final layer) | 0.99 | 0.89  | 0.92  | 0.93  | 0.95  | 0.96   |
| Width 8: offset (spreads)            | 0.04 | 0.17  | 0.25  | 0.30  | 0.34  | 0.38   |
| Width 8: probe $$R^2$$ for $$\pi$$   | 0.33 | 0.53  | 0.68  | 0.68  | 0.79  | 0.87   |
| Width 4: loss above optimum          | 7.0% | 2.1%  | 1.2%  | 0.52% | 0.30% | 0.14%  |
| Width 4: probe $$R^2$$ for $$\pi$$   | 0.02 | 0.17  | 0.20  | 0.18  | 0.13  | 0.12   |

The hypothesis is not borne out: the network never splits into two independent models.

- **Width 8: one model gains a coordinate.** The probe for $$\pi$$ rises steadily over training, and the offset between
  the A and B patches grows with it: the activations shift along a direction that tracks the process. The two belief
  planes stay nearly identical throughout, apart from a brief separation while loss is falling fastest, which reverses.
- **Width 4: learned, then given up.** The probe rises to about 0.2 by step 600 and then falls back to 0.12, while loss
  keeps improving. The model tracks the process early and lets it go as it sharpens the prediction, which fills its two
  usable dimensions. A feature is learned and then squeezed out by a more valuable one.
- **No sharp transition.** Loss falls smoothly at this snapshot resolution; nothing looks like a sudden phase change.

## Limitations

- **Width changes more than storage.** Shrinking the residual width shrinks the space the belief is stored in, but also
  the space the network computes the belief update in. A failure at width 3 could come from storage, computation or
  optimisation; the large spread between seeds at width 3 suggests optimisation plays a part.
- **The readout bound is a heuristic at finite context.** It is a theorem only in the long-context limit, and depends on
  the fractal's Hausdorff dimension, which is not computed here.
- **Linear probes only.** At small widths a belief may still be present but encoded nonlinearly, which a linear probe
  would miss. This matters for the link to superposition, which is about features recovered nonlinearly despite
  interference.
- **Training dynamics are sampled coarsely.** The development run has 3 seeds, two widths and a limited number of
  snapshots; a brief transition between snapshots would be missed.
- **Curvature is measured on averages.** The Gauss map and bending use averages over nearby beliefs and a finite
  neighbourhood, so wrinkles finer than the neighbourhood are not resolved.
- **The packing comparison is never constrained.** Dense models keep both beliefs only where orthogonal planes are
  possible, and their near-orthogonality may be inherited from the structure of the optimal logits.
- **One process pair.** A and B differ a lot in how much their memory is worth, which decides which one is dropped. A
  symmetric pair would test whether the model then shares the cost or still picks one.
- **Images are sensitive to cheap structure.** Width 8 loses visible fractal detail at no measurable cost in loss. The
  pictures magnify structure that is worth very little to the loss.
- **Small scale.** Three seeds, one architecture, fixed training lengths, final layer only. The dense comparison with
  packing rests on 7 models that keep both beliefs.

## How this was made

I set the questions and the direction, ran every experiment on my laptop, checked the mathematics and the numbers, and
decided what to keep, cut and follow up. The code and first drafts of the text and derivations were written with Claude
(Anthropic). Several turns in the project came from that checking and questioning: asking whether the softmax distorts
the linear picture, which space the angles should be measured in, whether separate tokens were giving the process away,
and whether a model of two processes starts as one and later splits in two.

This project grew out of the Iliad Intensive.
