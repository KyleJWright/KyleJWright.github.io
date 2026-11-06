---
layout: distill
title: Refinement in the Real World
description: Electromagnetism as a sequence of refinements, from two vector fields to differential cohomology, and what that might say about representations in machine learning.
tags: physics
categories: physics
date: 2026-11-06
featured: false
sitemap: false
related_posts: false

authors:
  - name: Kyle Wright
    url: "https://kylejwright.github.io"

bibliography: 2026-11-06-refinement.bib

toc:
  - name: Averages forget
  - name: Representations depend on context
  - name: Electromagnetism as a sequence of refinements
    subsections:
      - name: Two fields become one form
      - name: Charge is quantised
      - name: The potential is measurable
  - name: A limit and a colimit
  - name: Back to representations

_styles: >
  .tikz-figure { margin: 1.5rem 0; }
  .tikz-figure img { width: 100%; height: auto; }
---

A recurring move in physics is to notice that the object you were using was a shadow of a richer one. The richer object is not more decoration: it
explains facts that the shadow can't, and it often makes the shadow's awkward features (frame dependence, unexplained integrality, quantities that
should vanish but don't) fall out automatically. Mathematicians call this kind of upgrade a _refinement_.

This post walks through one example that I find especially clean, electromagnetism, and then asks what the pattern might say about how we describe
the internal representations of neural networks.

## Averages forget

Start with something familiar. A moment of a distribution, such as the mean $$\mathbb{E}[X]$$ or the variance $$\operatorname{Var}[X]$$, is an
average, and averages forget. Figure 1 shows three distributions with mean $$0$$ and variance $$1$$: a normal, a uniform, and a mixture of two narrow
bumps. Knowing the first two moments only pins down the distribution if you already know it belongs to a two-parameter family such as the normals.

<div class="tikz-figure">
  <img class="tikz" src="{{ '/assets/img/posts/refinement/moments.svg' | relative_url }}" alt="Three densities with the same mean and variance">
</div>
<div class="caption">
  Figure 1. Three densities with mean 0 and variance 1. The first two moments can't tell them apart.
</div>

Adding more moments does not always rescue you either. The lognormal distribution has finite moments of every order, yet there are other
distributions with exactly the same moments<d-cite key="heyde1963lognormal"></d-cite>. The sequence of moments is a genuinely coarser invariant
than the distribution.

## Representations depend on context

The same issue arises when we summarise what a neural network represents. It is tempting to look for a single global linear direction for a concept
and to average activations over a dataset to find it. But the geometry of activations can differ between contexts, for example between prose and
code, and a single global average blurs them together.

Two recent papers take this seriously using the language of gauge theory and sheaves. Javidnia<d-cite key="javidnia2026gauge"></d-cite> clusters
the activations of a language model into local _charts_, fits transport maps between overlapping charts, and measures the obstructions to stitching
them into one global description, including a holonomy-type obstruction around loops of charts. Grant and Wang<d-cite key="grant2026gluing"></d-cite>
build a cellular sheaf over a graph of paraphrases: each paraphrase carries its own representation, edges carry maps comparing them, and the global
sections of the sheaf pick out what is consistent across all ways of saying the same thing, while the first cohomology measures what fails to glue.

The way I read this: paraphrasing behaves like a gauge transformation. Two paraphrases are different presentations of the same content, so the
content should be what survives after identifying them. That is a categorical equivalence rather than an equality, and the natural way to
assemble locally consistent data into global data is the passage from a presheaf to a sheaf. This framing is my own gloss rather than a claim either
paper makes in exactly these words.

To see why the refinement matters, and not just the language, it helps to look at a case where the full story is known.

## Electromagnetism as a sequence of refinements

### Two fields become one form

Through the nineteenth century the electric and magnetic fields were gradually tied together. Ørsted (1820) saw a current deflect a compass needle,
Faraday (1831) found that a changing magnetic field induces a current, Lenz (1834) fixed the sign, Maxwell unified the equations, and Hertz
(1887–88) produced and detected the predicted waves.

Special relativity then shows that $$\mathbf{E}$$ and $$\mathbf{B}$$ are not separately meaningful. A charge at rest has a purely electric field; the
same charge seen from a moving frame also has a magnetic field circling its direction of motion (Figure 2). Which part of the field you call electric
and which magnetic depends on the observer.

<div class="tikz-figure">
  <img class="tikz" src="{{ '/assets/img/posts/refinement/boost.svg' | relative_url }}" alt="A charge at rest and seen from a moving frame">
</div>
<div class="caption">
  Figure 2. A boost mixes E and B. The frame-independent object is the field strength F.
</div>

The frame-independent object is a single 2-form on spacetime $$M$$, the field strength. Up to sign conventions,

$$
F = \sum_i E_i \, dx^i \wedge dt + \tfrac{1}{2} \sum_{i,j,k} \epsilon_{ijk} B_i \, dx^j \wedge dx^k .
$$

Maxwell's equations become two lines. The homogeneous pair (no magnetic monopoles, and Faraday's law) is

$$
dF = 0,
$$

and the pair with sources (Gauss's law and the Ampère–Maxwell law) is

$$
d{\star}F = J,
$$

where $$\star$$ is the Hodge star of the spacetime metric and $$J$$ is the current 3-form. In vacuum, $$d{\star}F = 0$$ as well, and on a compact
Riemannian slice the pair $$dF = d{\star}F = 0$$ says that $$F$$ is harmonic, which is where Hodge theory enters.

Because $$dF = 0$$, the field strength defines a de Rham class $$[F] \in H^2(M;\mathbb{R})$$. Integrating $$F$$ over a closed surface measures the
total magnetic flux leaving it, which is the magnetic charge enclosed. Locally, $$dF = 0$$ also means $$F = dA$$ for a potential 1-form $$A$$, and
the potential is only defined up to a gauge transformation $$A \mapsto A + d\lambda$$, which leaves $$F$$ unchanged.

So far the story is: two frame-dependent vector fields refine to one 2-form, and the 2-form carries a real cohomology class.

### Charge is quantised

Millikan's oil-drop experiment<d-cite key="millikan1913"></d-cite> showed that electric charge comes in integer multiples of a basic unit. Dirac
noticed that quantum mechanics links this to magnetism<d-cite key="dirac1931"></d-cite>. If a particle of charge $$q$$ moves in the field, its
wavefunction must be single-valued, and that forces

$$
\frac{q}{2\pi\hbar} \int_\Sigma F \in \mathbb{Z}
$$

for every closed surface $$\Sigma$$. In cohomological terms, the class $$\left[\tfrac{q}{2\pi\hbar}F\right]$$ must lie in the image of
$$H^2(M;\mathbb{Z}) \to H^2(M;\mathbb{R})$$. A single magnetic monopole anywhere in the universe would therefore force all electric charges to be
integer multiples of a basic unit.

The integral class is extra data. It is the first Chern class $$c_1$$ of a complex line bundle on which the wavefunction lives, and it is not
determined by $$F$$: the map $$H^2(M;\mathbb{Z}) \to H^2(M;\mathbb{R})$$ forgets torsion. So the naive object, a real cohomology class, is refined
to an integral one.

### The potential is measurable

Classically one regards $$A$$ as a convenient fiction and $$F$$ as the physical field. Aharonov and Bohm<d-cite key="aharonov1959"></d-cite>
showed that this is wrong in quantum mechanics. Send electrons around both sides of a long shielded solenoid (Figure 3). Outside the solenoid
$$F = 0$$, so every naive invariant built from $$F$$ vanishes where the electrons go. Yet the interference pattern shifts by the phase

$$
\exp\!\left( \frac{iq}{\hbar} \oint_{\gamma_1 - \gamma_2} A \right) = \exp\!\left( \frac{iq\Phi}{\hbar} \right),
$$

where $$\Phi$$ is the flux inside the solenoid. The shift was observed by Chambers<d-cite key="chambers1960"></d-cite> and, with the field completely
shielded by a superconductor, by Tonomura and collaborators<d-cite key="tonomura1986"></d-cite>.

<div class="tikz-figure">
  <img class="tikz" src="{{ '/assets/img/posts/refinement/ab.svg' | relative_url }}" alt="Aharonov-Bohm set-up">
</div>
<div class="caption">
  Figure 3. The Aharonov–Bohm set-up. The field strength vanishes on every path the electron can take, but the holonomy around the solenoid does not.
</div>

The region the electrons explore is space minus a cylinder, which deformation retracts onto a circle. Its second cohomology vanishes, so neither
$$[F]$$ nor the Chern class can see anything. What is measured is the _holonomy_ of $$A$$ around the loop, a gauge-invariant number in
$$U(1)$$. Wu and Yang summarised the situation neatly<d-cite key="wu1975"></d-cite>: $$F$$ underdescribes electromagnetism, $$A$$ overdescribes it,
and the phase factor describes it exactly.

The object that captures all of this is a line bundle with connection up to isomorphism, equivalently a Cheeger–Simons differential
character<d-cite key="cheeger1985"></d-cite> or a class in degree-2 Deligne cohomology<d-cite key="brylinski1993"></d-cite>. Normalising so that
$$F$$ has integer periods, write $$\check{H}^2(M)$$ for the group of these. It sits in two short exact sequences,

$$
0 \to \Omega^1(M)/\Omega^1_{\mathbb{Z}}(M) \to \check{H}^2(M) \to H^2(M;\mathbb{Z}) \to 0,
$$

$$
0 \to H^1(M;\mathbb{R}/\mathbb{Z}) \to \check{H}^2(M) \to \Omega^2_{\mathbb{Z}}(M) \to 0,
$$

where $$\Omega^k_{\mathbb{Z}}$$ denotes closed forms with integer periods. The first sequence says a differential class is a topological class plus
a potential modulo large gauge transformations. The second says it is a curvature plus flat data, and the flat part $$H^1(M;\mathbb{R}/\mathbb{Z})$$
is precisely where the Aharonov–Bohm phase lives. Simons and Sullivan showed that these sequences, organised into the hexagon of Figure 4,
characterise the theory uniquely<d-cite key="simons2008"></d-cite>.

<div class="tikz-figure">
  <img class="tikz" src="{{ '/assets/img/posts/refinement/hexagon.svg' | relative_url }}" alt="Differential cohomology hexagon">
</div>
<div class="caption">
  Figure 4. The differential cohomology hexagon in degree 2. The diagonals through the centre are the two exact sequences above.
</div>

The moral: the naive invariant $$H^2(M;\mathbb{R})$$ is not enough, and neither is $$H^2(M;\mathbb{Z})$$. Both must be refined to differential
cohomology before the theory accounts for what we can measure.

## A limit and a colimit

One might hope to define the refined object by a commuting square: a differential class should be a curvature form together with an integral class
whose images in real cohomology agree. That is the ordinary pullback of $$\Omega^2_{\mathrm{cl}}(M) \to H^2(M;\mathbb{R}) \leftarrow H^2(M;\mathbb{Z})$$,
and it is too coarse. The map from $$\check{H}^2(M)$$ onto it has kernel $$H^1(M;\mathbb{R})/H^1(M;\mathbb{Z})$$, the flat connections on a
topologically trivial bundle. For the Aharonov–Bohm region this kernel is $$\mathbb{R}/\mathbb{Z}$$: the strict pullback loses exactly the effect
we were trying to capture.

The fix is to remember not just _that_ the two images agree but _how_. Hopkins and Singer<d-cite key="hopkins2005"></d-cite> define a differential
cocycle as a triple $$(c, h, \omega)$$: an integral cochain $$c$$, a closed form $$\omega$$, and a real cochain $$h$$ with $$\delta h = \omega - c$$.
The cochain $$h$$ is a witness that the square commutes, and two cocycles are equivalent only if their witnesses are compatible too. This is a
_homotopy pullback_ (a homotopy limit), shown in Figure 5, and $$\check{H}^2(M)$$ is its group of components.

<div class="tikz-figure" style="max-width: 520px; margin-left: auto; margin-right: auto;">
  <img class="tikz" src="{{ '/assets/img/posts/refinement/pullback.svg' | relative_url }}" alt="Homotopy pullback square">
</div>
<div class="caption">
  Figure 5. Differential cohomology as a homotopy pullback. The square commutes only up to the homotopy h, and h is part of the data.
</div>

There is also a colimit description, which is closer to how a physicist thinks. On a single coordinate patch a gauge field is a 1-form $$A$$, and
gauge transformations $$g: U \to U(1)$$ act by $$A \mapsto A + g^{-1}dg$$. Rather than taking the quotient set, take the quotient groupoid,
a homotopy quotient $$\Omega^1 /\!/ C^\infty(-,U(1))$$. This is a presheaf of groupoids, and it does not yet know how to glue: on a sphere around
a monopole there is no global potential at all. Imposing descent (stackification, the groupoid version of sheafification) produces the stack
$$\mathbf{B}U(1)_\nabla$$ of line bundles with connection, and $$\check{H}^2(M)$$ is the set of isomorphism classes of its sections over $$M$$.
Bunke, Nikolaus and Völkl develop this point of view systematically using sheaves of spectra<d-cite key="bunke2016"></d-cite>, and Freed, Moore and
Segal show that the refinement has physical consequences for fluxes in string theory<d-cite key="freed2007"></d-cite>.

So the answer to the question "is there a (co)limit description?" is yes, twice: a homotopy limit that glues curvature to topology, and a homotopy
colimit (a gauge quotient) followed by sheafification that glues local potentials into global ones. In both, the important step is keeping
identifications as data instead of collapsing them to equalities.

## Back to representations

The electromagnetic story suggests a checklist for describing what a model represents.

- **Averages are curvature-level data.** A global mean direction, like $$[F]$$, is a coarse invariant. Two quite different internal geometries can
  share it.
- **Context charts and transports are connection-level data.** Local descriptions plus the maps between them carry more information than any
  single global summary, in the same way that a connection carries more than its curvature.
- **Holonomy can be non-zero when local data look trivial.** If you go around a loop of contexts and come back with a different representation,
  that is an obstruction no amount of averaging will detect. This is the analogue of the Aharonov–Bohm phase, and it is what the holonomy
  obstruction in Javidnia's atlas<d-cite key="javidnia2026gauge"></d-cite> is designed to find.
- **Equivalences should be kept as data.** Paraphrases, like gauge transformations, should be glued with their identifications rather than
  quotiented away too early, which is the presheaf to sheaf step.

Whether trained networks actually have non-trivial holonomy in a meaningful sense is an empirical question, and I think a good one. Physics at
least tells us what the right invariants look like, and that the naive ones can miss exactly the effects we care about.
