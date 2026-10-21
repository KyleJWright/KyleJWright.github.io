---
layout: page
permalink: /notes/
title: notes
description: Working notes written as part of my move into AI safety and alignment research.
nav: true
nav_order: 4
---

<a href="/assets/pdf/machines-preferences-world-models.pdf">
  <img src="/assets/img/notes-mpwm-cover.png" alt="Title page of Machines, Preferences, and World Models" class="notes-cover">
</a>

## [Machines, Preferences, and World Models](/assets/pdf/machines-preferences-world-models.pdf)

_An introduction to prediction, preference, and representation_

**Working draft** · Prologue, Introduction and Chapters 1–2 · last updated 21 October 2026 ·
[PDF, 49 pages](/assets/pdf/machines-preferences-world-models.pdf)

A trained language model is a fixed function: it reads a string of tokens and returns an output. Whatever it has learned
about the world must be stored somewhere in it. These notes ask what such a model must store to behave as it does, what
preferences, utilities or objectives its behaviour reveals, and how much of either can be recovered from behaviour
alone.

The notes start from the barest description of an AI system, a _machine_: a function $$f : A^* \to B$$ from strings of
tokens to outputs. Everything else is added one condition at a time, at the point where it is needed, and each
condition is recorded in a ledger.

- **Part I, machines and what they represent:** state machines and minimal realisations; why predictors output
  probability distributions, and what else they could output (sets, rankings, weights in a semiring), with the
  conditions that single out probability one at a time; stochastic processes and the mixtures behind in-context
  learning; entropy; hidden Markov representations; when storing a model beats memorising data; and which
  representation behaviour fixes.
- **Part II, what makes a machine good:** from pairwise preferences to utilities, beliefs and expected utility, training
  objectives, and proper scoring rules.
- **Part III, world models and decisions:** an outline of open questions on how a machine that plans could represent a
  world model and a utility of its own, where the alignment question becomes precise.

The notes are written for readers with a PhD-level background in mathematical physics, comfortable with abstract algebra
and categorical language; no background in machine learning, decision theory or economics is assumed.

This version contains the Prologue, the Introduction, Chapter 1, _Machines and their realisations_, and Chapter 2,
_Judgements_. Further chapters
will be added as they are finished. These are working notes, not a peer-reviewed text, and may contain errors.
