# Multitask Topic Classification and Headline Generation for Nigerian Languages

## Overview

This project fine-tunes a single sequence to sequence model to perform two tasks at once on news articles written in four Nigerian languages: Hausa, Igbo, Nigerian Pidgin, and Yoruba.

- **Task A: Topic classification.** Assign each article one of seven topics (business, entertainment, health, politics, religion, sports, technology).
- **Task B: Headline generation.** Generate a headline for each article.

The final score is a blend of both tasks: 0.5 x macro-F1 (Task A) + 0.5 x GenScore, where GenScore itself is 0.5 x ROUGE-L + 0.5 x BERTScore (Task B).

A hard constraint applied throughout: the model must stay under 1 billion parameters.

## Model

**castorini/afriteva_v2_base** (659.5M parameters)

This was chosen as the best available base model under the 1B parameter limit for this task, for a few reasons:

- It is pretrained on African languages, so its tokenizer already has reasonable subword coverage for Hausa, Igbo, Yoruba, and Pidgin, unlike a generic multilingual model.
- It is an encoder-decoder architecture, so both classification and generation can be framed as text-to-text problems using the same model and vocabulary.
- At 659.5M parameters it leaves headroom under the 1B cap while still being small enough to fully fine-tune on a single T4 GPU.

## Approach

**Multitask framing.** Every training example is converted into two records, one per task, distinguished by a prefix on the input source text: `topic <language>: <article>` mapped to the topic label, and `headline <language>: <article>` mapped to the headline. One encoder, one decoder, one training loop covering both tasks.

**Class balancing.** Topic labels are heavily imbalanced (technology has 203 combined examples, entertainment and politics have over 1250 each). Minority classes are oversampled up to 1000 examples each before training.

**Constrained decoding.** Task A uses a custom logits processor that masks the output vocabulary at every decoding step, restricting generation to the valid label set for that specific language. This guarantees every topic prediction is a legal label.

**Extractive fallback for headlines.** If a generated headline comes out empty or under three words, the pipeline falls back to extracting the lead sentence from the source article.

**Training setup.** Full fine-tuning with Adafactor (chosen for its smaller memory footprint versus Adam), batch size 2 with gradient accumulation of 8 (effective batch 16), learning rate 1e-3 with 200 warmup steps and a cosine schedule, weight decay 0.01, label smoothing 0.08, 3 epochs, gradient checkpointing enabled.

**Stochastic weight averaging (SWA).** Checkpoints saved at 2.5, 2.75, and 3.0 epochs, with final weights averaged across the three.

## Results

| Metric | Zero-Shot | Fine-Tuned |
|---|---|---|
| Task A Macro-F1 | 0.171 | 0.887 |
| Task B GenScore | 0.424 | 0.514 |
| Final Score | 0.297 | 0.701 |

Per-language macro-F1 (fine-tuned): Yoruba 0.926, Hausa 0.913, Igbo 0.889, Nigerian Pidgin 0.860.

## What helped

- **Language-tailored backbone.** AfriTeVa v2's African-language pretraining gave it a real head start on subword coverage, and it was the best base model available under the 1B parameter constraint.
- **Single multitask model instead of two separate models.** Prefix-based task conditioning let one model share representations across classification and generation rather than training two pipelines independently.
- **Minority class oversampling.** Flattening topic classes to 1000 examples each before training kept macro-F1 from collapsing on the rare classes (technology in particular).
- **Constrained decoding for Task A.** Masking the vocabulary to only valid per-language labels removed an entire failure mode where the model could generate a plausible but illegal label, which would otherwise cost macro-F1 points for free.
- **Extractive fallback for Task B.** Falling back to the lead sentence when generation degenerates avoided empty or near-empty headlines tanking ROUGE-L and BERTScore on the tail.
- **Adafactor optimizer.** Its lower memory footprint was the difference between fitting a full fine-tune of a 659M model on a single T4 and not fitting one at all.
- **SWA checkpoint averaging.** Averaging the last three checkpoints reduced variance from picking one arbitrary final checkpoint.

## What I would change on another run

- **Tune the learning rate properly.** 1e-3 with a cosine schedule was chosen by intuition, not a sweep. I would run a small sweep and watch validation loss and validation score directly, since Task B convergence looked weaker than Task A's.
- **Turn on evaluation during training.** `eval_strategy` was set to "no", so training ran blind for a fixed 3 epochs with no way to check whether an earlier epoch generalized better than the final one.
- **Fix repeated BERTScore model loading.** The scoring function reloads the full mBERT model from scratch on every call, and it is called separately for the pooled set and each of the four languages, both before and after fine-tuning. That is roughly ten full model loads with no training benefit, and likely a meaningful chunk of total runtime.
- **Actually run the few-shot baseline.** A function was written to test one labeled example per language with no gradient updates, to measure how much of the zero-shot to fine-tuned gap could be closed without training at all, but it was never executed before submitting.
- **Check the BERTScore language argument.** An English language flag was passed alongside a multilingual model type while scoring non-English headlines, which affects baseline rescaling and could be skewing the generation scores.
- **Tune generation hyperparameters against validation scores.** Beam count, length penalty, and repetition penalty were set once and never revisited, even though Task B showed the smallest improvement of the two tasks.
- **Reconsider the oversampling approach.** Flat oversampling to 1000 duplicates the smallest classes five to six times with no augmentation, risking memorization of a small set of repeated examples. A lower duplication ceiling or class-weighted loss would be worth testing instead.
- **Run multiple seeds.** The final score came from a single run, so there is no estimate of how much it might move on its own before treating 0.701 as a stable number to optimize further.

## Constraints

- Model must be under 1,000,000,000 parameters (verified at 659,471,616).
- Fine-tuning performed on a single T4-class GPU.