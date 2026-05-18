---
layout: post
title: Rank-1 LoRAs Encode Interpretable Reasoning Signals
date: '2026-05-11'
rating: 6
rating_label: Excellent, can't unsee it
tags:
- Machine Learning
- Mechanistic Interpretability
- Reasoning Models
tag_data:
- name: Machine Learning
  color: moss
- name: Mechanistic Interpretability
  color: teal
- name: Reasoning Models
  color: rose
paper_url: https://arxiv.org/abs/2511.06739
---

## Motivation 

Put simply, we do not have way to understand the internal mechanisms of reasoning models, especially regarding the improvements they provide over base models. Finetuning introduces changes to model weights that are distributed and while introducing many new behaviors.

## What they did

To make interpretation more tractable, they finetune Qwen-2.5-32B-Instruct's MLP on a dataset of DeepSeek R1 rollouts, but do so using only rank-1 LoRA adapters applied to all three MLP matrices (up_proj, down_proj, and gate_proj) and all four attention matrices (Q, K, V, and O) at every layer (192 MLP + 256 attention = 448 total adapters). 

For an <span class="math-render" data-display="inline" data-math="N\times M"></span> matrix, LoRA adapters contain <span class="math-render" data-display="inline" data-math="A \in \mathbb{R}^{N\times r}"></span> and <span class="math-render" data-display="inline" data-math="B\in\mathbb{R}^{r\times N}"></span> matrix, where <span class="math-render" data-display="inline" data-math="r"></span> is the rank. This means that in this implementation, each adapter just stores an <span class="math-render" data-display="inline" data-math="N"></span> dimensional vector and an <span class="math-render" data-display="inline" data-math="M"></span> dimensional vector, and the "activations" of a LoRA adapter take a very practical form. A weight matrix <span class="math-render" data-display="inline" data-math="W"></span> will be adapted to the form <span class="math-render" data-display="inline" data-math="W&#x27;=W+\Delta W"></span>, <span class="math-render" data-display="inline" data-math="\Delta W = BA"></span> such that the LoRA contribution given an input is <span class="math-render" data-display="inline" data-math="\Delta Wx = BAx = sB, s\in \mathbb{R}"></span>. Intuitively, <span class="math-render" data-display="inline" data-math="s=Ax"></span> is a scalar that captures "how strongly is the feature activated?" while B captures "what direction do we push to the hidden state"?. Importantly, this means the activation state is captured entirely by a single scalar <span class="math-render" data-display="inline" data-math="s"></span> per token. These individual adapter directions prove to have interpretable properties in their raw form and are additionally analyzed by a cross-layer SAE (batch-top-k SAE with k=16 and an expansion factor of 8 - contains around 2k features after filtering dead latents). Unlike other papers which must use a compressive SAE to capture interpretable reasoning mechanisms, this paper studies LoRA activations that correspond only to reasoning training, so all SAE features are reasoning-specific even with a larger latent size.

## Interesting findings

- Rougher direction interpretation:
	- LoRA activations are roughly around as monosemantic as MLP neurons, but tend to encode different feature categories.
	- MLP neurons tend to encode more domain-specific and function words, while LoRA activations encode more answer, solution, instruction, procedural, and mathematical markers/symbols.
- Cross-layer SAE interpretation:
	- Features tend to concentrate on mathematical operators and syntax, numbers, formatting tokens(view Appendix A.3 for interesting examples).
	- 62% of SAE features are cleanly monosemantic, up from 22% of LoRA features reasoning control-flow.
	- Component ablation shows that mid-to-late layers have the greatest effect on downstream KL and that MLP components show significantly larger impact than attention components.
	- Removing all attention adapters decreases performance but still outperforms the base model, while removing all MLP adapters causes severe degradation.

 The rank-1 LoRA recovers 73-90% of reasoning benchmark performance compared to a full-parameter finetune while only modifying ~0.03% as many trainable parameters. This implies that *reasoning capabilities may arise from and can be captured by minimal, interpretable parameter changes*.