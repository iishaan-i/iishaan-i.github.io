---
layout: post
title: Base Models Know How to Reason, Thinking Models Learn When
date: '2026-05-11'
rating: 5
rating_label: Great, a deep read is well worth it
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
paper_url: https://arxiv.org/abs/2510.07364
---

## Motivation 

Following the large (and growing) body of research that explores the differences and interplay between different stages of LLM training and the models they produce, one fundamental and unanswered question is what capabilities do reasoning models have that base models don't, and how can we demonstrate that this difference?

Earlier explanations include ideas such as:
- RL teaches models new capabilities
- RL teaches models to structure their reasoning more effectively 
- RL teaches models to repurpose pre-existing representations for new mechanisms
- additional inference time allows for more computation

This paper provides interesting evidence for a tighter claim: base models already possess reasoning capabilities, but reasoning models learn *when* to deploy these capabilities in a response.

## What they did

For their analysis, rather than presenting manual inspection of the model’s reasoning traces to identify the underlying mechanisms it uses to perform reasoning, they start by automating the creation of a "reasoning mechanism taxonomy" with the help of LLM labeling.

They first trained Top-K Sparse SAEs with small latents on activations from MMLU-Pro reasoning model traces that were sentence-averaged so that the latents would naturally form clusters representing fundamental reasoning mechanisms. It is worth noting that SAEs are usually trained with latents that are far larger than the LLM hidden dimension, but that methodology is useful for capturing more fine-grained knowledge concepts and facts, while latents that actually compress the representation (even though it is just a single linear down projection) are generally necessary to capture reasoning mechanisms.

They use an LLM to label the clusters by cognitive function based on the top activated sentences in that cluster, and then score clusters for:
- completeness: confidence that an LLM has in classifying sentences into their assigned categories
- consistency: how well an LLM can classify sentences from within and outside each category using the generated titles and descriptions
- independence: asking an LLM to evaluate how semantically similar all pairs of categories are in a cluster
to choose the clustering hyperparameters that created the best reasoning mechanism taxonomy.

They then used the identified reasoning mechanisms to create steering vectors that activate these behaviors in the *base models* (I do not discuss the steering vector optimization here, I felt it was not relevant). These steering vectors were applied to the base models to elicit the associated reasoning behaviors, *but only whenever the reasoning model used them*. This meant they were *inputting* the base model's *token outputs* to the reasoning model at every step and inspecting the reasoning model's activations each time to see if the reasoning model determined it was time to use a reasoning mechanism. If the reasoning model's activations indicated it was time (activations align with a reasoning mechanism cluster), they applied the corresponding steering vector to the base model. This is not a method that is created to provide value in practice (it depends on having a reasoning model in the first place and requires forward passes from both models at every step - they do mention the idea of steering windows, allowing them to potentially run the reasoning model less frequently, not unlike speculative decoding, but they don't use this for their main results), but rather prove a hypothesis.

## Interesting findings

- Recovered 91% of the performance gap between the QwQ-32B (reasoning) and Qwen2.5-32B (base) on MATH500 while steering only 12% of tokens. This is a convincing example, the original performance gap is 23%.
- They ablate the method by applying the steering vectors at random times, applying the general bias vector for steering without any category or mechanism-specific steering vectors, and applying random unit vectors on top of the bias vector, and all perform notably worse. This strongly suggests that *the mechanism-specific timing the reasoning model has learned to use* is the distinguishing factor (hence the title).
- It appeared that as model size increased, the amount of gap recovery increased.
- The results with other models (1.5b to 32b models on MATH500 and GSM8K) were still positively correlated and showed at least a partial gap recovery (especially around the 50-60% range of 20-30% gaps), but overall it seemed that the hypothesis left quite a lot of the performance still unexplained. However, this paper does a good job by explaining a significant portion.