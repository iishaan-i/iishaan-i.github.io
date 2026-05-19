---
layout: post
title: Brain Mapping with Dense Features - Grounding Cortical Semantic Selectivity
  in Natural Images With Vision Transformers
date: '2026-05-07'
rating: 5
rating_label: Great, a deep read is well worth it
tags:
- Machine Learning
- Neuroscience
- Encoding Models
tag_data:
- name: Machine Learning
  color: moss
- name: Neuroscience
  color: teal
- name: Encoding Models
  color: plum
paper_url: https://arxiv.org/abs/2410.05266
---

## Motivation 

To understand the visual system, researchers typically used controlled images, entirely hand crafted or objects against noise backgrounds that allow researchers to isolate neural patterns. However, this practice limits the relevance of any conclusions because in real life we experience richer and more complex scenes.

Ideally, we would be able relate the effects of different parts of a complex scene to different neural patterns, but it is unclear how to carry out this disentanglement. This paper approaches the problem of specifically **generating spatial attribution maps for arbitrary voxels in the higher visual cortex**.

## What they did

BrainSAIL's approach relies on using fMRI encoders, where a backbone model is frozen and a linear model is trained to predict fMRI activations from images. In particular, they rely on ViT backbones (they explain that by leveraging vision transformers trained on massive datasets of hundreds of millions of images, they can avoid the known limitations of convolutional backbones, which can lead to overfitting to the dataset’s specific features and exacerbate the inherent biases of convolutional networks).

By default, CLIP produces a CLS token and patch tokens that integrate information into the CLS token during attention computations, but only the CLS token is directly supervised, so the patch tokens are never directly trained to be semantically meaningful or spatially aligned. This means for the purpose of segmentation, you can't just query patch tokens against embeddings and expect coherent results.

This paper applies the Neighbor-Aware CLIP[^naclip] (NACLIP) adapter, which uses a modified form of CSA (Correlative Self-Attention, introduced in Spatially-Aware CLIP[^sclip] (SCLIP)). Instead of attending using <span class="math-render" data-display="inline" data-math="q_j"></span>​ (the query of patch <span class="math-render" data-display="inline" data-math="j"></span>) against keys <span class="math-render" data-display="inline" data-math="k_i"></span>​ (which creates cross-patch query-key interactions that scatter attention broadly), SCLIP replaces the standard <span class="math-render" data-display="inline" data-math="q_j k_i^T"></span> dot product with **<span class="math-render" data-display="inline" data-math="q_i q_j^T"></span>**, which measures similarity between queries instead of query-key pairs. In CLIP's ViT, the key and query projections are trained to implement contrastive global matching, not spatial consistency. But **queries tend to cluster by semantic visual content**. So <span class="math-render" data-display="inline" data-math="q_i q_j^T"></span> ​measures "how semantically similar is patch <span class="math-render" data-display="inline" data-math="i"></span> to patch <span class="math-render" data-display="inline" data-math="j"></span>?" rather than "how much should patch <span class="math-render" data-display="inline" data-math="j"></span> attend to <span class="math-render" data-display="inline" data-math="i"></span> globally?" This makes the attention map spatially coherent because similar patches cluster together, producing much cleaner segmentation boundaries.

NACLIP adds a **spatial attentive bias** <span class="math-render" data-display="inline" data-math="\omega_j​"></span> per patch that biases attention toward nearby patches. The motivation is that CSA alone can still produce attention maps that are semantically clustered but not spatially smooth, so patches far apart but visually similar can contaminate each other. This combination of *semantic coherence* (from CSA) and *spatial locality* (from NACLIP's <span class="math-render" data-display="inline" data-math="\omega_j"></span>​ modification) produces the cleanest dense features.

For ViTs that use register tokens (a newer architectural choice that uses a set of learned input tokens to give the model places to store, process and retrieve global information, as opposed to using redundant tokens[^registers]), the paper uses MaskCLIP[^maskclip], which simply outputs the value feature for each patch token directly.

**Formula Recap**
Non-Register Adapters:
- Original Self Attention: <span class="math-render" data-display="inline" data-math="\text{Out}^{\text{Orig}}_j = f\left(\sum_k \text{softmax}( \frac{q_j k^T}{C})_j \cdot v_k\right)"></span>
- Spatially-Aware CLIP (SCLIP): <span class="math-render" data-display="inline" data-math="\text{Out}^{\text{CSA}}_j = f\left(\sum_k \text{softmax}( \frac{q_j q^T}{C})_j \cdot v_k\right)"></span>
- Neighbor-Aware CLIP (NACLIP): <span class="math-render" data-display="inline" data-math="\text{Out}^{\text{NA}}_j = f\left(\sum_k \text{softmax}( \frac{q_j q^T + \omega_j}{C})_j \cdot v_k\right)"></span>

Register Adapters:
- MaskCLIP: <span class="math-render" data-display="inline" data-math="\text{Out}^{\text{NA}}_j = f\left(v_j\right)"></span>

Even NACLIP's dense features have spatial artifacts (spurious high or low activations in patches that don't reflect true semantic content) because no patch token was ever directly supervised, so the features are noisy. The authors find a way to reduce noise in the most statistically classical way possible: an average. The key insight (or assumption) is that **visual semantics are equivariant to shifts and horizontal flips**. Therefore, they can apply the following procedure:

1. Generate <span class="math-render" data-display="inline" data-math="n"></span> augmented views of the image (crops with horizontal/vertical offsets + random flips)
2. Extract dense NACLIP features for each view
3. Track image-space coordinates so they know which augmented patch corresponds to which location in the original image
4. Average the features for each spatial location across all views

The spatial artifacts are averaged out and only the true semantic content remains. This is the final method they use to localize the semantic selectivity of different brain regions.

They also make a formal argument that averaging <span class="math-render" data-display="inline" data-math="n"></span> noisy features vectors finds the optimal representative feature vector under MSE, but this is just a classic fact from statistics (minimizing MSE corresponds to finding the MLE under gaussian noise, which takes the form of a mean - or in an even simpler phrasing, the sample mean is the least-squares estimator). I would argue that the choice of MSE is somewhat arbitrary: I don't know if there is a particular reason to believe that CLIP artifact noise is isotropic Gaussian. In fact, I would probably assume attention-based features tend to be structured, potentially clustering at certain patch positions (e.g. corners, edges of the crop), correlating across feature dimensions, and having non-Gaussian heavy tails. 

I still don't disagree with the design choice at all: ViT patch features depend on both content and absolute positional embeddings, so artifacts, which are driven by positional-embedding-dependent attention patterns in unsupervised patch tokens, vary as a function of absolute position. Under crop augmentations, the same visual content is mapped to different absolute positions across views, so we can write <span class="math-render" data-display="inline" data-math="\vec{p}_i = \vec{p}^* + \epsilon_i"></span> where <span class="math-render" data-display="inline" data-math="\vec{p}^*"></span> is content-dependent and <span class="math-render" data-display="inline" data-math="\epsilon_i"></span> is positional-artifact noise. If the augmentations are applied such that they cancel on average (i.e., the augmentation distribution is symmetric), then <span class="math-render" data-display="inline" data-math="\mathbb{E}[\epsilon_i] = 0"></span>  by construction, making the sample mean an unbiased estimator of <span class="math-render" data-display="inline" data-math="\vec{p}^*"></span> whose variance decreases with <span class="math-render" data-display="inline" data-math="n"></span>. I find this to be cleaner without using a claim about a specific loss that assumes isotropic Gaussian noise. I find the MSE framing weird.

## Interesting findings

Setup:
- Dataset: Natural Scenes Dataset (7T fMRI), 4 subjects, 10,000 images each
- Backbone models: CLIP, DINOv2, and SigLIP

Distilled dense CLIP features achieve SOTA zero-shot open-vocabulary segmentation on ADE20k, COCO Object, and COCO Stuff, consistently outperforming SCLIP and NACLIP baselines, but are outperformed by SCLIP on VOC20.

UMAP of encoder weights, without using any functional localizer labels, recover the known cortical organization: body regions (EBA), face regions (FFA/aTL-faces), place regions (RSC/OPA/PPA), and food regions (flanking FFA) emerge in a data-driven manner. CLIP text-alignment of BrainSAIL relevance maps consistently confirm selectivity (for example, place-selective regions align the strongest alignment for images in the place category and far lower alignment for other categories - a pattern that exists across all categories).

The method also successfully identifies known scene selective regions (RSC/OPA/PPA) as preferring high depth, and is successful even in OPA where a previous convolutional backbone method fails[^conv].

All three backbones achieve comparable fMRI prediction R² on NSD test sets, consistent with previous work[^previous]. Despite similar overall R², qualitative grounding differences are substantial: DINO (image-only supervision) is more sensitive to low-level visual similarity and less semantically coherent (e.g., misses pizza toppings, less face-part specificity). CLIP and SigLIP (both language-supervised) produce more mutually similar relevance maps than either does to DINO, confirming that language supervision drives higher-level semantic alignment between model and brain.

[^naclip]: [Pay Attention to Your Neighbours](https://arxiv.org/abs/2404.08181)
[^sclip]: [SCLIP: Rethinking Self-Attention for Dense Vision-Language Inference](https://arxiv.org/abs/2312.01597)
[^registers]: [Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588)
[^maskclip]: [Extract Free Dense Labels from CLIP](https://arxiv.org/abs/2112.01071)
[^conv]: [Brain Dissection: fMRI-trained Networks Reveal Spatial Selectivity in the Processing of Natural Images](https://www.biorxiv.org/content/10.1101/2023.05.29.542635v1)
[^previous]: [Natural language supervision with a large and diverse dataset builds better models of human high-level visual cortex](https://www.biorxiv.org/content/10.1101/2022.09.27.508760v2)
