---
layout: post
title: Eccentricity-Constrained CNN Training Reveals Adaptive Information Coding Around
  the Visual Field
date: '2026-05-09'
rating: 5
rating_label: Great, a deep read is well worth it
tags:
- Machine Learning
- Neuroscience
- Encoding Models
- Visual Field Eccentricity
tag_data:
- name: Machine Learning
  color: moss
- name: Neuroscience
  color: teal
- name: Encoding Models
  color: plum
- name: Visual Field Eccentricity
  color: mauve
---

## Motivation 

Visual field eccentricity (distance of a stimulus from the center of gaze) is a known organizing dimension of the primate visual system. There are known topographic maps of preferred eccentricity found across early and higher visual cortex, which align with representations mid-level features (curvature, spatial frequency) and semantic categories (faces, words, scenes).

In particular:
- regions coding the center of visual space tend to have **higher resolution** and be selective for **curved contours**, as well as overlapping with **face-selective** and **word-selective** regions.
- regions preferring the periphery tend to overlap with **scene-selective** visual regions.

Two common (and not disjoint) accounts for this organization are:
- A reflection of the advantages for different behavioral tasks arising from the **retinotopic organization** (topographic mapping of visual space onto the cortex) of the central and peripheral visual fields:
	1. The central visual field has **small spatial receptive fields** and **high acuity**, so it is likely more useful for **fine-grained tasks** like face recognition and reading.
	2. More **global tasks** like scene recognition may benefit from the **larger receptive fields** of neurons in peripheral-coding regions.
- A result due to portion of retinotopic space in which face, word, and scene content typically occur (e.g., faces are often in the center of the visual field). In other words, in natural egocentric inputs, **image statistics differ** between the central and peripheral portions of the visual field, and if the **visual system is adapted to these statistics**, this eccentricity-specific coding could have **benefits for downstream tasks** like face and scene recognition.

There are many interesting questions here. This paper focuses on the problem of **testing if eccentricity-specific coding emerges from natural experience**.

## What they did

To actually test this hypothesis, it is necessary to use a large-scale naturalistic egocentric dataset that incorporate eye movement behavior, so they used the Visual Experience Dataset (VEDB). They trained CNNs that were constrained by eccentricity by extracting VEDB frames and created four conditions:
- Baseline: unmodified 224x224 video frame
- Fovea-Gaze: A 112×112 pixel crop taken from each frame, centered on the participant's tracked gaze position. The crop was then upsampled back to 224×224. The region outside the crop was masked with gray. Despite the name, they specifically note not to interpret this as a simulation of biological fovea.
- Periph: The complement of Fovea-Gaze. The full frame is kept, but the central region around the gaze point is masked out with a gray circle.
- Periph-NF: Built identically to Periph, but with one extra step first: before masking out the foveal region, the entire frame undergoes the NeuroFovea[^nf] transform: a style-transfer model that progressively blurs and "texturizes" the image the further it is from the gaze center, mimicking the reduced acuity of real peripheral vision. Then the foveal region is masked out as before.
	- NeuroFovea takes an input image and passes it through a VGG-Net encoder, then for each pooling region in the visual field (centered on a fixation point), it interpolates in feature space between the original image content and a texture-matched noise patch, with the blend shifting progressively toward texture as distance from the fovea increases. The resulting blended representation is then decoded back to pixel space, producing an image that looks sharp at the fixation point but increasingly blurred and "texturized" in the periphery, mimicking how human peripheral vision loses spatial detail.

They pretrained ResNet-18 encoders using SimCLR with a variety of data augmentations, and used pretrained ResNet-18 models as baselines.

Performance is measured in two ways:
- Downstream Classification with Linear Probes: Frozen ResNet-18 backbone with a linear probe evaluated by Macro-F1 (class-balanced, metric primary metric for in-domain due to label imbalance) and Top-1/Top-5 accuracy (for VGGFace2 and Places365 transfer tasks).
- Brain Alignment using fMRI Encoding Models: Cross-validated <span class="math-render" data-display="inline" data-math="R^2"></span> measuring how well model activations predict held-out voxel responses in the Natural Scenes Dataset. Variance partitioning (unique <span class="math-render" data-display="inline" data-math="R^2"></span>) was used to isolate contributions of individual models.

## Interesting findings

Downstream Task Performance:
- Fovea-Gaze outperformed peripheral models on in-domain classification (Macro-F1: 43.64% vs. 36.56% Periph and 30.93% Periph-NF), with a notable advantage on fine-grained/social tasks like "playing video game" and "socializing," while peripheral models were better for outdoor/large-scale tasks like "skateboarding" and "playing frisbee."
- VEDB-pretrained models were competitive with ImageNet-pretrained models on in-domain classification, with Fovea-Gaze approximately matching ImageNet-1K and Baseline approximately matching ImageNet-100, while generally outperforming STL-10.
- All VEDB models generalized better to scene recognition (Places365) than face recognition (VGGFace2), likely reflecting limited face content in VEDB.
- Non-VEDB models outperformed all VEDB models on VGGFace2.

fMRI Encoding:
- VEDB Baseline achieved comparable neural predictivity to ImageNet-100, consistently exceeding STL-10 and approaching ImageNet-1K.
- The Periph model significantly outperformed Fovea-Gaze in scene-selective regions PPA and RSC (both p < 0.01), while Fovea-Gaze significantly outperformed Periph in primary visual cortex V1 (p < 0.01).
- Variance partitioning confirmed that the Periph vs Fovea-Gaze dissociations in scene-selective regions (particularly PPA and RSC) were consistent across all 8 participants.
- The Periph-NF model explained more unique variance in V1 than Periph, while Periph explained more unique variance in scene-selective OPA and PPA than Periph-NF.
- Fovea-Gaze and Periph-NF showed no significant differences across any ROI, suggesting texturized peripheral input behaves more like foveal input in terms of neural alignment.

[^nf]: [Towards Metamerism via Foveated Style Transfer](https://arxiv.org/abs/1705.10041)