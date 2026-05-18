---
layout: post
title: Transformer brain encoders explain human high-level visual responses
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
paper_url: https://arxiv.org/abs/2505.17329
---

## Motivation 

To study neural computations, researchers train deep learning models on a variety of tasks and compare the learned representations to brain activity. One common approach to make this comparison is to train an **encoding model** to map from one feature space to another and measure accuracy.

Both brains and effective deep learning models use structured retinotopic maps (different parts of visual input mapping to different features), but maintaining this mapping when encoding from deep learning models to neural data is challenging with the standard interpretable choice of linear models. This paper focuses on **designing an encoding model that maintains a structured retinotopic map while being as interpretable as a linear model**.

## What they did

A natural solution is cross attention. Learn fixed queries for brain ROIs and produce image patch keys and values from a frozen backbone. A transformer decoder then maps these features to brain responses. In the last step, each ROI query is linearly mapped to a vector the length of the number of vertices, and all vertices not in that query are masked out (important for training so that gradients from unrelated vertices don't flow through).

## Interesting findings

Setup:
- Dataset: Natural Scenes Dataset (7T fMRI), 4 subjects, 10,000 images each
- Backbone Models: DINOv2, ResNet50, CLIP large

The method beats all previous methods on all subjects with a small parameter count. Additionally, cross attention allows for visualizing interpretable attention maps. The attention maps show the patterns we expect to see from different ROIs.