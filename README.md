## [SyMerge: From Non-Interference to Synergistic Merging via Single-Layer Adaptation](https://arxiv.org/abs/2412.19098)

> [Aecheon Jung<sup>1*](https://sites.google.com/view/kasurashan), [Seunghwan Lee<sup>1](https://nomis911.github.io/), [Dongyoon Han<sup>2</sup>&dagger;](https://dongyoonhan.github.io/), [Sungeun Hong<sup>1</sup>&dagger;](https://www.csehong.com/) <br>
> <sup>1</sup>[Sungkyunkwan University](https://www.skku.edu/eng/index.do), <sup>2</sup>[NAVER AI LAB](https://naver-career.gitbook.io/en/teams/clova-cic/ai-lab)

[![paper](https://img.shields.io/badge/arXiv-Paper-red.svg)](https://arxiv.org/abs/2412.19098)
[![Project Page](https://img.shields.io/badge/Project-Page-blue?logo=github)](https://aim-skku.github.io/SyMerge/)

![](https://raw.githubusercontent.com/AIM-SKKU/SyMerge/refs/heads/main/symerge-plot.png)

## Abstract
>Model merging offers an efficient alternative to multi-task learning by combining independently fine-tuned models, but most prior approaches focus mainly on avoiding task interference. We argue instead that the real potential of merging lies in achieving synergy, where tasks enhance one another. Our intuition comes from a pilot study showing that when a classifier trained on one task is paired with the encoder of another, the resulting cross-task performance strongly predicts merge quality. Moreover, adapting even a single task-specific layer can substantially improve this compatibility, suggesting a simple yet powerful lever for synergy. Building on this insight, we introduce SyMerge, a lightweight framework that jointly optimizes one task-specific layer and merging coefficients. To ensure stability without labels, SyMerge employs a robust self-labeling strategy guided by expert model predictions, avoiding the pitfalls of entropy-based adaptation. This minimalist yet principled design achieves state-of-the-art results across vision, dense prediction, and NLP benchmarks, while also producing adapted layers that transfer effectively to other merging methods.


## Datasets
Refer to datasets in the [Tall Masks](https://github.com/nik-dim/tall_masks?tab=readme-ov-file#datasets).

## Checkpoints
* ViT-B/32 8,14,20 tasks checkpoints for open_clip==2.24.0 [Link](https://huggingface.co/kasurashan/checkpoints_tint)
* ViT-B/32 8,14,20 tasks checkpoints for open_clip==2.0.2 [Link](https://huggingface.co/kasurashan/checkpoints_tint_2-0-2)
  - `git lfs install`
  - `git clone https://huggingface.co/kasurashan/checkpoints_tint` 
  - `git clone https://huggingface.co/kasurashan/checkpoints_tint_2-0-2`
* ViT-B/16 8 tasks and ViT-L/14 8 tasks checkpoints for open_clip==2.0.2 [Link](https://github.com/mlfoundations/task_vectors?tab=readme-ov-file#checkpoints)

## Train
Dependencies : refer to [task vectors](https://github.com/mlfoundations/task_vectors) \
`git clone https://github.com/AIM-SKKU/SyMerge` \
`cd SyMerge/src` \
`bash train_symerge.sh`


## Acknowledgement
We acknowledge the following code, which served as a reference for our implementation.
- https://github.com/mlfoundations/task_vectors 
- https://github.com/EnnengYang/AdaMerging
- https://github.com/EnnengYang/RepresentationSurgery
