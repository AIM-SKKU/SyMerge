## [SyMerge: From Non-Interference to Synergistic Merging via Single-Layer Adaptation (ICML 2026)](https://arxiv.org/abs/2412.19098)

> [Aecheon Jung<sup>1*](https://sites.google.com/view/kasurashan), [Seunghwan Lee<sup>1](https://nomis911.github.io/), [Dongyoon Han<sup>2</sup>&dagger;](https://dongyoonhan.github.io/), [Sungeun Hong<sup>1</sup>&dagger;](https://www.csehong.com/) <br>
> <sup>1</sup>[Sungkyunkwan University](https://www.skku.edu/eng/index.do), <sup>2</sup>[NAVER AI LAB](https://naver-career.gitbook.io/en/teams/clova-cic/ai-lab)

[![paper](https://img.shields.io/badge/arXiv-Paper-red.svg)](https://arxiv.org/abs/2412.19098)
[![Project Page](https://img.shields.io/badge/Project-Page-blue?logo=github)](https://aim-skku.github.io/SyMerge/)

![](https://raw.githubusercontent.com/AIM-SKKU/SyMerge/refs/heads/main/symerge-plot.png)

## Abstract
>Model merging combines independently trained models into a single multi-task model. However, most existing approaches focus primarily on avoiding task interference. We argue that its greater potential lies in enabling task synergy, where tasks actively improve one another. We identify cross-task performance, defined by compatibility between encoders and predictors across tasks, as a key indicator of merge quality. We demonstrate that adapting only a single task-specific layer is sufficient to induce such synergy. This study proposes SyMerge, a lightweight framework that jointly optimizes merging coefficients and a single task-specific layer. We adopt an expert-guided self-labeling objective, providing stable supervision beyond entropy minimization. Intriguingly, we further show that SyMerge successfully merges models trained from different initializations, a regime where standard methods break down. Our minimalist yet principled method achieves state-of-the-art results across vision, dense prediction, and NLP benchmarks.


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
