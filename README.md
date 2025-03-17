## [Why Train Everything? Tint a Single Layer for Multi-task Model Merging](https://arxiv.org/abs/2412.19098)

> [Aecheon Jung<sup>1*](https://github.com/kasurashan), [Seunghwan Lee<sup>1](https://github.com/nomis911), [Dongyoon Han<sup>2</sup>&dagger;](https://dongyoonhan.github.io/), [Sungeun Hong<sup>1</sup>&dagger;](https://www.csehong.com/) <br>
> <sup>1</sup>[Sungkyunkwan University](https://www.skku.edu/eng/index.do), <sup>2</sup>[NAVER AI LAB](https://naver-career.gitbook.io/en/teams/clova-cic/ai-lab)

<p align="center">
<img width="600" alt="image" src="https://github.com/user-attachments/assets/95dfec65-d1e0-4a97-9359-e252f6470192")
>
</p>

### Abstract
>Model merging integrates independently fine-tuned models into a single multi-task model, offering a flexible alternative to joint training. However, many existing model merging methods introduce additional task-specific components, increasing complexity and requiring extra modifications. We propose Model Tinting, a lightweight yet highly effective approach that improves model merging by updating just a single layer, accounting for as low as 0.5% of total parameters. Our key observation is that explicit task-specific modules are not necessary; instead, subtle adjustments to a single layer can effectively capture task-specific variations within the merged model while maintaining generalization. We introduce a confidence-based filtering mechanism to alleviate the impact of unreliable predictions from individual models on the merged model. Extensive experiments across vision and NLP tasks demonstrate that Model Tinting achieves state-of-the-art performance, even in challenging dense prediction tasks.


<p align="center">
<img width="600" alt="image" src="https://github.com/user-attachments/assets/45887493-2520-4acb-973b-ec4b0cfdd8ef">
</p>

***Model Tinting*** is a test-time method for merging multiple fine-tuned models by introducing a single trainable layer for each task. This layer adapts task-specific information from the task-agnostic representations of the merged encoder. The method supports using any layer for task adjustments, as demonstrated by additional results in the above figure. 

### Checkpoints
* ViT-B/32 8,14,20 tasks checkpoints for open_clip==2.24.0 [Link](https://huggingface.co/kasurashan/checkpoints_tint)
* ViT-B/32 8,14,20 tasks checkpoints for open_clip==2.0.2 [Link](https://huggingface.co/kasurashan/checkpoints_tint_2-0-2)
* ViT-B/16 8 tasks and ViT-L/14 8 tasks checkpoints for open_clip==2.0.2 [Link](https://github.com/mlfoundations/task_vectors?tab=readme-ov-file#checkpoints)

### Updates
* (2024/12/26): [Preprint](https://arxiv.org/abs/2412.19098) has been uploaded.

### Acknowledgement
We acknowledge the following code, which served as a reference for our implementation.
- https://github.com/mlfoundations/task_vectors 
- https://github.com/EnnengYang/AdaMerging
- https://github.com/yule-buaa/mergelm
- https://github.com/harveyhuang18/emr_merging
