## [Tint Your Models Task-wise for Improved Multi-task Model Merging]

> [Aecheon Jung<sup>1*](https://github.com/kasurashan), [Seunghwan Lee<sup>1](https://github.com/nomis911), [Dongyoon Han<sup>2</sup>&dagger;](https://dongyoonhan.github.io/), [Sungeun Hong<sup>1</sup>&dagger;](https://www.csehong.com/) <br>
> <sup>1</sup>[Sungkyunkwan University](https://www.skku.edu/eng/index.do), <sup>2</sup>[NAVER AI LAB](https://naver-career.gitbook.io/en/teams/clova-cic/ai-lab)

<p align="center">
<img width="600" alt="image" src="https://github.com/user-attachments/assets/0e7e8f5f-e947-4f37-91df-90087a525a39">
</p>

### Abstract
>Traditional model merging methods for multi-task learning (MTL) address task conflicts with straightforward strategies such as weight averaging, sign consensus, or minimal test-time adjustments. This presumably counts on the assumption that a merged encoder still retains abundant task knowledge from individual encoders, implying that its shared representation is sufficiently general across tasks. However, our insight is that adding just a single trainable task-specific layer further can bring striking performance gains, as demonstrated by our pilot study. Motivated by this finding, we propose Model Tinting, a new test-time approach that introduces a single task-specific layer for each task as trainable adjustments. Our method jointly trains merging coefficients and task-specific layers, which effectively reduces task conflicts with minimal additional costs. Additionally, we propose a sampling method that utilizes the difference in confidence levels of both merged and individual encoders. Extensive experiments demonstrate our method's effectiveness, which achieves state-of-the-art performance across both computer vision and natural language processing tasks and significantly surpasses prior works. Our code will be open-sourced.


<p align="center">
<img width="600" alt="image" src="https://github.com/user-attachments/assets/45887493-2520-4acb-973b-ec4b0cfdd8ef">
</p>

***Model Tinting*** is a test-time method for merging multiple fine-tuned models by introducing a single trainable layer for each task. This layer adapts task-specific information from the task-agnostic representations of the merged encoder. The method supports using any layer for task adjustments, as demonstrated by additional results in the above figure. 

### Updates
* (2024/xx/xx): [Preprint] has been uploaded.

### Acknowledgement
https://github.com/mlfoundations/task_vectors
https://github.com/EnnengYang/AdaMerging
https://github.com/nik-dim/tall_masks
